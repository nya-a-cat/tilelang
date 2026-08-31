"""Reproduce the free-Colab T4 screen for callee allocation vs output reuse."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable
import urllib.request


WHEEL_URL = (
    "https://github.com/nya-a-cat/tilelang/releases/download/"
    "colab-wheel-f4842cec1749e9b97530011f5a443158d108b231/"
    "tilelang-0.1.13%2Bcu130.gitf4842cec-cp39-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
WHEEL_SHA256 = "17a5c77dfc2277ee6ca1cb9d62e2eb1f765def3310f203863799671f625abd59"
NATIVE_SOURCE_SHA = "d71f863f4bb5ee261a382343bfc3bef2a722f33c"
WORKFLOW_SOURCE_SHA = "f4842cec1749e9b97530011f5a443158d108b231"
SHAPES = [128, 2048, 4096, 65536, 1048576]
THREADS = 256
CYCLES = 25
MIN_BATCH_SECONDS = 0.05
MAX_BATCH_ITERS = 16384


def run(command: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def install_exact_wheel() -> dict[str, str | int | float]:
    wheel_path = Path(
        "/tmp/tilelang-0.1.13+cu130.gitf4842cec-"
        "cp39-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    started = time.perf_counter()
    urllib.request.urlretrieve(WHEEL_URL, wheel_path)
    actual_hash = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    if actual_hash != WHEEL_SHA256:
        raise RuntimeError(f"wheel hash mismatch: {actual_hash}")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "apache-tvm-ffi==0.1.12",
            "torch-c-dlpack-ext==0.1.5",
            "z3-solver==4.15.4.0",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheel_path)])
    return {
        "url": WHEEL_URL,
        "sha256": actual_hash,
        "bytes": wheel_path.stat().st_size,
        "install_seconds": time.perf_counter() - started,
    }


os.environ["TILELANG_CACHE_DIR"] = "/tmp/tilelang-output-reuse-cache-f4842cec"
wheel = install_exact_wheel()

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(samples: list[dict[str, float | int]]) -> dict[str, float | int]:
    walls = [float(sample["wall_us"]) for sample in samples]
    gpus = [float(sample["gpu_us"]) for sample in samples]
    return {
        "samples": len(samples),
        "wall_p50_us": statistics.median(walls),
        "wall_p90_us": percentile(walls, 0.90),
        "wall_p99_us": percentile(walls, 0.99),
        "gpu_p50_us": statistics.median(gpus),
        "gpu_p90_us": percentile(gpus, 0.90),
        "gpu_p99_us": percentile(gpus, 0.99),
    }


def make_program(length: int, symbol: str):
    @T.prim_func
    def add_one(A: T.Tensor((length,), T.float32), B: T.Tensor((length,), T.float32)):
        with T.Kernel(T.ceildiv(length, THREADS), threads=THREADS) as (bx,):
            for tx in T.Parallel(THREADS):
                index = bx * THREADS + tx
                if index < length:
                    B[index] = A[index] + 1.0

    return add_one.with_attr("global_symbol", symbol)


def compile_pair(length: int):
    started = time.perf_counter()
    callee = tilelang.compile(
        make_program(length, f"output_alloc_{length}"),
        out_idx=-1,
        target="cuda",
        execution_backend="tvm_ffi",
    )
    callee_seconds = time.perf_counter() - started
    started = time.perf_counter()
    preallocated = tilelang.compile(
        make_program(length, f"output_reuse_{length}"),
        out_idx=None,
        target="cuda",
        execution_backend="tvm_ffi",
    )
    preallocated_seconds = time.perf_counter() - started
    return callee, preallocated, callee_seconds, preallocated_seconds


def warm_for_one_second(call: Callable[[], object]) -> int:
    count = 0
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        call()
        count += 1
    torch.cuda.synchronize()
    return count


def measure_batch(call: Callable[[], object], iterations: int) -> tuple[float, float]:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    for _ in range(iterations):
        call()
    end_event.record()
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1e6 / iterations
    gpu_us = start_event.elapsed_time(end_event) * 1e3 / iterations
    return wall_us, gpu_us


def calibrate(call: Callable[[], object]) -> int:
    iterations = 32
    while iterations < MAX_BATCH_ITERS:
        wall_us, _ = measure_batch(call, iterations)
        if wall_us * iterations >= MIN_BATCH_SECONDS * 1e6:
            break
        iterations *= 2
    return min(iterations, MAX_BATCH_ITERS)


def allocation_probe(call: Callable[[], object], pointer: Callable[[object], int]) -> dict[str, int]:
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    pointers: set[int] = set()
    for _ in range(100):
        result = call()
        pointers.add(pointer(result))
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    return {
        "calls": 100,
        "unique_output_addresses": len(pointers),
        "allocation_requests": int(after["allocation.all.allocated"] - before["allocation.all.allocated"]),
        "allocated_bytes_requests": int(after["allocated_bytes.all.allocated"] - before["allocated_bytes.all.allocated"]),
    }


def capture_graph(call: Callable[[], object]):
    for _ in range(5):
        captured_output = call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = call()
    torch.cuda.synchronize()
    return graph, captured_output


def measure_graph(graph: torch.cuda.CUDAGraph, iterations: int = 4096) -> dict[str, float | int]:
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    end_event.record()
    torch.cuda.synchronize()
    return {
        "iterations": iterations,
        "wall_us": (time.perf_counter() - wall_start) * 1e6 / iterations,
        "gpu_us": start_event.elapsed_time(end_event) * 1e3 / iterations,
    }


def nvidia_snapshot() -> str:
    try:
        return run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture=True,
        )
    except Exception as error:  # pragma: no cover - environment evidence path
        return f"unavailable: {type(error).__name__}: {error}"


if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required")

torch.manual_seed(20260831)
torch.cuda.manual_seed_all(20260831)
device = torch.device("cuda")
environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "tilelang": tilelang.__version__,
    "cuda_runtime": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(0),
    "device_capability": list(torch.cuda.get_device_capability(0)),
    "nvidia_smi_start": nvidia_snapshot(),
    "cache_dir": os.environ["TILELANG_CACHE_DIR"],
}

results: list[dict[str, object]] = []
for length in SHAPES:
    print(f"SCREEN length={length}: compiling", flush=True)
    callee, preallocated, callee_compile_s, preallocated_compile_s = compile_pair(length)
    x = torch.randn(length, device=device, dtype=torch.float32)
    reusable_output = torch.empty_like(x)

    last_callee: list[torch.Tensor | None] = [None]

    def call_callee():
        last_callee[0] = callee(x)
        return last_callee[0]

    def call_preallocated():
        preallocated(x, reusable_output)
        return reusable_output

    actual = call_callee()
    call_preallocated()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, x + 1.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reusable_output, x + 1.0, rtol=0.0, atol=0.0)

    warmups = {
        "callee": warm_for_one_second(call_callee),
        "preallocated": warm_for_one_second(call_preallocated),
    }
    iterations = max(calibrate(call_callee), calibrate(call_preallocated))
    raw = {"callee": [], "preallocated": []}
    order = ["callee", "preallocated", "preallocated", "callee"]
    calls = {"callee": call_callee, "preallocated": call_preallocated}
    for cycle in range(CYCLES):
        for order_index, label in enumerate(order):
            wall_us, gpu_us = measure_batch(calls[label], iterations)
            raw[label].append(
                {
                    "cycle": cycle,
                    "order_index": order_index,
                    "wall_us": wall_us,
                    "gpu_us": gpu_us,
                }
            )

    summaries = {label: summarize(samples) for label, samples in raw.items()}
    wall_speedup = summaries["callee"]["wall_p50_us"] / summaries["preallocated"]["wall_p50_us"]
    gpu_speedup = summaries["callee"]["gpu_p50_us"] / summaries["preallocated"]["gpu_p50_us"]

    allocation = {
        "callee": allocation_probe(call_callee, lambda result: result.data_ptr()),
        "preallocated": allocation_probe(call_preallocated, lambda result: result.data_ptr()),
    }
    graph: dict[str, object] = {}
    for label in order[:2]:
        try:
            captured_graph, captured_output = capture_graph(calls[label])
            graph[label] = {
                "status": "success",
                "output_address": captured_output.data_ptr(),
                "measurement": measure_graph(captured_graph),
            }
        except Exception as error:
            graph[label] = {"status": "error", "error": f"{type(error).__name__}: {error}"}

    result = {
        "length": length,
        "bytes": length * 4,
        "compile_seconds": {"callee": callee_compile_s, "preallocated": preallocated_compile_s},
        "warmup_calls": warmups,
        "batch_iterations": iterations,
        "cycles": CYCLES,
        "summaries": summaries,
        "speedup": {"wall_p50": wall_speedup, "gpu_p50": gpu_speedup},
        "allocation_probe": allocation,
        "cuda_graph": graph,
        "raw_samples": raw,
    }
    results.append(result)
    print(
        "SCREEN "
        f"length={length} iterations={iterations} "
        f"wall={summaries['callee']['wall_p50_us']:.3f}->{summaries['preallocated']['wall_p50_us']:.3f}us "
        f"speedup={wall_speedup:.3f}x "
        f"gpu={summaries['callee']['gpu_p50_us']:.3f}->{summaries['preallocated']['gpu_p50_us']:.3f}us",
        flush=True,
    )

environment["nvidia_smi_end"] = nvidia_snapshot()
payload = {
    "schema": "tilelang-output-reuse-t4-v1",
    "experiment": "callee-allocated-vs-caller-preallocated-output",
    "created_unix": time.time(),
    "repository": "nya-a-cat/tilelang",
    "native_source_sha": NATIVE_SOURCE_SHA,
    "workflow_source_sha": WORKFLOW_SOURCE_SHA,
    "wheel": wheel,
    "environment": environment,
    "method": {
        "order": ["callee", "preallocated", "preallocated", "callee"],
        "cycles": CYCLES,
        "warmup_seconds_per_mode_per_shape": 1.0,
        "minimum_batch_seconds": MIN_BATCH_SECONDS,
        "speedup_definition": "median(callee_allocated_wall_us) / median(preallocated_wall_us)",
        "correctness": "exact float32 add-one comparison for both ABIs",
    },
    "results": results,
}
serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload_sha256 = hashlib.sha256(serialized).hexdigest()
compressed = gzip.compress(serialized, compresslevel=9)
encoded = base64.b64encode(compressed).decode()
Path("/content/tilelang-output-reuse-t4.json").write_bytes(serialized)
Path("/content/tilelang-output-reuse-t4.json.gz").write_bytes(compressed)
print(f"RESULT_JSON_SHA256={payload_sha256}")
print(f"RESULT_JSON_BYTES={len(serialized)}")
print(f"RESULT_GZIP_BASE64={encoded}")
