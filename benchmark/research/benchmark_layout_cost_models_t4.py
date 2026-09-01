"""Runtime A/B for TileLang free-mode layout policies on one unchanged PrimFunc.

This is a free-Colab screening benchmark.  Every case is constructed once and
compiled twice; the only changed input to the backend is
``tl.layout_cost_model``.  It measures eager launches and single-kernel CUDA
Graph replay, checks both variants against one PyTorch reference, and records
generated-source hashes plus compiler resource usage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any

import torch
import tilelang
import tilelang.language as T
from tilelang.contrib.kernel_resource_info import to_dict as resource_usage_to_dict


POLICIES = ("register-count", "io-aware")
CYCLES = int(os.environ.get("TILELANG_LAYOUT_AB_CYCLES", "30"))
MIN_BATCH_MS = float(os.environ.get("TILELANG_LAYOUT_AB_MIN_BATCH_MS", "50"))
MAX_BATCH_ITERS = int(os.environ.get("TILELANG_LAYOUT_AB_MAX_BATCH_ITERS", "65536"))
WARM_SECONDS = float(os.environ.get("TILELANG_LAYOUT_AB_WARM_SECONDS", "1"))
RESULT_PATH = Path(os.environ.get("TILELANG_LAYOUT_AB_RESULT", "/content/tilelang-layout-cost-models-t4.json"))


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: str
    program: Any
    inputs: tuple[torch.Tensor, ...]
    output_shape: tuple[int, ...]
    output_dtype: torch.dtype
    reference: torch.Tensor
    atol: float
    rtol: float


@dataclass
class CompiledVariant:
    policy: str
    kernel: Any
    output: torch.Tensor
    call: Any
    graph: torch.cuda.CUDAGraph | None
    graph_call: Any | None
    graph_capture_error: str | None
    compile_seconds: float
    source_sha256: str
    generated_source: str
    resource_usage: dict[str, Any]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50_us": statistics.median(values),
        "p90_us": percentile(values, 0.90),
        "p99_us": percentile(values, 0.99),
        "minimum_us": min(values),
        "maximum_us": max(values),
    }


def ratio_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p10": percentile(values, 0.10),
        "p50": statistics.median(values),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - provenance should survive optional probes
        return f"{type(error).__name__}: {error}"


def with_symbol(program: Any, symbol: str) -> Any:
    return program.with_attr("global_symbol", symbol)


def make_broadcast(rows: int, cols: int, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(S: T.Tensor((rows,), T.float32), Out: T.Tensor((rows, cols), T.float32)):
        with T.Kernel(1, threads=threads):
            scalar = T.alloc_fragment((rows,), T.float32)
            T.copy(S, scalar)
            for i, j in T.Parallel(rows, cols):
                Out[i, j] = scalar[i] * 2.0

    return with_symbol(main, symbol)


def make_transpose(M: int, N: int, dtype: str, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((N, M), dtype)):
        with T.Kernel(1, threads=threads):
            fragment = T.alloc_fragment((M, N), dtype)
            T.copy(A, fragment)
            for i, j in T.Parallel(M, N):
                B[j, i] = fragment[i, j]

    return with_symbol(main, symbol)


def make_mixed_chain(M: int, N: int, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(A: T.Tensor((M, N), T.float16), B: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, threads=threads):
            x16 = T.alloc_fragment((M, N), T.float16)
            x32 = T.alloc_fragment((M, N), T.float32)
            T.copy(A, x16)
            for i, j in T.Parallel(M, N):
                x32[i, j] = x16[i, j].astype(T.float32) * 2.0
            T.copy(x32, B)

    return with_symbol(main, symbol)


def make_rmsnorm(rows: int, width: int, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(A: T.Tensor((rows, width), T.float32), B: T.Tensor((rows, width), T.float32)):
        with T.Kernel(rows, threads=threads) as bx:
            values = T.alloc_fragment((width,), T.float32)
            squares = T.alloc_fragment((width,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            T.copy(A[bx, 0], values)
            for index in T.Parallel(width):
                squares[index] = values[index] * values[index]
            T.reduce_sum(squares, total, dim=0)
            scale = T.rsqrt(total[0] / width + 1e-6)
            for index in T.Parallel(width):
                values[index] *= scale
            T.copy(values, B[bx, 0])

    return with_symbol(main, symbol)


def make_softmax(rows: int, width: int, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(A: T.Tensor((rows, width), T.float32), B: T.Tensor((rows, width), T.float32)):
        with T.Kernel(rows, threads=threads) as bx:
            values = T.alloc_fragment((width,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            T.copy(A[bx, 0], values)
            T.reduce_max(values, maximum, dim=0)
            for index in T.Parallel(width):
                values[index] = T.exp(values[index] - maximum[0])
            T.reduce_sum(values, total, dim=0)
            for index in T.Parallel(width):
                values[index] /= total[0]
            T.copy(values, B[bx, 0])

    return with_symbol(main, symbol)


def make_elementwise(length: int, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(A: T.Tensor((length,), T.float32), B: T.Tensor((length,), T.float32)):
        with T.Kernel(T.ceildiv(length, threads), threads=threads) as bx:
            for tx in T.Parallel(threads):
                index = bx * threads + tx
                if index < length:
                    B[index] = A[index] * 2.0 + 1.0

    return with_symbol(main, symbol)


def build_cases() -> list[BenchmarkCase]:
    device = torch.device("cuda")
    cases: list[BenchmarkCase] = []

    for rows, cols, threads in ((2, 2560, 256), (4, 1024, 128)):
        source = torch.randn(rows, device=device, dtype=torch.float32)
        cases.append(
            BenchmarkCase(
                name=f"broadcast_f32_{rows}x{cols}_t{threads}",
                family="broadcast",
                program=make_broadcast(rows, cols, threads, f"layout_broadcast_{rows}_{cols}_{threads}"),
                inputs=(source,),
                output_shape=(rows, cols),
                output_dtype=torch.float32,
                reference=source[:, None].expand(rows, cols) * 2.0,
                atol=0.0,
                rtol=0.0,
            )
        )

    for M, N, dtype, torch_dtype in (
        (128, 128, "float32", torch.float32),
        (64, 256, "float16", torch.float16),
    ):
        source = torch.randn(M, N, device=device, dtype=torch_dtype)
        cases.append(
            BenchmarkCase(
                name=f"transpose_{dtype}_{M}x{N}_t128",
                family="transpose",
                program=make_transpose(M, N, dtype, 128, f"layout_transpose_{dtype}_{M}_{N}"),
                inputs=(source,),
                output_shape=(N, M),
                output_dtype=torch_dtype,
                reference=source.T.contiguous(),
                atol=0.0,
                rtol=0.0,
            )
        )

    mixed = torch.randn(128, 128, device=device, dtype=torch.float16)
    cases.append(
        BenchmarkCase(
            name="mixed_f16_f32_128x128_t128",
            family="mixed_dtype",
            program=make_mixed_chain(128, 128, 128, "layout_mixed_f16_f32_128_128"),
            inputs=(mixed,),
            output_shape=(128, 128),
            output_dtype=torch.float32,
            reference=mixed.float() * 2.0,
            atol=0.0,
            rtol=0.0,
        )
    )

    norm_input = torch.randn(64, 1024, device=device, dtype=torch.float32)
    cases.append(
        BenchmarkCase(
            name="rmsnorm_f32_64x1024_t128",
            family="rmsnorm",
            program=make_rmsnorm(64, 1024, 128, "layout_rmsnorm_64_1024"),
            inputs=(norm_input,),
            output_shape=tuple(norm_input.shape),
            output_dtype=torch.float32,
            reference=norm_input * torch.rsqrt(norm_input.square().mean(dim=1, keepdim=True) + 1e-6),
            atol=2e-3,
            rtol=2e-3,
        )
    )

    softmax_input = torch.randn(64, 1024, device=device, dtype=torch.float32)
    cases.append(
        BenchmarkCase(
            name="softmax_f32_64x1024_t128",
            family="softmax",
            program=make_softmax(64, 1024, 128, "layout_softmax_64_1024"),
            inputs=(softmax_input,),
            output_shape=tuple(softmax_input.shape),
            output_dtype=torch.float32,
            reference=torch.softmax(softmax_input, dim=1),
            atol=2e-4,
            rtol=2e-3,
        )
    )

    elementwise_input = torch.randn(1 << 20, device=device, dtype=torch.float32)
    cases.append(
        BenchmarkCase(
            name="elementwise_f32_1048576_t256",
            family="control",
            program=make_elementwise(1 << 20, 256, "layout_elementwise_1048576"),
            inputs=(elementwise_input,),
            output_shape=tuple(elementwise_input.shape),
            output_dtype=torch.float32,
            reference=elementwise_input * 2.0 + 1.0,
            atol=0.0,
            rtol=0.0,
        )
    )
    return cases


def compile_variant(case: BenchmarkCase, policy: str) -> CompiledVariant:
    output = torch.empty(case.output_shape, device="cuda", dtype=case.output_dtype)
    started = time.perf_counter()
    kernel = tilelang.compile(
        case.program,
        out_idx=None,
        target="cuda",
        execution_backend="tvm_ffi",
        pass_configs={"tl.layout_cost_model": policy},
    )
    compile_seconds = time.perf_counter() - started

    def call() -> None:
        kernel(*case.inputs, output)

    call()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, case.reference, atol=case.atol, rtol=case.rtol)

    graph: torch.cuda.CUDAGraph | None = None
    graph_call = None
    graph_capture_error = None
    try:
        for _ in range(8):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, case.reference, atol=case.atol, rtol=case.rtol)
        graph_call = graph.replay
    except Exception as error:  # noqa: BLE001 - eager evidence remains valid
        graph_capture_error = f"{type(error).__name__}: {error}"
        graph = None
        graph_call = None

    source = kernel.get_kernel_source()
    return CompiledVariant(
        policy=policy,
        kernel=kernel,
        output=output,
        call=call,
        graph=graph,
        graph_call=graph_call,
        graph_capture_error=graph_capture_error,
        compile_seconds=compile_seconds,
        source_sha256=sha256_text(source),
        generated_source=source,
        resource_usage=resource_usage_to_dict(kernel.resource_usage),
    )


def elapsed_ms(call: Any, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def calibrate_iterations(calls: dict[str, Any]) -> int:
    iterations = 64
    while iterations < MAX_BATCH_ITERS:
        durations = [elapsed_ms(call, iterations) for call in calls.values()]
        if min(durations) >= MIN_BATCH_MS:
            break
        iterations *= 2
    return min(iterations, MAX_BATCH_ITERS)


def paired_benchmark(calls: dict[str, Any]) -> dict[str, Any]:
    for call in calls.values():
        started = time.perf_counter()
        while time.perf_counter() - started < WARM_SECONDS:
            call()
    torch.cuda.synchronize()
    iterations = calibrate_iterations(calls)
    samples: dict[str, list[float]] = {policy: [] for policy in POLICIES}
    cycle_records: list[dict[str, Any]] = []
    for cycle in range(CYCLES):
        order = (
            [POLICIES[0], POLICIES[1], POLICIES[1], POLICIES[0]] if cycle % 2 == 0 else [POLICIES[1], POLICIES[0], POLICIES[0], POLICIES[1]]
        )
        cycle_values: dict[str, list[float]] = {policy: [] for policy in POLICIES}
        for policy in order:
            per_launch_us = elapsed_ms(calls[policy], iterations) * 1000.0 / iterations
            samples[policy].append(per_launch_us)
            cycle_values[policy].append(per_launch_us)
        cycle_p50 = {policy: statistics.median(cycle_values[policy]) for policy in POLICIES}
        cycle_records.append(
            {
                "cycle": cycle,
                "order": order,
                "p50_us": cycle_p50,
                "io_aware_speedup_over_register_count": cycle_p50["register-count"] / cycle_p50["io-aware"],
            }
        )
    summaries = {policy: summary(samples[policy]) for policy in POLICIES}
    paired_speedups = [record["io_aware_speedup_over_register_count"] for record in cycle_records]
    return {
        "iterations_per_sample": iterations,
        "cycles": CYCLES,
        "samples_us": samples,
        "cycle_records": cycle_records,
        "summary": summaries,
        "paired_speedup_summary": ratio_summary(paired_speedups),
        "io_aware_speedup_over_register_count": statistics.median(paired_speedups),
    }


def benchmark_case(case: BenchmarkCase) -> dict[str, Any]:
    canonical_ir = str(case.program)
    result: dict[str, Any] = {
        "name": case.name,
        "family": case.family,
        "canonical_primfunc_sha256": sha256_text(canonical_ir),
        "canonical_primfunc": canonical_ir,
        "policies": {},
    }
    variants: dict[str, CompiledVariant] = {}
    for policy in POLICIES:
        try:
            variant = compile_variant(case, policy)
            variants[policy] = variant
            result["policies"][policy] = {
                "status": "correct",
                "compile_seconds": variant.compile_seconds,
                "generated_source_sha256": variant.source_sha256,
                "generated_source": variant.generated_source,
                "resource_usage": variant.resource_usage,
                "cuda_graph_captured": variant.graph_call is not None,
                "cuda_graph_capture_error": variant.graph_capture_error,
            }
        except Exception as error:  # noqa: BLE001 - retain the other policy's evidence
            result["policies"][policy] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
    if set(variants) != set(POLICIES):
        result["status"] = "incomplete"
        return result

    result["eager"] = paired_benchmark({policy: variants[policy].call for policy in POLICIES})
    if all(variants[policy].graph_call is not None for policy in POLICIES):
        result["cuda_graph"] = paired_benchmark({policy: variants[policy].graph_call for policy in POLICIES})
    result["status"] = "complete"
    return result


def main() -> None:
    started = time.time()
    payload: dict[str, Any] = {
        "schema": "tilelang-layout-cost-model-runtime-ab-v1",
        "repository": "nya-a-cat/tilelang",
        "source_sha": os.environ.get("TILELANG_SOURCE_SHA"),
        "native_base_sha": os.environ.get("TILELANG_NATIVE_BASE_SHA"),
        "started_unix": started,
        "policies": list(POLICIES),
        "evidence_boundary": (
            "One free Colab T4 compares unchanged PrimFuncs under two existing "
            "TileLang backend layout policies. It screens runtime, correctness, "
            "generated code, and compiler resources. Cross-architecture selection "
            "and the global 1.50x goal remain outside this evidence."
        ),
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "tilelang": tilelang.__version__,
            "tilelang_file": str(Path(tilelang.__file__).resolve()),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc": command_output(["nvcc", "--version"]),
        },
    }
    try:
        torch.manual_seed(20260901)
        cases = build_cases()
        payload["cases"] = [benchmark_case(case) for case in cases]
        complete = [case for case in payload["cases"] if case["status"] == "complete"]
        payload["aggregate"] = {
            "complete_cases": len(complete),
            "total_cases": len(cases),
            "eager_io_aware_speedup_geomean": geometric_mean([case["eager"]["io_aware_speedup_over_register_count"] for case in complete]),
            "cuda_graph_io_aware_speedup_geomean": geometric_mean(
                [case["cuda_graph"]["io_aware_speedup_over_register_count"] for case in complete if "cuda_graph" in case]
            ),
        }
        payload["status"] = "complete" if len(complete) == len(cases) else "partial"
    except Exception as error:  # noqa: BLE001 - always emit a forensic result
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
    finally:
        payload["duration_seconds"] = time.time() - started
        payload["finished_nvidia_smi"] = command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,memory.used",
                "--format=csv,noheader",
            ]
        )
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "TILELANG_LAYOUT_AB_RESULT="
            + json.dumps(
                {
                    "path": str(RESULT_PATH),
                    "sha256": hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest(),
                    "status": payload.get("status"),
                    "duration_seconds": payload["duration_seconds"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
