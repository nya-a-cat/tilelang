"""Version-neutral TileLang worker for fixed-commit T4 A/B measurements.

The controller launches this exact file under two isolated Python environments.
Each worker compiles the same 18 frozen PrimFuncs with its installed TileLang
defaults, validates deterministic inputs, and serves bounded timing commands.
No layout policy or candidate-only pass configuration is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any
from collections.abc import Callable

import torch
import tilelang
import tilelang.language as T


RPC_PREFIX = "TILELANG_COMMIT_AB_RPC="
LABEL = os.environ.get("TILELANG_COMMIT_AB_LABEL", "unknown")
SOURCE_COMMIT = os.environ.get("TILELANG_COMMIT_AB_SOURCE_COMMIT", "unknown")
INVENTORY_PATH = Path(
    os.environ.get(
        "TILELANG_COMMIT_AB_INVENTORY",
        f"/tmp/tilelang-commit-ab-{LABEL}-inventory.json",
    )
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: str
    program: Any
    inputs: tuple[torch.Tensor, ...]
    input_sha256: tuple[str, ...]
    output_shape: tuple[int, ...]
    output_dtype: torch.dtype
    reference: torch.Tensor
    atol: float
    rtol: float


@dataclass
class RuntimeCase:
    spec: BenchmarkCase
    kernel: Any
    output: torch.Tensor
    eager_call: Callable[[], None]
    graph: torch.cuda.CUDAGraph | None
    graph_call: Callable[[], None] | None
    graph_capture_error: str | None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - optional provenance must not hide results
        return f"{type(error).__name__}: {error}"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def with_symbol(program: Any, symbol: str) -> Any:
    return program.with_attr("global_symbol", symbol)


def make_row_broadcast(rows: int, cols: int, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(S: T.Tensor((rows,), T.float32), Out: T.Tensor((rows, cols), T.float32)):
        with T.Kernel(1, threads=threads):
            scalar = T.alloc_fragment((rows,), T.float32)
            T.copy(S, scalar)
            for i, j in T.Parallel(rows, cols):
                Out[i, j] = scalar[i] * 2.0

    return with_symbol(main, symbol)


def make_column_broadcast(
    rows: int,
    cols: int,
    dtype: str,
    threads: int,
    symbol: str,
) -> Any:
    @T.prim_func
    def main(A: T.Tensor((cols,), dtype), B: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1, threads=threads):
            values = T.alloc_fragment((cols,), dtype)
            T.copy(A, values)
            for i, j in T.Parallel(rows, cols):
                B[i, j] = values[j] * 2.0

    return with_symbol(main, symbol)


def make_mixed_chain(rows: int, cols: int, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(A: T.Tensor((rows, cols), T.float16), B: T.Tensor((rows, cols), T.float32)):
        with T.Kernel(1, threads=threads):
            x16 = T.alloc_fragment((rows, cols), T.float16)
            x32 = T.alloc_fragment((rows, cols), T.float32)
            T.copy(A, x16)
            for i, j in T.Parallel(rows, cols):
                x32[i, j] = x16[i, j].astype(T.float32) * 2.0
            T.copy(x32, B)

    return with_symbol(main, symbol)


def make_affine(
    rows: int,
    cols: int,
    dtype: str,
    threads: int,
    symbol: str,
) -> Any:
    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), dtype),
        Scale: T.Tensor((cols,), dtype),
        Bias: T.Tensor((cols,), dtype),
        B: T.Tensor((rows, cols), dtype),
    ):
        with T.Kernel(1, threads=threads):
            values = T.alloc_fragment((rows, cols), dtype)
            scale = T.alloc_fragment((cols,), dtype)
            bias = T.alloc_fragment((cols,), dtype)
            T.copy(A, values)
            T.copy(Scale, scale)
            T.copy(Bias, bias)
            for i, j in T.Parallel(rows, cols):
                values[i, j] = values[i, j] * scale[j] + bias[j]
            T.copy(values, B)

    return with_symbol(main, symbol)


def make_transpose(
    rows: int,
    cols: int,
    dtype: str,
    threads: int,
    symbol: str,
) -> Any:
    @T.prim_func
    def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((cols, rows), dtype)):
        with T.Kernel(1, threads=threads):
            fragment = T.alloc_fragment((rows, cols), dtype)
            T.copy(A, fragment)
            for i, j in T.Parallel(rows, cols):
                B[j, i] = fragment[i, j]

    return with_symbol(main, symbol)


def deterministic_cpu_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    salt: int,
) -> tuple[torch.Tensor, str]:
    count = 1
    for extent in shape:
        count *= extent
    values = torch.arange(count, dtype=torch.int64)
    values = ((values * (17 + salt * 2) + 13 + salt) % 251) - 125
    tensor = (values.to(torch.float32) / (31.0 + salt)).reshape(shape).to(dtype)
    digest = sha256_bytes(tensor.contiguous().numpy().tobytes())
    return tensor, digest


def to_cuda(cpu_tensor: torch.Tensor) -> torch.Tensor:
    return cpu_tensor.to(device="cuda", non_blocking=False)


def build_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    for rows, cols, threads in (
        (2, 2560, 256),
        (4, 1024, 128),
        (8, 512, 128),
        (16, 256, 128),
        (4, 4096, 256),
    ):
        name = f"row_broadcast_f32_{rows}x{cols}_t{threads}"
        source_cpu, source_hash = deterministic_cpu_tensor((rows,), torch.float32, rows)
        source = to_cuda(source_cpu)
        cases.append(
            BenchmarkCase(
                name=name,
                family="row_broadcast",
                program=make_row_broadcast(rows, cols, threads, f"scan_{name}"),
                inputs=(source,),
                input_sha256=(source_hash,),
                output_shape=(rows, cols),
                output_dtype=torch.float32,
                reference=source[:, None].expand(rows, cols) * 2.0,
                atol=0.0,
                rtol=0.0,
            )
        )

    for rows, cols, dtype, torch_dtype, threads in (
        (64, 256, "float16", torch.float16, 128),
        (128, 128, "float32", torch.float32, 128),
    ):
        name = f"column_broadcast_{dtype}_{rows}x{cols}_t{threads}"
        source_cpu, source_hash = deterministic_cpu_tensor((cols,), torch_dtype, rows + cols)
        source = to_cuda(source_cpu)
        cases.append(
            BenchmarkCase(
                name=name,
                family="column_broadcast",
                program=make_column_broadcast(rows, cols, dtype, threads, f"scan_{name}"),
                inputs=(source,),
                input_sha256=(source_hash,),
                output_shape=(rows, cols),
                output_dtype=torch_dtype,
                reference=source[None, :].expand(rows, cols) * 2.0,
                atol=0.0,
                rtol=0.0,
            )
        )

    for rows, cols, threads in (
        (64, 64, 128),
        (128, 128, 128),
        (64, 256, 128),
        (128, 256, 256),
    ):
        name = f"mixed_f16_f32_{rows}x{cols}_t{threads}"
        source_cpu, source_hash = deterministic_cpu_tensor((rows, cols), torch.float16, rows + cols)
        source = to_cuda(source_cpu)
        cases.append(
            BenchmarkCase(
                name=name,
                family="mixed_dtype",
                program=make_mixed_chain(rows, cols, threads, f"scan_{name}"),
                inputs=(source,),
                input_sha256=(source_hash,),
                output_shape=(rows, cols),
                output_dtype=torch.float32,
                reference=source.float() * 2.0,
                atol=0.0,
                rtol=0.0,
            )
        )

    for rows, cols, dtype, torch_dtype, threads in (
        (32, 128, "float16", torch.float16, 128),
        (64, 128, "float16", torch.float16, 128),
        (32, 128, "float32", torch.float32, 128),
        (64, 256, "float32", torch.float32, 256),
    ):
        name = f"affine_{dtype}_{rows}x{cols}_t{threads}"
        source_cpu, source_hash = deterministic_cpu_tensor((rows, cols), torch_dtype, rows)
        scale_cpu, scale_hash = deterministic_cpu_tensor((cols,), torch_dtype, cols + 1)
        bias_cpu, bias_hash = deterministic_cpu_tensor((cols,), torch_dtype, cols + 2)
        source, scale, bias = map(to_cuda, (source_cpu, scale_cpu, bias_cpu))
        cases.append(
            BenchmarkCase(
                name=name,
                family="affine",
                program=make_affine(rows, cols, dtype, threads, f"scan_{name}"),
                inputs=(source, scale, bias),
                input_sha256=(source_hash, scale_hash, bias_hash),
                output_shape=(rows, cols),
                output_dtype=torch_dtype,
                reference=source * scale[None, :] + bias[None, :],
                atol=5e-3 if dtype == "float16" else 1e-6,
                rtol=5e-3 if dtype == "float16" else 1e-6,
            )
        )

    for rows, cols, dtype, torch_dtype, threads in (
        (128, 128, "float16", torch.float16, 128),
        (128, 256, "float16", torch.float16, 256),
        (128, 128, "float32", torch.float32, 128),
    ):
        name = f"transpose_{dtype}_{rows}x{cols}_t{threads}"
        source_cpu, source_hash = deterministic_cpu_tensor((rows, cols), torch_dtype, rows + cols)
        source = to_cuda(source_cpu)
        cases.append(
            BenchmarkCase(
                name=name,
                family="transpose",
                program=make_transpose(rows, cols, dtype, threads, f"scan_{name}"),
                inputs=(source,),
                input_sha256=(source_hash,),
                output_shape=(cols, rows),
                output_dtype=torch_dtype,
                reference=source.T.contiguous(),
                atol=0.0,
                rtol=0.0,
            )
        )

    if len(cases) != 18:
        raise RuntimeError(f"expected 18 frozen cases, got {len(cases)}")
    return cases


def compile_case(case: BenchmarkCase) -> tuple[RuntimeCase, dict[str, Any]]:
    canonical_ir = str(case.program)
    output = torch.empty(case.output_shape, device="cuda", dtype=case.output_dtype)
    started = time.perf_counter()
    kernel = tilelang.compile(
        case.program,
        out_idx=None,
        target="cuda",
        execution_backend="tvm_ffi",
    )
    compile_seconds = time.perf_counter() - started

    def eager_call() -> None:
        kernel(*case.inputs, output)

    eager_call()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, case.reference, atol=case.atol, rtol=case.rtol)

    graph: torch.cuda.CUDAGraph | None = None
    graph_call: Callable[[], None] | None = None
    graph_capture_error: str | None = None
    try:
        for _ in range(8):
            eager_call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            eager_call()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, case.reference, atol=case.atol, rtol=case.rtol)
        graph_call = graph.replay
    except Exception as error:  # noqa: BLE001 - eager evidence remains forensic
        graph_capture_error = f"{type(error).__name__}: {error}"
        graph = None
        graph_call = None

    generated_source = kernel.get_kernel_source()
    resource_usage = json_safe(getattr(kernel, "resource_usage", {}))
    runtime = RuntimeCase(
        spec=case,
        kernel=kernel,
        output=output,
        eager_call=eager_call,
        graph=graph,
        graph_call=graph_call,
        graph_capture_error=graph_capture_error,
    )
    inventory = {
        "name": case.name,
        "family": case.family,
        "status": "correct",
        "canonical_primfunc_sha256": sha256_text(canonical_ir),
        "canonical_primfunc": canonical_ir,
        "input_sha256": list(case.input_sha256),
        "output_shape": list(case.output_shape),
        "output_dtype": str(case.output_dtype),
        "atol": case.atol,
        "rtol": case.rtol,
        "compile_seconds": compile_seconds,
        "generated_source_sha256": sha256_text(generated_source),
        "generated_source": generated_source,
        "resource_usage": resource_usage,
        "cuda_graph_captured": graph_call is not None,
        "cuda_graph_capture_error": graph_capture_error,
    }
    return runtime, inventory


def timed_batch(call: Callable[[], None], iterations: int) -> dict[str, float | int]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_started = time.perf_counter()
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    wall_elapsed = time.perf_counter() - wall_started
    event_ms = float(start.elapsed_time(end))
    return {
        "iterations": iterations,
        "event_batch_ms": event_ms,
        "event_per_launch_us": event_ms * 1000.0 / iterations,
        "wall_batch_ms": wall_elapsed * 1000.0,
        "wall_per_launch_us": wall_elapsed * 1e6 / iterations,
    }


def warm(call: Callable[[], None], seconds: float) -> dict[str, float | int]:
    if seconds <= 0:
        raise ValueError("warm seconds must be positive")
    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        call()
        count += 1
    torch.cuda.synchronize()
    return {"launches": count, "elapsed_seconds": time.perf_counter() - started}


def rpc_response(payload: dict[str, Any]) -> None:
    print(RPC_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def runtime_call(runtime: RuntimeCase, mode: str) -> Callable[[], None]:
    if mode == "eager":
        return runtime.eager_call
    if mode == "cuda_graph" and runtime.graph_call is not None:
        return runtime.graph_call
    raise ValueError(f"mode unavailable for {runtime.spec.name}: {mode}")


def system_inventory() -> dict[str, Any]:
    return {
        "label": LABEL,
        "source_commit": SOURCE_COMMIT,
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "tilelang": tilelang.__version__,
        "tilelang_file": str(Path(tilelang.__file__).resolve()),
        "cache_dir": os.environ.get("TILELANG_CACHE_DIR"),
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
    }


def main() -> int:
    started = time.time()
    runtimes: dict[str, RuntimeCase] = {}
    inventory: dict[str, Any] = {
        "schema": "tilelang-fixed-commit-worker-inventory-v1",
        "status": "failed",
        "system": system_inventory(),
        "cases": [],
        "started_unix": started,
    }
    try:
        for case in build_cases():
            runtime, case_inventory = compile_case(case)
            runtimes[case.name] = runtime
            inventory["cases"].append(case_inventory)
        inventory["status"] = "ready"
    except Exception as error:  # noqa: BLE001 - preserve partial compiler evidence
        inventory["error"] = f"{type(error).__name__}: {error}"
        inventory["traceback"] = traceback.format_exc()
    inventory["startup_seconds"] = time.time() - started
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inventory_sha256 = sha256_file(INVENTORY_PATH)
    if inventory["status"] != "ready":
        rpc_response(
            {
                "status": "failed",
                "inventory_path": str(INVENTORY_PATH),
                "inventory_sha256": inventory_sha256,
                "error": inventory.get("error"),
            }
        )
        return 2

    rpc_response(
        {
            "status": "ready",
            "label": LABEL,
            "source_commit": SOURCE_COMMIT,
            "case_names": list(runtimes),
            "graph_cases": [name for name, runtime in runtimes.items() if runtime.graph_call is not None],
            "inventory_path": str(INVENTORY_PATH),
            "inventory_sha256": inventory_sha256,
            "startup_seconds": inventory["startup_seconds"],
        }
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "shutdown":
                rpc_response({"status": "bye", "label": LABEL})
                return 0
            name = str(request["case"])
            mode = str(request["mode"])
            call = runtime_call(runtimes[name], mode)
            if command == "warm":
                result = warm(call, float(request["seconds"]))
            elif command in ("probe", "measure"):
                result = timed_batch(call, int(request["iterations"]))
            else:
                raise ValueError(f"unsupported command: {command!r}")
            rpc_response(
                {
                    "status": "ok",
                    "command": command,
                    "label": LABEL,
                    "case": name,
                    "mode": mode,
                    "result": result,
                }
            )
        except Exception as error:  # noqa: BLE001 - keep server alive for controller cleanup
            rpc_response(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
