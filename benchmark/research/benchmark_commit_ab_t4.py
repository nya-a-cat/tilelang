"""Version-neutral TileLang worker for fixed-commit T4 A/B evidence.

The controller launches this exact file under two isolated Python environments.
Each worker compiles one frozen suite with its installed TileLang defaults,
validates deterministic inputs, and serves bounded timing commands. An explicit
diagnostic mode recompiles generated CUDA source to SASS for instruction-level
comparison. No layout policy or candidate-only pass configuration is supplied.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from typing import Any

import torch
import tilelang
import tilelang.language as T


RPC_PREFIX = "TILELANG_COMMIT_AB_RPC="
LABEL = os.environ.get("TILELANG_COMMIT_AB_LABEL", "unknown")
SOURCE_COMMIT = os.environ.get("TILELANG_COMMIT_AB_SOURCE_COMMIT", "unknown")
SUITE = os.environ.get("TILELANG_COMMIT_AB_SUITE", "layout")
SUITE_CASES = {"layout": 18, "reduction": 12}
CAPTURE_SASS = os.environ.get("TILELANG_COMMIT_AB_CAPTURE_SASS", "0") == "1"
SASS_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)",
    re.MULTILINE,
)
SASS_REGISTER_PATTERNS = (
    re.compile(r"SHI_REGISTERS=(\d+)"),
    re.compile(r'\.sectioninfo\s+@"SHI_REGISTERS=(\d+)"'),
)
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


def dynamic_shared_bytes(kernel: Any) -> int:
    artifact = getattr(kernel, "artifact", None)
    device_mod = getattr(artifact, "device_mod", None)
    functions = getattr(device_mod, "functions", {})
    values = [int(func.attrs.get("dyn_shared_memory_buf", 0)) for func in functions.values()]
    return max(values, default=0)


def capture_sass(kernel: Any) -> dict[str, Any]:
    getter = getattr(kernel, "_get_sass", None)
    if getter is None:
        raise RuntimeError("installed TileLang JITKernel does not expose _get_sass")
    started = time.perf_counter()
    sass = getter(verbose=False)
    if not isinstance(sass, str) or not sass.strip():
        raise RuntimeError("SASS capture returned no disassembly")
    opcodes = Counter(match.group("opcode").upper() for match in SASS_INSTRUCTION_RE.finditer(sass))

    def count_prefixes(*prefixes: str) -> int:
        return sum(count for opcode, count in opcodes.items() if opcode.startswith(prefixes))

    registers = None
    for pattern in SASS_REGISTER_PATTERNS:
        match = pattern.search(sass)
        if match is not None:
            registers = int(match.group(1))
            break
    return {
        "method": "JITKernel._get_sass generated-source recompile",
        "duration_seconds": time.perf_counter() - started,
        "sass_sha256": sha256_text(sass),
        "sass_chars": len(sass),
        "registers": registers,
        "instruction_count": sum(opcodes.values()),
        "groups": {
            "barrier": count_prefixes("BAR", "MBAR"),
            "shuffle": count_prefixes("SHFL"),
            "shared_load": count_prefixes("LDS", "LDSM"),
            "shared_store": count_prefixes("STS"),
            "global_load": count_prefixes("LDG"),
            "global_store": count_prefixes("STG"),
        },
        "opcodes": dict(opcodes.most_common()),
        "sass": sass,
    }


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


def make_row_reduce(
    batch: int,
    width: int,
    dtype: str,
    op: str,
    threads: int,
    symbol: str,
) -> Any:
    @T.prim_func
    def main(A: T.Tensor((batch, width), dtype), B: T.Tensor((batch,), dtype)):
        with T.Kernel(batch, threads=threads) as bx:
            values = T.alloc_fragment((width,), dtype)
            result = T.alloc_fragment((1,), dtype)
            T.copy(A[bx, 0], values)
            if op == "sum":
                T.reduce_sum(values, result, dim=0)
            elif op == "max":
                T.reduce_max(values, result, dim=0, clear=True)
            elif op == "min":
                T.reduce_min(values, result, dim=0, clear=True)
            elif op == "abssum":
                T.reduce_abssum(values, result, dim=0)
            elif op == "bitor":
                T.reduce_bitor(values, result, dim=0, clear=True)
            B[bx] = result[0]

    return with_symbol(main, symbol)


def make_rmsnorm(rows: int, width: int, symbol: str) -> Any:
    @T.prim_func
    def main(A: T.Tensor((rows, width), T.float32), B: T.Tensor((rows, width), T.float32)):
        with T.Kernel(rows, threads=128) as bx:
            values = T.alloc_fragment((width,), T.float32)
            squares = T.alloc_fragment((width,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            T.copy(A[bx, 0], values)
            for j in T.Parallel(width):
                squares[j] = values[j] * values[j]
            T.reduce_sum(squares, total, dim=0)
            scale = T.rsqrt(total[0] / width + 1e-6)
            for j in T.Parallel(width):
                values[j] *= scale
            T.copy(values, B[bx, 0])

    return with_symbol(main, symbol)


def make_softmax(rows: int, width: int, symbol: str) -> Any:
    @T.prim_func
    def main(A: T.Tensor((rows, width), T.float32), B: T.Tensor((rows, width), T.float32)):
        with T.Kernel(rows, threads=128) as bx:
            values = T.alloc_fragment((width,), T.float32)
            maximum = T.alloc_fragment((1,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            T.copy(A[bx, 0], values)
            T.reduce_max(values, maximum, dim=0, clear=True)
            for j in T.Parallel(width):
                values[j] = T.exp(values[j] - maximum[0])
            T.reduce_sum(values, total, dim=0)
            for j in T.Parallel(width):
                values[j] /= total[0]
            T.copy(values, B[bx, 0])

    return with_symbol(main, symbol)


def make_layernorm(rows: int, width: int, symbol: str) -> Any:
    @T.prim_func
    def main(
        X: T.Tensor((rows, width), T.float16),
        Gamma: T.Tensor((width,), T.float16),
        Beta: T.Tensor((width,), T.float16),
        Y: T.Tensor((rows, width), T.float16),
    ):
        with T.Kernel(rows, threads=128) as bx:
            x_shared = T.alloc_shared((width,), T.float16)
            gamma_shared = T.alloc_shared((width,), T.float16)
            beta_shared = T.alloc_shared((width,), T.float16)
            values = T.alloc_fragment((width,), T.float32)
            squares = T.alloc_fragment((width,), T.float32)
            sum_value = T.alloc_fragment((1,), T.float32)
            sumsq = T.alloc_fragment((1,), T.float32)

            T.copy(X[bx, 0], x_shared)
            T.copy(Gamma, gamma_shared)
            T.copy(Beta, beta_shared)
            for j in T.Parallel(width):
                values[j] = T.cast(x_shared[j], T.float32)
                squares[j] = values[j] * values[j]
            T.reduce_sum(values, sum_value, dim=0)
            T.reduce_sum(squares, sumsq, dim=0)
            mean = sum_value[0] / width
            rstd = T.rsqrt(sumsq[0] / width - mean * mean + 1e-5)
            for j in T.Parallel(width):
                x_shared[j] = T.cast(
                    (values[j] - mean) * rstd * T.cast(gamma_shared[j], T.float32) + T.cast(beta_shared[j], T.float32),
                    T.float16,
                )
            T.copy(x_shared, Y[bx, 0])

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


def deterministic_int_cpu_tensor(
    shape: tuple[int, ...],
    salt: int,
) -> tuple[torch.Tensor, str]:
    count = 1
    for extent in shape:
        count *= extent
    values = torch.arange(count, dtype=torch.int64)
    tensor = (((values * (29 + salt * 2) + 7 + salt) % 65521) - 32760).reshape(shape).to(torch.int32)
    digest = sha256_bytes(tensor.contiguous().numpy().tobytes())
    return tensor, digest


def to_cuda(cpu_tensor: torch.Tensor) -> torch.Tensor:
    return cpu_tensor.to(device="cuda", non_blocking=False)


def build_layout_cases() -> list[BenchmarkCase]:
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


def bitwise_or_reference(source: torch.Tensor) -> torch.Tensor:
    result = source[:, 0].clone()
    for column in range(1, source.shape[1]):
        result.bitwise_or_(source[:, column])
    return result


def build_reduction_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for op, batch, width, dtype, torch_dtype, threads in (
        ("sum", 1, 128, "float32", torch.float32, 128),
        ("max", 4, 1024, "float32", torch.float32, 128),
        ("min", 8, 4096, "float32", torch.float32, 256),
        ("abssum", 4, 16384, "float32", torch.float32, 256),
        ("bitor", 8, 4096, "int32", torch.int32, 256),
    ):
        name = f"reduce_{op}_{dtype}_b{batch}_n{width}_t{threads}"
        if torch_dtype == torch.int32:
            source_cpu, source_hash = deterministic_int_cpu_tensor((batch, width), batch + width)
        else:
            source_cpu, source_hash = deterministic_cpu_tensor((batch, width), torch_dtype, batch + width)
        source = to_cuda(source_cpu)
        if op == "sum":
            reference = source.sum(dim=1).to(torch_dtype)
        elif op == "max":
            reference = source.max(dim=1).values
        elif op == "min":
            reference = source.min(dim=1).values
        elif op == "abssum":
            reference = source.abs().sum(dim=1).to(torch_dtype)
        elif op == "bitor":
            reference = bitwise_or_reference(source_cpu).to(device="cuda")
        else:
            raise ValueError(op)
        cases.append(
            BenchmarkCase(
                name=name,
                family="reduction",
                program=make_row_reduce(batch, width, dtype, op, threads, f"commit_ab_{name}"),
                inputs=(source,),
                input_sha256=(source_hash,),
                output_shape=(batch,),
                output_dtype=torch_dtype,
                reference=reference,
                atol=2e-3 if op == "abssum" else 1e-5 if torch_dtype == torch.float32 else 0.0,
                rtol=2e-3 if op == "abssum" else 1e-5 if torch_dtype == torch.float32 else 0.0,
            )
        )

    for rows, width in ((1, 128), (32, 4096), (1280, 16384)):
        name = f"softmax_f32_b{rows}_h{width}_t128"
        source_cpu, source_hash = deterministic_cpu_tensor((rows, width), torch.float32, rows + width)
        source = to_cuda(source_cpu)
        cases.append(
            BenchmarkCase(
                name=name,
                family="softmax",
                program=make_softmax(rows, width, f"commit_ab_{name}"),
                inputs=(source,),
                input_sha256=(source_hash,),
                output_shape=(rows, width),
                output_dtype=torch.float32,
                reference=torch.softmax(source, dim=1),
                atol=2e-4,
                rtol=2e-3,
            )
        )

    for rows, width in ((1, 128), (32, 4096), (1280, 8192)):
        name = f"rmsnorm_f32_b{rows}_h{width}_t128"
        source_cpu, source_hash = deterministic_cpu_tensor((rows, width), torch.float32, rows + width + 1)
        source = to_cuda(source_cpu)
        cases.append(
            BenchmarkCase(
                name=name,
                family="rmsnorm",
                program=make_rmsnorm(rows, width, f"commit_ab_{name}"),
                inputs=(source,),
                input_sha256=(source_hash,),
                output_shape=(rows, width),
                output_dtype=torch.float32,
                reference=source * torch.rsqrt(source.square().mean(dim=1, keepdim=True) + 1e-6),
                atol=2e-3,
                rtol=2e-3,
            )
        )

    rows, width = 1280, 4096
    name = f"layernorm_f16_b{rows}_h{width}_t128"
    source_cpu, source_hash = deterministic_cpu_tensor((rows, width), torch.float16, rows + width + 2)
    gamma_cpu, gamma_hash = deterministic_cpu_tensor((width,), torch.float16, width + 3)
    beta_cpu, beta_hash = deterministic_cpu_tensor((width,), torch.float16, width + 4)
    source, gamma, beta = map(to_cuda, (source_cpu, gamma_cpu, beta_cpu))
    cases.append(
        BenchmarkCase(
            name=name,
            family="layernorm",
            program=make_layernorm(rows, width, f"commit_ab_{name}"),
            inputs=(source, gamma, beta),
            input_sha256=(source_hash, gamma_hash, beta_hash),
            output_shape=(rows, width),
            output_dtype=torch.float16,
            reference=torch.nn.functional.layer_norm(source, (width,), gamma, beta, eps=1e-5),
            atol=2e-2,
            rtol=2e-2,
        )
    )

    if len(cases) != SUITE_CASES["reduction"]:
        raise RuntimeError(f"expected {SUITE_CASES['reduction']} frozen reduction cases, got {len(cases)}")
    return cases


def build_cases() -> list[BenchmarkCase]:
    if SUITE == "layout":
        return build_layout_cases()
    if SUITE == "reduction":
        return build_reduction_cases()
    raise ValueError(f"unsupported suite: {SUITE!r}")


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
    sass_capture = capture_sass(kernel) if CAPTURE_SASS else None
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
        "dynamic_shared_bytes": dynamic_shared_bytes(kernel),
        "sass_capture": sass_capture,
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
        "suite": SUITE,
        "capture_sass": CAPTURE_SASS,
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
        "schema": "tilelang-fixed-commit-worker-inventory-v3",
        "status": "failed",
        "suite": SUITE,
        "capture_sass": CAPTURE_SASS,
        "system": system_inventory(),
        "cases": [],
        "started_unix": started,
    }
    try:
        if SUITE not in SUITE_CASES:
            raise ValueError(f"unsupported suite: {SUITE!r}")
        cases = build_cases()
        if len(cases) != SUITE_CASES[SUITE]:
            raise RuntimeError(f"suite case count drifted: {SUITE}={len(cases)}")
        for case in cases:
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
            "suite": SUITE,
            "capture_sass": CAPTURE_SASS,
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
