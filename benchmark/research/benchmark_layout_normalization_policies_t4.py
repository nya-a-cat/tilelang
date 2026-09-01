"""Calibration A/B for layout policies on normalization-heavy kernels.

The upstream IO-aware default was reverted after wider inferred layouts raised
register pressure and reduced occupancy in reduction-heavy normalization
kernels.  This free-T4 matrix recreates that risk family with unchanged
PrimFuncs.  Alongside runtime and ptxas resources, it records the exact legacy
fragment-slot proxy visible at LayoutInference so a single-lowering guard can
be calibrated against compiler-reported registers and spills.
"""

from __future__ import annotations

import math
import os
from typing import Any

os.environ.setdefault("TILELANG_LAYOUT_AB_RESULT", "/content/tilelang-layout-normalization-policies-t4.json")
os.environ.setdefault("TILELANG_LAYOUT_AB_SCHEMA", "tilelang-layout-normalization-runtime-ab-v1")
os.environ.setdefault(
    "TILELANG_LAYOUT_AB_EVIDENCE_BOUNDARY",
    (
        "One free Colab T4 compares unchanged normalization-heavy PrimFuncs under register-count and "
        "io-aware layout selection. It calibrates layout-stage fragment-slot proxies against ptxas resources, "
        "correctness, eager launches, and CUDA Graph replay. It does not establish Hopper, Blackwell, ROCm, "
        "model-level, or global 1.50x results."
    ),
)

import benchmark_layout_cost_models_t4 as base  # noqa: E402
import torch  # noqa: E402
import tilelang as tl  # noqa: E402
import tilelang.language as T  # noqa: E402
from tilelang import tvm  # noqa: E402
from tilelang.backend.target import determine_target  # noqa: E402
from tilelang.layout import Fragment  # noqa: E402
from tvm.tirx.stmt_functor import post_order_visit  # noqa: E402


ROWS = int(os.environ.get("TILELANG_LAYOUT_NORM_ROWS", "1280"))


def make_rmsnorm(rows: int, width: int, block_rows: int, symbol: str) -> Any:
    """The in-tree compile-speed RMSNorm shape, retained exactly in spirit."""

    @T.prim_func
    def main(A: T.Tensor((rows, width), T.float32), B: T.Tensor((rows, width), T.float32)):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=128) as bx:
            values = T.alloc_fragment((block_rows, width), T.float32)
            squares = T.alloc_fragment((block_rows, width), T.float32)
            totals = T.alloc_fragment((block_rows,), T.float32)
            T.copy(A[bx * block_rows, 0], values)
            for i, j in T.Parallel(block_rows, width):
                squares[i, j] = values[i, j] * values[i, j]
            T.reduce_sum(squares, totals, dim=1)
            for i in T.Parallel(block_rows):
                totals[i] = T.rsqrt(totals[i] / width + 1e-6)
            for i, j in T.Parallel(block_rows, width):
                values[i, j] *= totals[i]
            T.copy(values, B[bx * block_rows, 0])

    return base.with_symbol(main, symbol)


def make_softmax(rows: int, width: int, block_rows: int, symbol: str) -> Any:
    """The in-tree compile-speed row-wise softmax shape."""

    @T.prim_func
    def main(A: T.Tensor((rows, width), T.float32), B: T.Tensor((rows, width), T.float32)):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=128) as bx:
            values = T.alloc_fragment((block_rows, width), T.float32)
            maxima = T.alloc_fragment((block_rows,), T.float32)
            totals = T.alloc_fragment((block_rows,), T.float32)
            T.copy(A[bx * block_rows, 0], values)
            T.reduce_max(values, maxima, dim=1, clear=True)
            for i, j in T.Parallel(block_rows, width):
                values[i, j] = T.exp(values[i, j] - maxima[i])
            T.reduce_sum(values, totals, dim=1)
            for i, j in T.Parallel(block_rows, width):
                values[i, j] /= totals[i]
            T.copy(values, B[bx * block_rows, 0])

    return base.with_symbol(main, symbol)


def make_layernorm(rows: int, width: int, block_rows: int, symbol: str) -> Any:
    """Forward LayerNorm distilled from examples/norm/layernorm.py."""

    @T.prim_func
    def main(
        X: T.Tensor((rows, width), T.float16),
        Gamma: T.Tensor((width,), T.float16),
        Beta: T.Tensor((width,), T.float16),
        Y: T.Tensor((rows, width), T.float16),
    ):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=128) as bx:
            x_shared = T.alloc_shared((block_rows, width), T.float16)
            gamma_shared = T.alloc_shared((width,), T.float16)
            beta_shared = T.alloc_shared((width,), T.float16)
            values = T.alloc_fragment((block_rows, width), T.float32)
            squares = T.alloc_fragment((block_rows, width), T.float32)
            sums = T.alloc_fragment((block_rows,), T.float32)
            sumsq = T.alloc_fragment((block_rows,), T.float32)
            means = T.alloc_fragment((block_rows,), T.float32)
            rstd = T.alloc_fragment((block_rows,), T.float32)

            T.copy(X[bx * block_rows, 0], x_shared)
            T.copy(Gamma, gamma_shared)
            T.copy(Beta, beta_shared)
            for i, j in T.Parallel(block_rows, width):
                values[i, j] = T.cast(x_shared[i, j], T.float32)
                squares[i, j] = values[i, j] * values[i, j]
            T.reduce_sum(values, sums, dim=1)
            T.reduce_sum(squares, sumsq, dim=1)
            for i in T.Parallel(block_rows):
                means[i] = sums[i] / width
                rstd[i] = T.rsqrt(sumsq[i] / width - means[i] * means[i] + 1e-5)
            for i, j in T.Parallel(block_rows, width):
                x_shared[i, j] = T.cast(
                    (values[i, j] - means[i]) * rstd[i] * T.cast(gamma_shared[j], T.float32) + T.cast(beta_shared[j], T.float32),
                    T.float16,
                )
            T.copy(x_shared, Y[bx * block_rows, 0])

    return base.with_symbol(main, symbol)


def make_splitk_rmsnorm(rows: int, width: int, block_rows: int, block_k: int, symbol: str) -> Any:
    """Bounded-register RMSNorm used as a control for large hidden sizes."""

    @T.prim_func
    def main(A: T.Tensor((rows, width), T.float32), B: T.Tensor((rows, width), T.float32)):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=128) as bx:
            values_shared = T.alloc_shared((block_rows, block_k), T.float32)
            squares = T.alloc_fragment((block_rows, block_k), T.float32)
            totals = T.alloc_fragment((block_rows,), T.float32)
            T.clear(squares)
            for k in T.serial(T.ceildiv(width, block_k)):
                T.copy(A[bx * block_rows, k * block_k], values_shared)
                for i, j in T.Parallel(block_rows, block_k):
                    squares[i, j] += values_shared[i, j] * values_shared[i, j]
            T.reduce_sum(squares, totals, dim=1)
            for i in T.Parallel(block_rows):
                totals[i] = T.rsqrt(totals[i] / width + 1e-6)
            for k in T.serial(T.ceildiv(width, block_k)):
                T.copy(A[bx * block_rows, k * block_k], values_shared)
                for i, j in T.Parallel(block_rows, block_k):
                    values_shared[i, j] *= totals[i]
                T.copy(values_shared, B[bx * block_rows, k * block_k])

    return base.with_symbol(main, symbol)


def _normalization_case(
    *,
    name: str,
    family: str,
    program: Any,
    inputs: tuple[torch.Tensor, ...],
    reference: torch.Tensor,
    atol: float,
    rtol: float,
) -> base.BenchmarkCase:
    return base.BenchmarkCase(
        name=name,
        family=family,
        program=program,
        inputs=inputs,
        output_shape=tuple(reference.shape),
        output_dtype=reference.dtype,
        reference=reference,
        atol=atol,
        rtol=rtol,
    )


def build_cases() -> list[base.BenchmarkCase]:
    device = torch.device("cuda")
    cases: list[base.BenchmarkCase] = []

    rms_shapes = ((1024, 1), (1024, 8), (1024, 32), (4096, 1), (4096, 4), (8192, 1))
    for width, block_rows in rms_shapes:
        source = torch.randn(ROWS, width, device=device, dtype=torch.float32)
        reference = source * torch.rsqrt(source.square().mean(dim=1, keepdim=True) + 1e-6)
        cases.append(
            _normalization_case(
                name=f"rmsnorm_f32_{ROWS}x{width}_bm{block_rows}_t128",
                family="rmsnorm",
                program=make_rmsnorm(ROWS, width, block_rows, f"layout_rmsnorm_{ROWS}_{width}_bm{block_rows}"),
                inputs=(source,),
                reference=reference,
                atol=2e-3,
                rtol=2e-3,
            )
        )

    softmax_shapes = ((1024, 1), (1024, 8), (1024, 32), (4096, 1), (4096, 4), (8192, 1))
    for width, block_rows in softmax_shapes:
        source = torch.randn(ROWS, width, device=device, dtype=torch.float32)
        cases.append(
            _normalization_case(
                name=f"softmax_f32_{ROWS}x{width}_bm{block_rows}_t128",
                family="softmax",
                program=make_softmax(ROWS, width, block_rows, f"layout_softmax_{ROWS}_{width}_bm{block_rows}"),
                inputs=(source,),
                reference=torch.softmax(source, dim=1),
                atol=2e-4,
                rtol=2e-3,
            )
        )

    for width in (1024, 4096):
        source = torch.randn(ROWS, width, device=device, dtype=torch.float16)
        gamma = torch.randn(width, device=device, dtype=torch.float16)
        beta = torch.randn(width, device=device, dtype=torch.float16)
        reference = torch.nn.functional.layer_norm(source, (width,), gamma, beta, eps=1e-5)
        cases.append(
            _normalization_case(
                name=f"layernorm_f16_{ROWS}x{width}_bm1_t128",
                family="layernorm",
                program=make_layernorm(ROWS, width, 1, f"layout_layernorm_{ROWS}_{width}_bm1"),
                inputs=(source, gamma, beta),
                reference=reference,
                atol=2e-2,
                rtol=2e-2,
            )
        )

    for block_rows in (1, 4):
        width, block_k = 8192, 512
        source = torch.randn(ROWS, width, device=device, dtype=torch.float32)
        reference = source * torch.rsqrt(source.square().mean(dim=1, keepdim=True) + 1e-6)
        cases.append(
            _normalization_case(
                name=f"rmsnorm_splitk_f32_{ROWS}x{width}_bm{block_rows}_bk{block_k}_t128",
                family="rmsnorm_splitk",
                program=make_splitk_rmsnorm(
                    ROWS,
                    width,
                    block_rows,
                    block_k,
                    f"layout_rmsnorm_splitk_{ROWS}_{width}_bm{block_rows}_bk{block_k}",
                ),
                inputs=(source,),
                reference=reference,
                atol=2e-3,
                rtol=2e-3,
            )
        )

    return cases


def layout_register_proxy(program: Any, policy: str) -> dict[str, Any]:
    """Reproduce CountRegisterSlots and retain dtype-aware lower bounds."""
    target = tvm.target.Target(determine_target("auto"))
    module = tvm.IRModule({"main": program})
    with target, tvm.transform.PassContext(config={"tl.layout_cost_model": policy}):
        module = tvm.tirx.transform.BindTarget(target)(module)
        module = tl.transform.MaterializeKernelLaunch()(module)
        module = tl.transform.LayoutInference()(module)

    seen: set[Any] = set()
    buffers: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if not isinstance(node, tvm.tirx.SBlock) or "layout_map" not in node.annotations:
            return
        for buffer, layout in node.annotations["layout_map"].items():
            if buffer in seen or not isinstance(layout, Fragment):
                continue
            seen.add(buffer)
            output_shape = [int(extent) for extent in layout.get_output_shape()]
            slots = math.prod(output_shape)
            element_bits = int(buffer.dtype.bits) * int(buffer.dtype.lanes)
            buffers.append(
                {
                    "name": buffer.name,
                    "dtype": str(buffer.dtype),
                    "output_shape": output_shape,
                    "slots": slots,
                    "element_bits": element_bits,
                    "packed_32bit_words_lower_bound": math.ceil(slots * element_bits / 32),
                    "thread_extent": int(layout.get_thread_size()),
                    "replicate_extent": int(layout.replicate_size),
                }
            )

    post_order_visit(module["main"].body, visit)
    buffers.sort(key=lambda entry: entry["name"])
    return {
        "legacy_register_slots": sum(entry["slots"] for entry in buffers),
        "packed_32bit_words_lower_bound": sum(entry["packed_32bit_words_lower_bound"] for entry in buffers),
        "fragment_buffers": buffers,
    }


_base_benchmark_case = base.benchmark_case


def benchmark_case(case: base.BenchmarkCase) -> dict[str, Any]:
    result = _base_benchmark_case(case)
    for policy in base.POLICIES:
        try:
            result["policies"][policy]["layout_register_proxy"] = layout_register_proxy(case.program, policy)
        except Exception as error:  # noqa: BLE001 - runtime evidence remains useful
            result["policies"][policy]["layout_register_proxy_error"] = f"{type(error).__name__}: {error}"
    if all(result["policies"].get(policy, {}).get("status") == "correct" for policy in base.POLICIES):
        baseline = result["policies"][base.BASELINE_POLICY]
        candidate = result["policies"][base.CANDIDATE_POLICY]
        baseline_slots = baseline["layout_register_proxy"]["legacy_register_slots"]
        candidate_slots = candidate["layout_register_proxy"]["legacy_register_slots"]
        result["policy_delta"] = {
            "identical_generated_source": baseline["generated_source_sha256"] == candidate["generated_source_sha256"],
            "candidate_register_slot_ratio": candidate_slots / baseline_slots if baseline_slots else None,
        }
    return result


base.build_cases = build_cases
base.benchmark_case = benchmark_case


if __name__ == "__main__":
    base.main()
