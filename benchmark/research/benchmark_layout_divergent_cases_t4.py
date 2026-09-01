"""Runtime A/B for cases found by the compile-only layout-divergence scan.

The scan constructs each PrimFunc once.  This benchmark reuses those exact
program objects for T4 correctness, ptxas resources, eager launch timing, and
CUDA Graph replay under ``register-count`` and ``io-aware``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILELANG_LAYOUT_AB_RESULT", "/content/tilelang-layout-divergent-cases-t4.json")
os.environ.setdefault("TILELANG_LAYOUT_AB_SCHEMA", "tilelang-layout-divergent-runtime-ab-v1")
os.environ.setdefault(
    "TILELANG_LAYOUT_AB_EVIDENCE_BOUNDARY",
    (
        "One free Colab T4 measures unchanged PrimFuncs selected by the CPU-only layout-policy divergence scan. "
        "It compares register-count and io-aware with correctness, ptxas resources, eager launches, and CUDA "
        "Graph replay. Other GPU architectures, model-level workloads, and the global 1.50x goal remain outside "
        "this evidence."
    ),
)

import benchmark_layout_cost_models_t4 as base  # noqa: E402
import scan_layout_policy_divergence as scanner  # noqa: E402
import torch  # noqa: E402


def _case(
    programs: dict[str, scanner.ScanCase],
    *,
    name: str,
    inputs: tuple[torch.Tensor, ...],
    reference: torch.Tensor,
    atol: float,
    rtol: float,
) -> base.BenchmarkCase:
    scanned = programs[name]
    return base.BenchmarkCase(
        name=name,
        family=scanned.family,
        program=scanned.program,
        inputs=inputs,
        output_shape=tuple(reference.shape),
        output_dtype=reference.dtype,
        reference=reference,
        atol=atol,
        rtol=rtol,
    )


def build_cases() -> list[base.BenchmarkCase]:
    device = torch.device("cuda")
    programs = {case.name: case for case in scanner.build_scan_cases()}
    cases: list[base.BenchmarkCase] = []

    for rows, cols, threads in (
        (2, 2560, 256),
        (4, 1024, 128),
        (8, 512, 128),
        (16, 256, 128),
        (4, 4096, 256),
    ):
        name = f"row_broadcast_f32_{rows}x{cols}_t{threads}"
        source = torch.randn(rows, device=device, dtype=torch.float32)
        cases.append(
            _case(
                programs,
                name=name,
                inputs=(source,),
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
        source = torch.randn(cols, device=device, dtype=torch_dtype)
        cases.append(
            _case(
                programs,
                name=name,
                inputs=(source,),
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
        source = torch.randn(rows, cols, device=device, dtype=torch.float16)
        cases.append(
            _case(
                programs,
                name=name,
                inputs=(source,),
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
        source = torch.randn(rows, cols, device=device, dtype=torch_dtype)
        scale = torch.randn(cols, device=device, dtype=torch_dtype)
        bias = torch.randn(cols, device=device, dtype=torch_dtype)
        cases.append(
            _case(
                programs,
                name=name,
                inputs=(source, scale, bias),
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
        source = torch.randn(rows, cols, device=device, dtype=torch_dtype)
        cases.append(
            _case(
                programs,
                name=name,
                inputs=(source,),
                reference=source.T.contiguous(),
                atol=0.0,
                rtol=0.0,
            )
        )

    if len(cases) != 18:
        raise RuntimeError(f"expected 18 divergent T4 cases, got {len(cases)}")
    return cases


base.build_cases = build_cases


if __name__ == "__main__":
    base.main()
