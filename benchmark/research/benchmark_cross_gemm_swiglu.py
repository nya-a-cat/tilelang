"""Cross-GEMM SwiGLU feasibility benchmark for free Colab T4 screening.

The fused kernel keeps both GEMM accumulators on chip, loads each activation
tile once, and applies the gated activation before the only output store.  It
is intentionally architecture-portable TileLang code: the first experiment
targets SM75, while the same program can be compiled for newer CUDA targets.

This is an oracle-style experiment.  It measures whether a profitable fused
schedule exists before investing in graph recognition and schedule synthesis.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T
from tilelang.autotuner import TargetProfile
from tilelang.carver.arch.driver.cuda_driver import cudaDeviceAttrNames, get_device_attribute

if __package__:
    from .cross_gemm_schedule_space import (
        CrossGemmWorkload,
        cross_gemm_schedule_estimate,
        cross_gemm_search_summary,
        ranked_cross_gemm_schedules,
    )
else:
    from cross_gemm_schedule_space import (
        CrossGemmWorkload,
        cross_gemm_schedule_estimate,
        cross_gemm_search_summary,
        ranked_cross_gemm_schedules,
    )


DEFAULT_SHAPES = (
    (256, 896, 4864, "qwen2.5-0.5b"),
    (512, 896, 4864, "qwen2.5-0.5b"),
    (256, 1536, 8960, "qwen2.5-1.5b"),
)


def cuda_target_profile(device: int | None = None) -> TargetProfile:
    """Build a resource profile from the selected CUDA device."""

    device = torch.cuda.current_device() if device is None else device
    properties = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    max_threads_per_block = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxThreadsPerBlock, device)
    max_registers_per_block = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxRegistersPerBlock, device)
    max_threads_per_compute_unit = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxThreadsPerMultiProcessor, device)
    max_shared_bytes_per_compute_unit = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxSharedMemoryPerMultiprocessor, device)
    max_registers_per_compute_unit = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxRegistersPerMultiprocessor, device)
    max_blocks_per_compute_unit = get_device_attribute(cudaDeviceAttrNames.cudaDevAttrMaxBlocksPerMultiprocessor, device)
    limits = {
        "max_threads_per_block": int(max_threads_per_block or getattr(properties, "max_threads_per_block", 1024)),
        "max_shared_bytes_per_block": int(properties.shared_memory_per_block),
        "max_registers_per_thread": 255,
        "max_registers_per_block": int(max_registers_per_block or 64 * 1024),
        "multiprocessor_count": int(properties.multi_processor_count),
        "compute_unit_count": int(properties.multi_processor_count),
        "max_threads_per_compute_unit": int(
            max_threads_per_compute_unit or getattr(properties, "max_threads_per_multi_processor", max_threads_per_block or 1024)
        ),
        "max_shared_bytes_per_compute_unit": int(
            max_shared_bytes_per_compute_unit or getattr(properties, "shared_memory_per_multiprocessor", properties.shared_memory_per_block)
        ),
        "max_registers_per_compute_unit": int(max_registers_per_compute_unit or max_registers_per_block or 64 * 1024),
        "max_blocks_per_compute_unit": int(max_blocks_per_compute_unit or 1),
        "warp_size": int(properties.warp_size),
        "preferred_warps_per_block": 4,
    }
    features = {"async_copy"} if major >= 8 else set()
    return TargetProfile("cuda", f"sm_{major}{minor}", features=frozenset(features), limits=limits)


def target_profile_payload(target: TargetProfile) -> dict[str, Any]:
    """Serialize the exact profile used by the deterministic ranker."""

    return {
        "backend": target.backend,
        "arch": target.arch,
        "features": sorted(target.features),
        "limits": dict(target.limits),
    }


def select_ranked_schedule(
    workload: CrossGemmWorkload,
    target: TargetProfile,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select and describe one statically ranked schedule."""

    if rank < 0:
        raise ValueError("schedule rank must be non-negative")
    ranked = ranked_cross_gemm_schedules(workload, target)
    if rank >= len(ranked):
        raise ValueError(f"schedule rank {rank} exceeds {len(ranked)} accepted candidates")
    config = ranked[rank]
    summary = cross_gemm_search_summary(workload, target, top_k=max(8, rank + 1))
    return config, {
        "policy": "resource_rank_v2",
        "rank": rank,
        "selected_config": config,
        "selected_estimate": cross_gemm_schedule_estimate(config, workload, target),
        "space": summary["space"],
        "top_candidates": summary["top_candidates"],
    }


def cross_gemm_swiglu(
    M: int,
    K: int,
    N: int,
    *,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 0,
    threads: int = 128,
    variant: str = "swiglu",
):
    """Build a single, dual-add, or SwiGLU Cross-GEMM diagnostic."""

    if variant not in {"single", "dual_add", "swiglu"}:
        raise ValueError(f"unsupported variant: {variant}")

    dtype = T.float16
    accum_dtype = T.float32
    log2e = 1.4426950408889634

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        W_gate: T.Tensor((N, K), dtype),
        W_up: T.Tensor((N, K), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_gate_shared = T.alloc_shared((block_N, block_K), dtype)
            gate = T.alloc_fragment((block_M, block_N), accum_dtype)
            Out_shared = T.alloc_shared((block_M, block_N), dtype)
            if variant != "single":
                W_up_shared = T.alloc_shared((block_N, block_K), dtype)
                up = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(gate)
            if variant != "single":
                T.clear(up)
            if num_stages > 0:
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # The shared A tile feeds both independent GEMMs.
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(W_gate[bx * block_N, ko * block_K], W_gate_shared)
                    T.gemm(A_shared, W_gate_shared, gate, transpose_B=True)
                    if variant != "single":
                        T.copy(W_up[bx * block_N, ko * block_K], W_up_shared)
                        T.gemm(A_shared, W_up_shared, up, transpose_B=True)
            else:
                # The execution-tested SM75 GEMM path uses a serial K-loop.
                for ko in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(W_gate[bx * block_N, ko * block_K], W_gate_shared)
                    T.gemm(A_shared, W_gate_shared, gate, transpose_B=True)
                    if variant != "single":
                        T.copy(W_up[bx * block_N, ko * block_K], W_up_shared)
                        T.gemm(A_shared, W_up_shared, up, transpose_B=True)

            if variant == "dual_add":
                for i, j in T.Parallel(block_M, block_N):
                    gate[i, j] += up[i, j]
            elif variant == "swiglu":
                for i, j in T.Parallel(block_M, block_N):
                    x = gate[i, j]
                    gate[i, j] = (x / (1.0 + T.exp2(-x * log2e))) * up[i, j]
            T.copy(gate, Out_shared)
            T.copy(Out_shared, Out[by * block_M, bx * block_N])

    return main


def eager_operation(
    A: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    gate = A @ W_gate.T
    if variant == "single":
        return gate
    up = A @ W_up.T
    if variant == "dual_add":
        return gate + up
    return F.silu(gate) * up


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def time_cuda(call: Callable[[], Any], repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def summarize(samples_us: list[float]) -> dict[str, float]:
    return {
        "p50_us": statistics.median(samples_us),
        "p90_us": percentile(samples_us, 0.90),
        "p99_us": percentile(samples_us, 0.99),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def compile_torch_baseline(
    enabled: bool,
    variant: str,
) -> tuple[str | None, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None]:
    if not enabled:
        return None, None
    try:
        compiled = torch.compile(
            lambda A, W_gate, W_up: eager_operation(A, W_gate, W_up, variant),
            mode="max-autotune",
            fullgraph=True,
        )
        return None, compiled
    except Exception as error:  # pragma: no cover - depends on remote PyTorch build
        return f"{type(error).__name__}: {error}", None


def benchmark_shape(
    M: int,
    K: int,
    N: int,
    label: str,
    *,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    cycles: int,
    repeats: int,
    warmup: int,
    use_torch_compile: bool,
    variant: str,
    schedule_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(0)
    A = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.02
    W_gate = torch.randn((N, K), device="cuda", dtype=torch.float16) * 0.02
    W_up = torch.randn((N, K), device="cuda", dtype=torch.float16) * 0.02

    program = cross_gemm_swiglu(
        M,
        K,
        N,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        variant=variant,
    )
    fused = tilelang.compile(program, target="cuda", out_idx=[3])

    compiled_error, compiled = compile_torch_baseline(use_torch_compile, variant)
    if compiled is not None:
        try:
            compiled(A, W_gate, W_up)
            torch.cuda.synchronize()
        except Exception as error:  # pragma: no cover - remote compiler dependent
            compiled_error = f"{type(error).__name__}: {error}"
            compiled = None
    methods: dict[str, Callable[[], Any]] = {
        "torch_eager": lambda: eager_operation(A, W_gate, W_up, variant),
        "tilelang_fused": lambda: fused(A, W_gate, W_up),
    }
    if compiled is not None:
        methods["torch_compile"] = lambda: compiled(A, W_gate, W_up)

    # Trigger compilation before numerical validation and timing.
    outputs = {name: call() for name, call in methods.items()}
    torch.cuda.synchronize()
    reference = eager_operation(A.float(), W_gate.float(), W_up.float(), variant).half()
    correctness = {}
    for name, output in outputs.items():
        difference = (output - reference).abs()
        correctness[name] = {
            "max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item(),
            "allclose": torch.allclose(output, reference, rtol=2e-2, atol=2e-2),
        }
    if not all(item["allclose"] for item in correctness.values()):
        raise AssertionError(f"numerical validation failed: {correctness}")

    for call in methods.values():
        for _ in range(warmup):
            call()
    torch.cuda.synchronize()

    samples = {name: [] for name in methods}
    base_order = ["torch_eager", "tilelang_fused", "tilelang_fused", "torch_eager"]
    if "torch_compile" in methods:
        base_order = ["torch_eager", "torch_compile", "tilelang_fused", "tilelang_fused", "torch_compile", "torch_eager"]
    for cycle in range(cycles):
        order = base_order if cycle % 2 == 0 else list(reversed(base_order))
        for name in order:
            samples[name].append(time_cuda(methods[name], repeats))

    summaries = {name: summarize(values) for name, values in samples.items()}
    best_baseline = min(
        (name for name in summaries if name != "tilelang_fused"),
        key=lambda name: summaries[name]["p50_us"],
    )
    speedup = summaries[best_baseline]["p50_us"] / summaries["tilelang_fused"]["p50_us"]
    return {
        "label": label,
        "variant": variant,
        "shape": {"M": M, "K": K, "N": N},
        "config": {
            "block_M": block_M,
            "block_N": block_N,
            "block_K": block_K,
            "num_stages": num_stages,
            "threads": threads,
        },
        "schedule_selection": schedule_selection,
        "correctness": correctness,
        "latency": summaries,
        "best_baseline": best_baseline,
        "speedup": speedup,
        "torch_compile_error": compiled_error,
    }


def parse_shape(value: str) -> tuple[int, int, int, str]:
    parts = value.split(":")
    dims = parts[0].lower().split("x")
    if len(dims) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxN[:label]")
    try:
        M, K, N = (int(item) for item in dims)
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from error
    label = parts[1] if len(parts) > 1 else "custom"
    return M, K, N, label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", type=parse_shape, dest="shapes")
    parser.add_argument("--block-m", type=int, default=int(os.environ.get("TILELANG_BENCH_BLOCK_M", "64")))
    parser.add_argument("--block-n", type=int, default=int(os.environ.get("TILELANG_BENCH_BLOCK_N", "64")))
    parser.add_argument("--block-k", type=int, default=int(os.environ.get("TILELANG_BENCH_BLOCK_K", "32")))
    parser.add_argument("--num-stages", type=int, default=int(os.environ.get("TILELANG_BENCH_NUM_STAGES", "0")))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("TILELANG_BENCH_THREADS", "128")))
    schedule_rank = os.environ.get("TILELANG_BENCH_SCHEDULE_RANK")
    parser.add_argument(
        "--schedule-rank",
        type=int,
        default=int(schedule_rank) if schedule_rank is not None else None,
        help="select this zero-based rank from the deterministic resource-aware schedule space",
    )
    parser.add_argument("--cycles", type=int, default=int(os.environ.get("TILELANG_BENCH_CYCLES", "20")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("TILELANG_BENCH_REPEATS", "100")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("TILELANG_BENCH_WARMUP", "20")))
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("single", "dual_add", "swiglu"),
        default=os.environ.get("TILELANG_BENCH_VARIANT", "swiglu"),
    )
    parser.add_argument("--output", type=Path, default=Path("cross-gemm-swiglu-result.json"))
    # Colab executes uploaded files inside ipykernel, which appends its own
    # ``-f kernel.json`` arguments.  They are unrelated to this benchmark.
    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    environment_shapes = os.environ.get("TILELANG_BENCH_SHAPES")
    shapes = tuple(
        args.shapes or ([parse_shape(value) for value in environment_shapes.split(";")] if environment_shapes else DEFAULT_SHAPES)
    )
    started = time.time()
    target_profile = cuda_target_profile()
    results = []
    for M, K, N, label in shapes:
        config = {
            "block_M": args.block_m,
            "block_N": args.block_n,
            "block_K": args.block_k,
            "num_stages": args.num_stages,
            "threads": args.threads,
        }
        selection = None
        if args.schedule_rank is not None:
            workload = CrossGemmWorkload(M=M, K=K, N=N, gemm_count=1 if args.variant == "single" else 2)
            config, selection = select_ranked_schedule(workload, target_profile, args.schedule_rank)
        results.append(
            benchmark_shape(
                M,
                K,
                N,
                label,
                **config,
                cycles=args.cycles,
                repeats=args.repeats,
                warmup=args.warmup,
                use_torch_compile=args.torch_compile,
                variant=args.variant,
                schedule_selection=selection,
            )
        )
    geomean_speedup = math.exp(statistics.mean(math.log(item["speedup"]) for item in results))
    payload = {
        "schema": "tilelang-cross-gemm-swiglu-v2",
        "repository": "nya-a-cat/tilelang",
        "commit": os.environ.get("TILELANG_SOURCE_SHA") or git_head(),
        "native_base_sha": os.environ.get("TILELANG_NATIVE_BASE_SHA"),
        "native_wheel_sha256": os.environ.get("TILELANG_WHEEL_SHA256"),
        "created_unix": started,
        "duration_seconds": time.time() - started,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tilelang": getattr(tilelang, "__version__", None),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "target_profile": target_profile_payload(target_profile),
        },
        "protocol": {
            "cycles": args.cycles,
            "repeats_per_sample": args.repeats,
            "warmup_calls_per_method": args.warmup,
            "statistic": "median CUDA-event kernel latency",
            "order": "alternating baseline/fused/fused/baseline",
            "schedule_policy": "resource_rank_v2" if args.schedule_rank is not None else "manual",
            "schedule_rank": args.schedule_rank,
        },
        "results": results,
        "geomean_speedup": geomean_speedup,
        "gate_1_5x": geomean_speedup >= 1.5,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
