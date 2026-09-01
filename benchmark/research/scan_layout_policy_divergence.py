"""Compile-only discovery of TileLang layout-policy decision divergence.

Every matrix entry constructs one PrimFunc and runs that same object through
the two policies selected by ``TILELANG_LAYOUT_SCAN_POLICIES``. LayoutInference
is compared first. CUDA source lowering runs only when the inferred modules
differ, keeping this scan cheap enough for a CPU-only remote worker. No device
execution is required.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback
from typing import Any

import benchmark_layout_cost_models_t4 as base
import benchmark_layout_normalization_policies_t4 as normalization
import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.layout import Fragment
from tvm.tirx.stmt_functor import post_order_visit


POLICIES = tuple(
    policy.strip()
    for policy in os.environ.get(
        "TILELANG_LAYOUT_SCAN_POLICIES",
        "register-count,io-aware",
    ).split(",")
    if policy.strip()
)
if len(POLICIES) != 2 or len(set(POLICIES)) != 2:
    raise ValueError("TILELANG_LAYOUT_SCAN_POLICIES must name exactly two distinct policies")
TARGET_ARCHES = tuple(
    arch.strip()
    for arch in os.environ.get(
        "TILELANG_LAYOUT_SCAN_ARCHES",
        "sm_75,sm_80,sm_90a,sm_100a,sm_120a",
    ).split(",")
    if arch.strip()
)
RESULT_PATH = Path(
    os.environ.get(
        "TILELANG_LAYOUT_SCAN_RESULT",
        "/content/tilelang-layout-policy-divergence-scan.json",
    )
)
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")


@dataclass(frozen=True)
class ScanCase:
    name: str
    family: str
    program: Any


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def with_symbol(program: Any, symbol: str) -> Any:
    return program.with_attr("global_symbol", symbol)


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


def make_dual_output(
    rows: int,
    cols: int,
    dtype: str,
    threads: int,
    symbol: str,
) -> Any:
    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), dtype),
        B: T.Tensor((rows, cols), dtype),
        BT: T.Tensor((cols, rows), dtype),
    ):
        with T.Kernel(1, threads=threads):
            values = T.alloc_fragment((rows, cols), dtype)
            T.copy(A, values)
            for i, j in T.Parallel(rows, cols):
                B[i, j] = values[i, j]
                BT[j, i] = values[i, j]

    return with_symbol(main, symbol)


def make_permute3d(
    d0: int,
    d1: int,
    d2: int,
    permutation: str,
    dtype: str,
    threads: int,
    symbol: str,
) -> Any:
    if permutation == "021":

        @T.prim_func
        def main(
            A: T.Tensor((d0, d1, d2), dtype),
            B: T.Tensor((d0, d2, d1), dtype),
        ):
            with T.Kernel(1, threads=threads):
                values = T.alloc_fragment((d0, d1, d2), dtype)
                T.copy(A, values)
                for i, j, k in T.Parallel(d0, d1, d2):
                    B[i, k, j] = values[i, j, k]

    elif permutation == "201":

        @T.prim_func
        def main(
            A: T.Tensor((d0, d1, d2), dtype),
            B: T.Tensor((d2, d0, d1), dtype),
        ):
            with T.Kernel(1, threads=threads):
                values = T.alloc_fragment((d0, d1, d2), dtype)
                T.copy(A, values)
                for i, j, k in T.Parallel(d0, d1, d2):
                    B[k, i, j] = values[i, j, k]

    else:
        raise ValueError(f"unsupported permutation: {permutation}")

    return with_symbol(main, symbol)


def build_scan_cases() -> list[ScanCase]:
    cases: list[ScanCase] = []

    for rows, cols, threads in (
        (1, 4096, 128),
        (2, 2560, 256),
        (4, 1024, 128),
        (8, 512, 128),
        (16, 256, 128),
        (4, 4096, 256),
    ):
        name = f"row_broadcast_f32_{rows}x{cols}_t{threads}"
        cases.append(
            ScanCase(
                name,
                "row_broadcast",
                base.make_broadcast(rows, cols, threads, f"scan_{name}"),
            )
        )

    for rows, cols, dtype, threads in (
        (8, 4096, "float32", 256),
        (32, 1024, "float32", 128),
        (64, 256, "float16", 128),
        (128, 128, "float32", 128),
    ):
        name = f"column_broadcast_{dtype}_{rows}x{cols}_t{threads}"
        cases.append(
            ScanCase(
                name,
                "column_broadcast",
                make_column_broadcast(rows, cols, dtype, threads, f"scan_{name}"),
            )
        )

    for rows, cols, dtype, threads in (
        (32, 32, "float32", 64),
        (64, 64, "float16", 128),
        (64, 64, "float32", 128),
        (64, 256, "float16", 128),
        (64, 256, "float32", 128),
        (128, 128, "float16", 128),
        (128, 128, "float32", 128),
        (128, 256, "float16", 256),
        (256, 64, "float32", 128),
    ):
        name = f"transpose_{dtype}_{rows}x{cols}_t{threads}"
        cases.append(
            ScanCase(
                name,
                "transpose",
                base.make_transpose(rows, cols, dtype, threads, f"scan_{name}"),
            )
        )

    for rows, cols, threads in (
        (64, 64, 128),
        (128, 128, 128),
        (64, 256, 128),
        (128, 256, 256),
    ):
        name = f"mixed_f16_f32_{rows}x{cols}_t{threads}"
        cases.append(
            ScanCase(
                name,
                "mixed_dtype",
                base.make_mixed_chain(rows, cols, threads, f"scan_{name}"),
            )
        )

    for rows, cols, dtype, threads in (
        (32, 128, "float16", 128),
        (32, 128, "float32", 128),
        (64, 128, "float16", 128),
        (64, 256, "float32", 256),
    ):
        name = f"affine_{dtype}_{rows}x{cols}_t{threads}"
        cases.append(
            ScanCase(
                name,
                "affine",
                make_affine(rows, cols, dtype, threads, f"scan_{name}"),
            )
        )

    for rows, cols, dtype, threads in (
        (32, 128, "float16", 128),
        (32, 128, "float32", 128),
        (64, 256, "float16", 128),
    ):
        name = f"dual_output_{dtype}_{rows}x{cols}_t{threads}"
        cases.append(
            ScanCase(
                name,
                "dual_output",
                make_dual_output(rows, cols, dtype, threads, f"scan_{name}"),
            )
        )

    for d0, d1, d2, permutation, dtype, threads in (
        (4, 32, 32, "021", "float16", 128),
        (4, 32, 32, "201", "float32", 128),
        (8, 16, 64, "021", "float32", 128),
        (8, 16, 64, "201", "float16", 128),
    ):
        name = f"permute{permutation}_{dtype}_{d0}x{d1}x{d2}_t{threads}"
        cases.append(
            ScanCase(
                name,
                "permute3d",
                make_permute3d(
                    d0,
                    d1,
                    d2,
                    permutation,
                    dtype,
                    threads,
                    f"scan_{name}",
                ),
            )
        )

    for width, block_rows in (
        (1024, 1),
        (1024, 8),
        (1024, 32),
        (4096, 1),
        (4096, 4),
        (8192, 1),
    ):
        rms_name = f"rmsnorm_f32_1280x{width}_bm{block_rows}_t128"
        cases.append(
            ScanCase(
                rms_name,
                "rmsnorm",
                normalization.make_rmsnorm(
                    1280,
                    width,
                    block_rows,
                    f"scan_{rms_name}",
                ),
            )
        )
        softmax_name = f"softmax_f32_1280x{width}_bm{block_rows}_t128"
        cases.append(
            ScanCase(
                softmax_name,
                "softmax",
                normalization.make_softmax(
                    1280,
                    width,
                    block_rows,
                    f"scan_{softmax_name}",
                ),
            )
        )

    for width in (1024, 4096):
        name = f"layernorm_f16_1280x{width}_bm1_t128"
        cases.append(
            ScanCase(
                name,
                "layernorm",
                normalization.make_layernorm(1280, width, 1, f"scan_{name}"),
            )
        )

    for block_rows in (1, 4):
        name = f"rmsnorm_splitk_f32_1280x8192_bm{block_rows}_bk512_t128"
        cases.append(
            ScanCase(
                name,
                "rmsnorm_splitk",
                normalization.make_splitk_rmsnorm(
                    1280,
                    8192,
                    block_rows,
                    512,
                    f"scan_{name}",
                ),
            )
        )

    return cases


def _static_value(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def layout_summary(module: Any) -> dict[str, Any]:
    seen: set[Any] = set()
    buffers: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if not isinstance(node, tvm.tirx.SBlock) or "layout_map" not in node.annotations:
            return
        for buffer, layout in node.annotations["layout_map"].items():
            if buffer in seen or not isinstance(layout, Fragment):
                continue
            seen.add(buffer)
            output_shape = [_static_value(extent) for extent in layout.get_output_shape()]
            static_output_shape = [extent for extent in output_shape if isinstance(extent, int)]
            slots = math.prod(static_output_shape) if len(static_output_shape) == len(output_shape) else None
            element_bits = int(buffer.dtype.bits) * int(buffer.dtype.lanes)
            debug = repr(layout)
            buffers.append(
                {
                    "name": buffer.name,
                    "dtype": str(buffer.dtype),
                    "input_shape": [_static_value(extent) for extent in layout.get_input_shape()],
                    "output_shape": output_shape,
                    "slots": slots,
                    "element_bits": element_bits,
                    "packed_32bit_words_lower_bound": math.ceil(slots * element_bits / 32) if slots is not None else None,
                    "thread_extent": _static_value(layout.get_thread_size()),
                    "replicate_extent": _static_value(layout.replicate_size),
                    "layout_debug": debug,
                    "layout_debug_sha256": sha256_text(debug),
                }
            )

    post_order_visit(module["main"].body, visit)
    buffers.sort(key=lambda entry: entry["name"])
    legacy_slots = [entry["slots"] for entry in buffers]
    packed_words = [entry["packed_32bit_words_lower_bound"] for entry in buffers]
    fingerprint_payload = [
        {
            "name": entry["name"],
            "dtype": entry["dtype"],
            "layout_debug_sha256": entry["layout_debug_sha256"],
        }
        for entry in buffers
    ]
    return {
        "legacy_register_slots": sum(legacy_slots) if all(value is not None for value in legacy_slots) else None,
        "packed_32bit_words_lower_bound": sum(packed_words) if all(value is not None for value in packed_words) else None,
        "layout_fingerprint_sha256": sha256_text(json.dumps(fingerprint_payload, sort_keys=True)),
        "fragment_buffers": buffers,
    }


def infer_layout(program: Any, target: Any, policy: str) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    with target, tvm.transform.PassContext(config={"tl.layout_cost_model": policy}):
        module = tvm.IRModule({"main": program})
        module = tvm.tirx.transform.BindTarget(target)(module)
        module = tl.transform.MaterializeKernelLaunch()(module)
        module = tl.transform.LayoutInference()(module)
    module_text = str(module)
    return module, {
        "inference_seconds": time.perf_counter() - started,
        "inferred_ir_sha256": sha256_text(module_text),
        "layout": layout_summary(module),
    }


def source_features(source: str) -> dict[str, Any]:
    vector_types = Counter(
        re.findall(
            r"\b(?:float|double|half|nv_bfloat16|int|uint)(?:2|4|8|16)\b",
            source,
        )
    )
    return {
        "source_bytes": len(source.encode()),
        "source_lines": len(source.splitlines()),
        "for_loops": source.count("for ("),
        "sync_threads": source.count("__syncthreads"),
        "vector_type_counts": dict(sorted(vector_types.items())),
    }


def lower_source(program: Any, target: Any, policy: str) -> dict[str, Any]:
    started = time.perf_counter()
    with target, tvm.transform.PassContext(config={"tl.layout_cost_model": policy}):
        artifact = tl.lower(
            program,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    source = str(artifact.kernel_source)
    return {
        "lower_seconds": time.perf_counter() - started,
        "generated_source_sha256": sha256_text(source),
        "generated_source": source,
        "source_features": source_features(source),
    }


def scan_case(case: ScanCase, arch: str) -> dict[str, Any]:
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    canonical_ir = str(case.program)
    result: dict[str, Any] = {
        "name": case.name,
        "family": case.family,
        "arch": arch,
        "canonical_primfunc_sha256": sha256_text(canonical_ir),
        "policies": {},
    }
    inferred: dict[str, Any] = {}
    for policy in POLICIES:
        try:
            module, metadata = infer_layout(case.program, target, policy)
            inferred[policy] = module
            result["policies"][policy] = {"status": "inferred", **metadata}
        except Exception as error:  # noqa: BLE001 - retain all other scan evidence
            result["policies"][policy] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }

    if set(inferred) != set(POLICIES):
        result["status"] = "incomplete"
        return result

    baseline_module = inferred[POLICIES[0]]
    candidate_module = inferred[POLICIES[1]]
    structurally_equal = bool(tvm.ir.structural_equal(baseline_module, candidate_module))
    baseline_layout = result["policies"][POLICIES[0]]["layout"]
    candidate_layout = result["policies"][POLICIES[1]]["layout"]
    baseline_slots = baseline_layout["legacy_register_slots"]
    candidate_slots = candidate_layout["legacy_register_slots"]
    result["policy_delta"] = {
        "inferred_ir_structurally_equal": structurally_equal,
        "layout_fingerprint_equal": baseline_layout["layout_fingerprint_sha256"] == candidate_layout["layout_fingerprint_sha256"],
        "candidate_register_slot_ratio": candidate_slots / baseline_slots
        if baseline_slots not in (None, 0) and candidate_slots is not None
        else None,
    }

    if structurally_equal:
        result["policy_delta"]["source_comparison"] = "skipped_structurally_equal"
        result["policy_delta"]["decision_different"] = False
        result["status"] = "complete"
        return result

    lowered: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        try:
            lowered[policy] = lower_source(case.program, target, policy)
            result["policies"][policy].update(lowered[policy])
            result["policies"][policy]["status"] = "lowered"
        except Exception as error:  # noqa: BLE001 - preserve inference divergence
            result["policies"][policy]["status"] = "lower_failed"
            result["policies"][policy]["lower_error"] = f"{type(error).__name__}: {error}"
            result["policies"][policy]["lower_traceback"] = traceback.format_exc()

    if set(lowered) != set(POLICIES):
        result["policy_delta"]["decision_different"] = True
        result["status"] = "incomplete"
        return result

    source_equal = lowered[POLICIES[0]]["generated_source_sha256"] == lowered[POLICIES[1]]["generated_source_sha256"]
    result["policy_delta"]["source_comparison"] = "lowered"
    result["policy_delta"]["generated_source_equal"] = source_equal
    result["policy_delta"]["decision_different"] = True
    result["status"] = "complete"
    return result


def write_payload(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    cases = build_scan_cases()
    payload: dict[str, Any] = {
        "schema": "tilelang-layout-policy-divergence-scan-v1",
        "repository": "nya-a-cat/tilelang",
        "source_sha": SOURCE_SHA,
        "started_unix": started,
        "policies": list(POLICIES),
        "target_arches": list(TARGET_ARCHES),
        "case_count": len(cases),
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "tilelang": tl.__version__,
            "tilelang_file": str(Path(tl.__file__).resolve()),
        },
        "evidence_boundary": (
            "CPU-only compile scan of unchanged PrimFuncs. It discovers policy decision divergence across explicit "
            "CUDA targets and performs no GPU timing, correctness execution, ptxas resource measurement, or global "
            "performance validation."
        ),
        "results": [],
    }
    total = len(TARGET_ARCHES) * len(cases)
    completed = 0
    try:
        for arch in TARGET_ARCHES:
            for case in cases:
                result = scan_case(case, arch)
                payload["results"].append(result)
                completed += 1
                print(
                    f"LAYOUT_SCAN_PROGRESS {completed}/{total} arch={arch} case={case.name} "
                    f"status={result['status']} different={result.get('policy_delta', {}).get('decision_different')}",
                    flush=True,
                )
                payload["completed_entries"] = completed
                payload["duration_seconds"] = time.time() - started
                write_payload(payload)

        complete_results = [result for result in payload["results"] if result["status"] == "complete"]
        divergent_results = [result for result in complete_results if result.get("policy_delta", {}).get("decision_different")]
        lowered_divergent = [result for result in divergent_results if result.get("policy_delta", {}).get("source_comparison") == "lowered"]
        payload["aggregate"] = {
            "total_entries": total,
            "complete_entries": len(complete_results),
            "incomplete_entries": total - len(complete_results),
            "divergent_entries": len(divergent_results),
            "divergent_generated_source_entries": sum(
                not result["policy_delta"].get("generated_source_equal", True) for result in lowered_divergent
            ),
            "divergent_case_names": sorted({result["name"] for result in divergent_results}),
            "divergent_arches": sorted({result["arch"] for result in divergent_results}),
        }
        payload["status"] = "complete" if len(complete_results) == total else "partial"
    except Exception as error:  # noqa: BLE001 - preserve completed scan entries
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
    finally:
        payload["duration_seconds"] = time.time() - started
        write_payload(payload)
        print(
            "TILELANG_LAYOUT_SCAN_RESULT="
            + json.dumps(
                {
                    "path": str(RESULT_PATH),
                    "sha256": hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest(),
                    "status": payload.get("status"),
                    "completed_entries": payload.get("completed_entries", 0),
                    "duration_seconds": payload["duration_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
