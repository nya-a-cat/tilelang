"""Trace register-only ``T.Parallel`` vectorization without a GPU.

The byte-identical PrimFunc is lowered for explicit CUDA architecture targets
under the default planner, an explicit planner enable, and the legacy gate.
The optional base trace lets fork CI prove that the legacy override reproduces
the exact generated source from the pre-change native wheel. No device compiler
or GPU is used.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import tilelang as tl
import tilelang.language as T
from tilelang import tvm


RESULT_PATH = Path(
    os.environ.get(
        "TILELANG_PARALLEL_VECTOR_TRACE_RESULT",
        "tilelang-parallel-vectorization-trace.json",
    )
)
BASE_TRACE_PATH = os.environ.get("TILELANG_PARALLEL_VECTOR_BASE_TRACE")
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")
ARCHES = tuple(
    arch.strip()
    for arch in os.environ.get(
        "TILELANG_PARALLEL_VECTOR_TRACE_ARCHES",
        "sm_75,sm_80,sm_90a,sm_100a,sm_120a",
    ).split(",")
    if arch.strip()
)
MODES = tuple(
    mode.strip()
    for mode in os.environ.get(
        "TILELANG_PARALLEL_VECTOR_TRACE_MODES",
        "default,planner,legacy",
    ).split(",")
    if mode.strip()
)
EXPECTED_DEFAULT = os.environ.get(
    "TILELANG_PARALLEL_VECTOR_EXPECT_DEFAULT",
    "planner",
)
CONFIG_KEY = "tl.vectorize_local_parallel"
VALID_MODES = {"default", "planner", "legacy"}

if not ARCHES or len(set(ARCHES)) != len(ARCHES):
    raise ValueError("trace architectures must be non-empty and distinct")
if not MODES or len(set(MODES)) != len(MODES) or not set(MODES) <= VALID_MODES:
    raise ValueError(f"trace modes must be distinct members of {sorted(VALID_MODES)}")
if EXPECTED_DEFAULT not in {"planner", "legacy"}:
    raise ValueError("TILELANG_PARALLEL_VECTOR_EXPECT_DEFAULT must be planner or legacy")


@T.prim_func
def local_register_loop():
    with T.Kernel(1):
        x = T.alloc_fragment((256,), T.float32)
        y = T.alloc_fragment((256,), T.float32)
        z = T.alloc_var(T.float32)
        for i in T.Parallel(256):
            y[i] = x[i] * z


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def lower_case(arch: str, mode: str) -> dict[str, object]:
    target = {"kind": "cuda", "arch": arch}
    pass_configs: dict[str, object] = {}
    if mode != "default":
        pass_configs[CONFIG_KEY] = mode == "planner"
    with tvm.transform.PassContext(config=pass_configs), tvm.target.Target(target):
        artifact = tl.lower(local_register_loop, target=target, enable_device_compile=False)
    source = artifact.kernel_source
    if source is None:
        raise RuntimeError(f"missing generated source for {arch}/{mode}")
    float2_occurrences = source.count("float2")
    expected_policy = EXPECTED_DEFAULT if mode == "default" else mode
    if expected_policy == "planner" and float2_occurrences == 0:
        raise RuntimeError(f"planner did not emit float2 for {arch}/{mode}")
    if expected_policy == "legacy" and float2_occurrences != 0:
        raise RuntimeError(
            f"legacy gate unexpectedly emitted float2 for {arch}/{mode}: "
            f"count={float2_occurrences}"
        )
    return {
        "arch": arch,
        "mode": mode,
        "expected_policy": expected_policy,
        "float2_occurrences": float2_occurrences,
        "generated_source_bytes": len(source.encode()),
        "generated_source_sha256": sha256_text(source),
    }


def verify_candidate_relations(cases: list[dict[str, object]]) -> None:
    by_key = {(str(case["arch"]), str(case["mode"])): case for case in cases}
    if {mode for _, mode in by_key} >= {"default", "planner"}:
        for arch in ARCHES:
            default = by_key[(arch, "default")]
            planner = by_key[(arch, "planner")]
            if default["generated_source_sha256"] != planner["generated_source_sha256"]:
                raise RuntimeError(f"default and explicit planner source differ for {arch}")


def verify_legacy_matches_base(cases: list[dict[str, object]]) -> bool | None:
    if BASE_TRACE_PATH is None:
        return None
    base = json.loads(Path(BASE_TRACE_PATH).read_text(encoding="utf-8"))
    primfunc_sha256 = sha256_text(str(local_register_loop))
    if base.get("primfunc_sha256") != primfunc_sha256:
        raise RuntimeError("base and candidate PrimFunc hashes differ")
    base_by_arch = {
        str(case["arch"]): case
        for case in base.get("cases", [])
        if case.get("mode") == "default"
    }
    candidate_by_arch = {
        str(case["arch"]): case
        for case in cases
        if case.get("mode") == "legacy"
    }
    if set(base_by_arch) != set(ARCHES) or set(candidate_by_arch) != set(ARCHES):
        raise RuntimeError("base/default and candidate/legacy architecture sets differ")
    for arch in ARCHES:
        if (
            base_by_arch[arch]["generated_source_sha256"]
            != candidate_by_arch[arch]["generated_source_sha256"]
        ):
            raise RuntimeError(f"candidate legacy source does not match base default for {arch}")
    return True


def main() -> int:
    cases = [lower_case(arch, mode) for arch in ARCHES for mode in MODES]
    verify_candidate_relations(cases)
    legacy_matches_base = verify_legacy_matches_base(cases)
    payload = {
        "schema": "tilelang-parallel-vectorization-trace-v1",
        "repository": "nya-a-cat/tilelang",
        "source_sha": SOURCE_SHA,
        "device_compile": False,
        "gpu_execution": False,
        "config_key": CONFIG_KEY,
        "expected_default": EXPECTED_DEFAULT,
        "primfunc_sha256": sha256_text(str(local_register_loop)),
        "legacy_matches_base": legacy_matches_base,
        "base_trace": BASE_TRACE_PATH,
        "cases": cases,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
