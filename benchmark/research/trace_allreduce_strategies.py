"""Verify CUDA AllReduce strategy selection without a GPU.

The script lowers one byte-identical 128-thread reduction for explicit CUDA
targets and JIT strategy overrides. It performs no device compilation or
execution. The emitted record makes the selected template, workspace size,
and generated-source identity auditable in fork-only CI evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import tilelang as tl
import tilelang.language as T
from tilelang import tvm


RESULT_PATH = Path(os.environ.get("TILELANG_ALLREDUCE_TRACE_RESULT", "tilelang-allreduce-strategy-trace.json"))
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")
ARCHES = ("sm_75", "sm_80")
STRATEGIES = ("auto", "butterfly", "hierarchical")
EXPECTED_HIERARCHICAL = {
    ("sm_75", "auto"): False,
    ("sm_75", "butterfly"): False,
    ("sm_75", "hierarchical"): True,
    ("sm_80", "auto"): True,
    ("sm_80", "butterfly"): False,
    ("sm_80", "hierarchical"): True,
}
ALLREDUCE_CALL_RE = re.compile(r"tl::AllReduce<[^;]+?::run")


@T.prim_func
def reduction(
    A: T.Tensor((1024,), T.float32),
    B: T.Tensor((1,), T.float32),
):
    with T.Kernel(1, threads=128):
        values = T.alloc_fragment((1024,), T.float32)
        total = T.alloc_fragment((1,), T.float32)
        T.copy(A, values)
        T.reduce_sum(values, total, dim=0)
        if T.get_thread_binding() == 0:
            B[0] = total[0]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def lower_case(arch: str, strategy: str) -> dict[str, object]:
    target = {"kind": "cuda", "arch": arch}
    pass_configs = {tl.PassConfigKey.TL_CUDA_ALLREDUCE_STRATEGY: strategy}
    with tvm.transform.PassContext(config=pass_configs), tvm.target.Target(target):
        artifact = tl.lower(reduction, target=target, enable_device_compile=False)
    source = artifact.kernel_source
    if source is None:
        raise RuntimeError(f"missing generated source for {arch}/{strategy}")
    calls = ALLREDUCE_CALL_RE.findall(source)
    if len(calls) != 1:
        raise RuntimeError(f"expected one AllReduce call for {arch}/{strategy}, got {len(calls)}")
    shared_values = [int(func.attrs.get("dyn_shared_memory_buf", 0)) for func in artifact.device_mod.functions.values()]
    if len(shared_values) != 1:
        raise RuntimeError(f"expected one device function for {arch}/{strategy}")

    hierarchical = ", true>::run" in calls[0]
    expected_hierarchical = EXPECTED_HIERARCHICAL[(arch, strategy)]
    expected_shared_bytes = 4 * 4 if expected_hierarchical else 128 * 4
    if hierarchical != expected_hierarchical:
        raise RuntimeError(
            f"unexpected algorithm for {arch}/{strategy}: hierarchical={hierarchical}, expected={expected_hierarchical}"
        )
    if shared_values[0] != expected_shared_bytes:
        raise RuntimeError(
            f"unexpected workspace for {arch}/{strategy}: {shared_values[0]}, expected={expected_shared_bytes}"
        )
    return {
        "arch": arch,
        "strategy": strategy,
        "hierarchical": hierarchical,
        "dynamic_shared_bytes": shared_values[0],
        "allreduce_call": calls[0],
        "generated_source_sha256": sha256_text(source),
    }


def main() -> int:
    cases = [lower_case(arch, strategy) for arch in ARCHES for strategy in STRATEGIES]
    payload = {
        "schema": "tilelang-allreduce-strategy-trace-v1",
        "repository": "nya-a-cat/tilelang",
        "source_sha": SOURCE_SHA,
        "device_compile": False,
        "gpu_execution": False,
        "primfunc_sha256": sha256_text(str(reduction)),
        "cases": cases,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
