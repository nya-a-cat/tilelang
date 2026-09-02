"""Compile FP32x2 reduction A/B variants without initializing a GPU.

The matrix uses one installed TileLang build and lowers each byte-identical
PrimFunc twice. The default mode enables SM100+ packed FP32x2 accumulation;
the rollback mode sets ``tl.enable_fp32x2_reduction=False``. Generated CUDA,
CUBIN, SASS, and ptxas resources are retained as static screening evidence.
"""

from __future__ import annotations

import concurrent.futures
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
import zipfile

import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.contrib.cuda_resource_info import pop_recorded, reset_recorder
from tilelang.cuda.backend import tilelang_callback_cuda_compile


REPOSITORY = "nya-a-cat/tilelang"
CONFIG_KEY = "tl.enable_fp32x2_reduction"
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")
ARCHES = tuple(
    value.strip()
    for value in os.environ.get(
        "TILELANG_FP32X2_REDUCTION_ARCHES",
        "sm_80,sm_90a,sm_100a,sm_120a",
    ).split(",")
    if value.strip()
)
MAX_WORKERS = int(os.environ.get("TILELANG_FP32X2_REDUCTION_WORKERS", "4"))
RESULT_PATH = Path(
    os.environ.get(
        "TILELANG_FP32X2_REDUCTION_RESULT",
        "tilelang-fp32x2-reduction.json",
    )
)
REPORT_PATH = Path(
    os.environ.get(
        "TILELANG_FP32X2_REDUCTION_REPORT",
        "tilelang-fp32x2-reduction.md",
    )
)
RAW_DIR = Path(
    os.environ.get(
        "TILELANG_FP32X2_REDUCTION_RAW_DIR",
        "tilelang-fp32x2-reduction-raw",
    )
)
RAW_ARCHIVE_PATH = Path(
    os.environ.get(
        "TILELANG_FP32X2_REDUCTION_RAW_ARCHIVE",
        "tilelang-fp32x2-reduction-raw.zip",
    )
)
MODES = ("default", "rollback")
CASES = (
    {"name": "row_sum_w128_t128", "kind": "row_reduce", "op": "sum", "width": 128, "threads": 128},
    {"name": "row_sum_w1024_t128", "kind": "row_reduce", "op": "sum", "width": 1024, "threads": 128},
    {
        "name": "row_abssum_w1024_t128",
        "kind": "row_reduce",
        "op": "abssum",
        "width": 1024,
        "threads": 128,
    },
    {"name": "rmsnorm_w1024_t128", "kind": "rmsnorm", "op": "sum", "width": 1024, "threads": 128},
    {"name": "rmsnorm_w4096_t128", "kind": "rmsnorm", "op": "sum", "width": 4096, "threads": 128},
    {"name": "softmax_w1024_t128", "kind": "softmax", "op": "sum", "width": 1024, "threads": 128},
    {"name": "softmax_w4096_t128", "kind": "softmax", "op": "sum", "width": 4096, "threads": 128},
    {
        "name": "batched_sum_r4_w512_t128_b2",
        "kind": "batched_reduce",
        "op": "sum",
        "rows": 4,
        "width": 512,
        "threads": 128,
        "batch": 2,
    },
)
SASS_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)",
    re.MULTILINE,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def make_kernel(case: dict[str, Any]):
    kind = case["kind"]
    width = case["width"]
    threads = case["threads"]

    if kind == "row_reduce":
        op = case["op"]

        @T.prim_func
        def kernel(A: T.Tensor((1, width), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=threads) as bx:
                values = T.alloc_fragment((width,), T.float32)
                result = T.alloc_fragment((1,), T.float32)
                T.copy(A[bx, 0], values)
                if op == "sum":
                    T.reduce_sum(values, result, dim=0)
                elif op == "abssum":
                    T.reduce_abssum(values, result, dim=0)
                B[bx] = result[0]

        return kernel

    if kind == "rmsnorm":

        @T.prim_func
        def kernel(A: T.Tensor((1, width), T.float32), B: T.Tensor((1, width), T.float32)):
            with T.Kernel(1, threads=threads) as bx:
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

        return kernel

    if kind == "softmax":

        @T.prim_func
        def kernel(A: T.Tensor((1, width), T.float32), B: T.Tensor((1, width), T.float32)):
            with T.Kernel(1, threads=threads) as bx:
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

        return kernel

    if kind == "batched_reduce":
        rows = case["rows"]
        batch = case["batch"]
        vec_size = 8

        def fragment_layout(i, j):
            linear = i * width + j
            thread_id = linear // vec_size % threads
            local_id = linear // (threads * vec_size) * vec_size + linear % vec_size
            return thread_id, local_id

        @T.prim_func
        def kernel(A: T.Tensor((rows, width), T.float32), B: T.Tensor((rows,), T.float32)):
            with T.Kernel(1, threads=threads):
                values = T.alloc_fragment((rows, width), T.float32)
                result = T.alloc_fragment((rows,), T.float32)
                T.annotate_layout({values: T.Fragment(values.shape, forward_fn=fragment_layout)})
                T.copy(A, values)
                T.reduce_sum(values, result, dim=1, batch=batch)
                T.copy(result, B)

        return kernel

    raise ValueError(f"unknown FP32x2 trace case kind: {kind}")


def lower_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    compile_inputs: list[dict[str, Any]] = []
    for spec in CASES:
        prim_func = make_kernel(spec)
        primfunc_sha256 = sha256_text(str(prim_func))
        for arch in ARCHES:
            for mode in MODES:
                target = tvm.target.Target({"kind": "cuda", "arch": arch})
                pass_configs = {
                    CONFIG_KEY: mode == "default",
                    "tl.enable_reducer_plan_verbose": True,
                }
                started = time.perf_counter()
                with tvm.transform.PassContext(opt_level=3, config=pass_configs), target:
                    artifact = tl.lower(prim_func, target=target, enable_device_compile=False)
                source = str(artifact.kernel_source or "")
                if not source.strip():
                    raise RuntimeError(f"empty CUDA source for {spec['name']}/{arch}/{mode}")
                packed_markers = source.count("tl::add2") + source.count("SumOp_f32x2")
                if mode == "rollback" and packed_markers:
                    raise RuntimeError(
                        f"rollback emitted FP32x2 for {spec['name']}/{arch}: {packed_markers}"
                    )
                if arch not in ("sm_100a", "sm_120a") and packed_markers:
                    raise RuntimeError(
                        f"pre-SM100 control emitted FP32x2 for {spec['name']}/{arch}/{mode}"
                    )
                case_dir = RAW_DIR / spec["name"] / arch / mode
                case_dir.mkdir(parents=True, exist_ok=True)
                source_path = case_dir / "kernel.cu"
                source_path.write_text(source, encoding="utf-8")
                record = {
                    **spec,
                    "arch": arch,
                    "mode": mode,
                    "primfunc_sha256": primfunc_sha256,
                    "lower_seconds": time.perf_counter() - started,
                    "packed_source_markers": packed_markers,
                    "source_sha256": sha256_text(source),
                    "source_bytes": len(source.encode()),
                    "source_path": source_path.relative_to(RAW_DIR).as_posix(),
                }
                records.append(record)
                compile_inputs.append(
                    {
                        "record": record,
                        "source": source,
                        "source_path": source_path,
                        "target": target,
                        "pass_configs": pass_configs,
                    }
                )
    return records, compile_inputs


def parse_sass(sass: str) -> dict[str, Any]:
    opcodes = Counter(match.group("opcode").upper() for match in SASS_INSTRUCTION_RE.finditer(sass))
    if not opcodes:
        raise RuntimeError("nvdisasm output contains no recognized instructions")

    def count_prefixes(*prefixes: str) -> int:
        return sum(count for opcode, count in opcodes.items() if opcode.startswith(prefixes))

    return {
        "sass_sha256": sha256_text(sass),
        "sass_chars": len(sass),
        "instruction_count": sum(opcodes.values()),
        "groups": {
            "barrier": count_prefixes("BAR", "MBAR"),
            "floating_add": count_prefixes("FADD"),
            "packed_floating": count_prefixes("FADD2", "FMNMX2"),
            "shuffle": count_prefixes("SHFL"),
            "shared_load": count_prefixes("LDS", "LDSM"),
            "shared_store": count_prefixes("STS"),
            "local_load": count_prefixes("LDL"),
            "local_store": count_prefixes("STL"),
        },
        "opcodes": dict(opcodes.most_common()),
    }


def compile_case(item: dict[str, Any], nvdisasm: str) -> dict[str, Any]:
    reset_recorder()
    started = time.perf_counter()
    cubin = bytes(
        tilelang_callback_cuda_compile(item["source"], item["target"], item["pass_configs"])
    )
    compile_seconds = time.perf_counter() - started
    resources = {name: asdict(value) for name, value in sorted(pop_recorded().items())}
    case_dir = item["source_path"].parent
    cubin_path = case_dir / "kernel.cubin"
    sass_path = case_dir / "kernel.sass"
    cubin_path.write_bytes(cubin)
    sass = subprocess.run(
        [nvdisasm, str(cubin_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    ).stdout
    sass_path.write_text(sass, encoding="utf-8")
    return {
        "compile_seconds": compile_seconds,
        "cubin_sha256": sha256_bytes(cubin),
        "cubin_bytes": len(cubin),
        "cubin_path": cubin_path.relative_to(RAW_DIR).as_posix(),
        "sass_path": sass_path.relative_to(RAW_DIR).as_posix(),
        "resources": resources,
        **parse_sass(sass),
    }


def compile_all(inputs: list[dict[str, Any]]) -> None:
    nvdisasm = os.environ.get("NVDISASM") or shutil.which("nvdisasm")
    if nvdisasm is None:
        cuda_home = os.environ.get("CUDA_HOME")
        candidate = Path(cuda_home) / "bin/nvdisasm" if cuda_home else None
        if candidate is not None and candidate.is_file():
            nvdisasm = str(candidate)
    if nvdisasm is None or not Path(nvdisasm).is_file():
        raise RuntimeError("nvdisasm is required for the FP32x2 reduction trace")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compile_case, item, nvdisasm): item for item in inputs}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            compiled = future.result()
            item["record"].update(compiled)
            print(
                f"compiled {item['record']['name']}/{item['record']['arch']}/{item['record']['mode']}: "
                f"{compiled['instruction_count']} instructions, "
                f"{compiled['groups']['floating_add']} floating adds"
            )


def max_regs(case: dict[str, Any]) -> int:
    return max(value["n_regs"] for value in case["resources"].values())


def build_comparisons(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(case["name"], case["arch"], case["mode"]): case for case in records}
    comparisons = []
    eligible_comparisons = 0
    strict_instruction_reductions = 0
    equal_instruction_comparisons = 0
    instruction_regressions = 0
    register_reductions = 0
    register_regressions = 0
    for spec in CASES:
        for arch in ARCHES:
            default = by_key[(spec["name"], arch, "default")]
            rollback = by_key[(spec["name"], arch, "rollback")]
            group_delta = {
                name: default["groups"].get(name, 0) - rollback["groups"].get(name, 0)
                for name in sorted(set(default["groups"]) | set(rollback["groups"]))
            }
            instruction_delta = default["instruction_count"] - rollback["instruction_count"]
            register_delta = max_regs(default) - max_regs(rollback)
            # Some realistic one-row reductions remain scalar even on SM100+;
            # preserve them as exact coverage controls. Eligibility is derived
            # from emitted code instead of being inferred from architecture.
            eligible = default["packed_source_markers"] > 0
            comparison = {
                "name": spec["name"],
                "kind": spec["kind"],
                "op": spec["op"],
                "width": spec["width"],
                "threads": spec["threads"],
                "arch": arch,
                "eligible": eligible,
                "default": default,
                "rollback": rollback,
                "default_minus_rollback": {
                    "instructions": instruction_delta,
                    "registers": register_delta,
                    "cubin_bytes": default["cubin_bytes"] - rollback["cubin_bytes"],
                    "groups": group_delta,
                },
            }
            if eligible:
                eligible_comparisons += 1
                strict_instruction_reductions += instruction_delta < 0
                equal_instruction_comparisons += instruction_delta == 0
                instruction_regressions += instruction_delta > 0
                register_reductions += register_delta < 0
                register_regressions += register_delta > 0
            else:
                for field in ("source_sha256", "instruction_count", "cubin_bytes"):
                    if default[field] != rollback[field]:
                        raise RuntimeError(
                            f"pre-SM100 control changed {field} for {spec['name']}/{arch}"
                        )
                if default["groups"] != rollback["groups"]:
                    raise RuntimeError(
                        f"scalar control changed opcode groups for {spec['name']}/{arch}"
                    )
            for mode_case in (default, rollback):
                if any(
                    usage["n_spills"] or usage["spill_load_bytes"] or usage["spill_store_bytes"]
                    for usage in mode_case["resources"].values()
                ):
                    raise RuntimeError(f"spill detected in {spec['name']}/{arch}/{mode_case['mode']}")
            comparisons.append(comparison)
    if instruction_regressions:
        raise RuntimeError(f"FP32x2 reduction introduced {instruction_regressions} instruction regressions")
    required_wide_gains = {"rmsnorm_w4096_t128", "softmax_w4096_t128"}
    observed_wide_gains = {
        item["name"]
        for item in comparisons
        if item["arch"] == "sm_100a"
        and item["eligible"]
        and item["default_minus_rollback"]["instructions"] < 0
    }
    missing_wide_gains = sorted(required_wide_gains - observed_wide_gains)
    if missing_wide_gains:
        raise RuntimeError(f"missing strict SM100 wide-reduction gains: {missing_wide_gains}")
    acceptance = {
        "eligible_comparisons": eligible_comparisons,
        "strict_instruction_reductions": strict_instruction_reductions,
        "equal_instruction_comparisons": equal_instruction_comparisons,
        "instruction_regressions": instruction_regressions,
        "register_reductions": register_reductions,
        "register_regressions": register_regressions,
        "controls_exact": True,
        "zero_spill": True,
        "candidate_gate": "no-instruction-regression-and-wide-sm100-gain",
    }
    return comparisons, acceptance


def write_report(payload: dict[str, Any]) -> None:
    acceptance = payload["acceptance"]
    lines = [
        "# FP32x2 reduction CUBIN/SASS trace",
        "",
        f"- Source: `{payload['source_sha']}`",
        f"- Matrix: `{len(CASES)}` kernels x `{len(ARCHES)}` targets x 2 same-build modes.",
        "- GPU execution: `false`.",
        f"- Eligible SM100+ comparisons: `{acceptance['eligible_comparisons']}`.",
        f"- Instruction outcomes (better/equal/worse): "
        f"`{acceptance['strict_instruction_reductions']}/"
        f"{acceptance['equal_instruction_comparisons']}/"
        f"{acceptance['instruction_regressions']}`.",
        f"- Register outcomes (better/equal/worse): "
        f"`{acceptance['register_reductions']}/"
        f"{acceptance['eligible_comparisons'] - acceptance['register_reductions'] - acceptance['register_regressions']}/"
        f"{acceptance['register_regressions']}`.",
        "",
        "| Case | Target | Instructions | FADD | Packed float | Registers |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["comparisons"]:
        default = item["default"]
        rollback = item["rollback"]
        lines.append(
            f"| `{item['name']}` | `{item['arch']}` | "
            f"{rollback['instruction_count']}→{default['instruction_count']} | "
            f"{rollback['groups']['floating_add']}→{default['groups']['floating_add']} | "
            f"{rollback['groups']['packed_floating']}→{default['groups']['packed_floating']} | "
            f"{max_regs(rollback)}→{max_regs(default)} |"
        )
    lines.extend(
        [
            "",
            "The default and rollback modes lower the same PrimFunc from one installed build. "
            "The rollback sets `tl.enable_fp32x2_reduction=False`. Static machine-code "
            "differences guide the target policy; latency and end-to-end speedup require real GPU timing.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def archive_raw() -> None:
    RAW_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RAW_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(RAW_DIR).as_posix())


def main() -> int:
    if not ARCHES or len(set(ARCHES)) != len(ARCHES):
        raise ValueError("FP32x2 reduction architectures must be distinct")
    if MAX_WORKERS < 1:
        raise ValueError("FP32x2 reduction worker count must be positive")
    if RAW_DIR.exists():
        raise RuntimeError(f"raw output directory already exists: {RAW_DIR}")
    started = time.time()
    records, compile_inputs = lower_cases()
    compile_all(compile_inputs)
    comparisons, acceptance = build_comparisons(records)
    payload = {
        "schema": "tilelang-fp32x2-reduction-v1",
        "repository": REPOSITORY,
        "source_sha": SOURCE_SHA,
        "architectures": ARCHES,
        "config_key": CONFIG_KEY,
        "device_compile": True,
        "gpu_execution": False,
        "duration_seconds": time.time() - started,
        "cases": CASES,
        "acceptance": acceptance,
        "comparisons": comparisons,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    archive_raw()
    print(
        json.dumps(
            {
                "status": "complete",
                "comparisons": len(comparisons),
                "acceptance": acceptance,
                "result": str(RESULT_PATH),
                "report": str(REPORT_PATH),
                "raw_archive": str(RAW_ARCHIVE_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
