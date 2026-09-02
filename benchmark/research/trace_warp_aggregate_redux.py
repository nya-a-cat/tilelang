"""Compile hierarchical AllReduce hardware-aggregate variants without a GPU.

The frozen kernels isolate the second level of TileLang's hierarchical CUDA
AllReduce. Each byte-identical PrimFunc is lowered twice from one installed
build: the default hardware-redux policy and its pass-config rollback. The
generated CUDA is compiled and disassembled for explicit NVIDIA targets; no
GPU is initialized or executed.
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
CONFIG_KEY = "tl.disable_warp_aggregate_redux"
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")
ARCHES = tuple(
    value.strip()
    for value in os.environ.get(
        "TILELANG_WARP_AGGREGATE_REDUX_ARCHES",
        "sm_75,sm_80,sm_90a,sm_100a,sm_120a",
    ).split(",")
    if value.strip()
)
MAX_WORKERS = int(os.environ.get("TILELANG_WARP_AGGREGATE_REDUX_WORKERS", "4"))
RESULT_PATH = Path(
    os.environ.get(
        "TILELANG_WARP_AGGREGATE_REDUX_RESULT",
        "tilelang-warp-aggregate-redux.json",
    )
)
REPORT_PATH = Path(
    os.environ.get(
        "TILELANG_WARP_AGGREGATE_REDUX_REPORT",
        "tilelang-warp-aggregate-redux.md",
    )
)
RAW_DIR = Path(
    os.environ.get(
        "TILELANG_WARP_AGGREGATE_REDUX_RAW_DIR",
        "tilelang-warp-aggregate-redux-raw",
    )
)
RAW_ARCHIVE_PATH = Path(
    os.environ.get(
        "TILELANG_WARP_AGGREGATE_REDUX_RAW_ARCHIVE",
        "tilelang-warp-aggregate-redux-raw.zip",
    )
)
MODES = ("default", "rollback")
CASES = (
    {"name": "f32_max_t128", "dtype": "float32", "op": "max", "threads": 128},
    {"name": "f32_min_t1024", "dtype": "float32", "op": "min", "threads": 1024},
    {"name": "f16_max_t256", "dtype": "float16", "op": "max", "threads": 256},
    {"name": "i32_sum_t128", "dtype": "int32", "op": "sum", "threads": 128},
    {"name": "i32_sum_t1024", "dtype": "int32", "op": "sum", "threads": 1024},
    {"name": "i32_max_t128", "dtype": "int32", "op": "max", "threads": 128},
    {"name": "i32_min_t256", "dtype": "int32", "op": "min", "threads": 256},
    {"name": "i32_bitor_t1024", "dtype": "int32", "op": "bitor", "threads": 1024},
    {"name": "i32_bitxor_t256", "dtype": "int32", "op": "bitxor", "threads": 256},
    {"name": "i32_bitxor_t1024", "dtype": "int32", "op": "bitxor", "threads": 1024},
)
ALLREDUCE_CALL_RE = re.compile(r"tl::AllReduce<[^;]+?::run(?:_batch)?")
SASS_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)",
    re.MULTILINE,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def make_kernel(case: dict[str, Any]):
    width = case["threads"]
    threads = case["threads"]
    dtype = case["dtype"]
    op = case["op"]

    @T.prim_func
    def kernel(A: T.Tensor((width,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=threads):
            values = T.alloc_fragment((width,), dtype)
            result = T.alloc_fragment((1,), dtype)
            T.copy(A, values)
            if op == "max":
                T.reduce_max(values, result, dim=0, clear=True)
            elif op == "min":
                T.reduce_min(values, result, dim=0, clear=True)
            elif op == "bitor":
                T.reduce_bitor(values, result, dim=0, clear=True)
            elif op == "sum":
                T.reduce_sum(values, result, dim=0, clear=True)
            elif op == "bitxor":
                T.reduce_bitxor(values, result, dim=0, clear=True)
            if T.get_thread_binding() == 0:
                B[0] = result[0]

    return kernel


def lower_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    compile_inputs: list[dict[str, Any]] = []
    for spec in CASES:
        prim_func = make_kernel(spec)
        primfunc_sha256 = sha256_text(str(prim_func))
        for arch in ARCHES:
            for mode in MODES:
                target = tvm.target.Target({"kind": "cuda", "arch": arch})
                pass_configs = {CONFIG_KEY: mode == "rollback"}
                started = time.perf_counter()
                with tvm.transform.PassContext(opt_level=3, config=pass_configs), target:
                    artifact = tl.lower(
                        prim_func,
                        target=target,
                        enable_device_compile=False,
                    )
                source = str(artifact.kernel_source or "")
                if not source.strip():
                    raise RuntimeError(f"empty CUDA source for {spec['name']}/{arch}/{mode}")
                calls = ALLREDUCE_CALL_RE.findall(source)
                if len(calls) != 1:
                    raise RuntimeError(
                        f"expected one AllReduce call for {spec['name']}/{arch}/{mode}, got {len(calls)}"
                    )
                hierarchical = ", true" in calls[0]
                rollback_marker = ", true, false" in calls[0]
                expected_hierarchical = arch != "sm_75"
                if hierarchical != expected_hierarchical:
                    raise RuntimeError(
                        f"unexpected AllReduce strategy for {spec['name']}/{arch}/{mode}: {calls[0]}"
                    )
                if rollback_marker != (expected_hierarchical and mode == "rollback"):
                    raise RuntimeError(
                        f"unexpected rollback marker for {spec['name']}/{arch}/{mode}: {calls[0]}"
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
                    "allreduce_call": calls[0],
                    "hierarchical": hierarchical,
                    "rollback_marker": rollback_marker,
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
            "redux": count_prefixes("REDUX"),
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
        tilelang_callback_cuda_compile(
            item["source"],
            item["target"],
            item["pass_configs"],
        )
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
        raise RuntimeError("nvdisasm is required for the warp-aggregate-redux trace")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compile_case, item, nvdisasm): item for item in inputs}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            compiled = future.result()
            item["record"].update(compiled)
            print(
                f"compiled {item['record']['name']}/{item['record']['arch']}/{item['record']['mode']}: "
                f"{compiled['instruction_count']} instructions, "
                f"{compiled['groups']['redux']} redux, {compiled['groups']['shuffle']} shuffle"
            )


def build_comparisons(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(case["name"], case["arch"], case["mode"]): case for case in records}
    comparisons = []
    eligible_comparisons = 0
    shuffle_reductions = 0
    strict_instruction_reductions = 0
    strict_instruction_reductions_by_dtype: Counter[str] = Counter()
    strict_instruction_reductions_by_op: Counter[str] = Counter()
    for spec in CASES:
        for arch in ARCHES:
            default = by_key[(spec["name"], arch, "default")]
            rollback = by_key[(spec["name"], arch, "rollback")]
            group_delta = {
                name: default["groups"].get(name, 0) - rollback["groups"].get(name, 0)
                for name in sorted(set(default["groups"]) | set(rollback["groups"]))
            }
            comparison = {
                "name": spec["name"],
                "dtype": spec["dtype"],
                "op": spec["op"],
                "threads": spec["threads"],
                "arch": arch,
                "default": default,
                "rollback": rollback,
                "default_minus_rollback": {
                    "instructions": default["instruction_count"] - rollback["instruction_count"],
                    "cubin_bytes": default["cubin_bytes"] - rollback["cubin_bytes"],
                    "groups": group_delta,
                },
            }
            # The current target feature table enables scalar floating-point
            # redux on SM100a. CUDA's integer warp-reduce intrinsics are
            # available from SM80 onward. nvdisasm's human-readable Blackwell
            # spelling is not stable enough to make a REDUX mnemonic count a
            # correctness gate, so the gate uses the eliminated aggregate
            # shuffles and total instruction count instead.
            eligible = (
                spec["dtype"] == "int32"
                and arch != "sm_75"
                and (
                    spec["op"] in ("max", "min", "bitor")
                    or (spec["op"] in ("sum", "bitxor") and spec["threads"] == 1024)
                )
            ) or (spec["dtype"] in ("float32", "float16") and arch == "sm_100a")
            comparison["eligible"] = eligible
            if eligible:
                eligible_comparisons += 1
                if group_delta["shuffle"] >= 0:
                    raise RuntimeError(
                        f"hardware redux did not replace aggregate shuffles for {spec['name']}/{arch}: {group_delta}"
                    )
                shuffle_reductions += 1
                instruction_delta = comparison["default_minus_rollback"]["instructions"]
                if instruction_delta > 0:
                    raise RuntimeError(
                        f"hardware redux increased instructions for {spec['name']}/{arch}: "
                        f"{default['instruction_count']} vs {rollback['instruction_count']}"
                    )
                if instruction_delta < 0:
                    strict_instruction_reductions += 1
                    strict_instruction_reductions_by_dtype[spec["dtype"]] += 1
                    strict_instruction_reductions_by_op[spec["op"]] += 1
            else:
                for field in ("instruction_count", "cubin_bytes"):
                    if default[field] != rollback[field]:
                        raise RuntimeError(
                            f"ineligible control changed {field} for {spec['name']}/{arch}: "
                            f"{default[field]} vs {rollback[field]}"
                        )
                if default["groups"] != rollback["groups"]:
                    raise RuntimeError(f"ineligible control changed opcode groups for {spec['name']}/{arch}")
            for mode_case in (default, rollback):
                if any(
                    usage["n_spills"] or usage["spill_load_bytes"] or usage["spill_store_bytes"]
                    for usage in mode_case["resources"].values()
                ):
                    raise RuntimeError(f"spill detected in {spec['name']}/{arch}/{mode_case['mode']}")
            comparisons.append(comparison)
    minimum_strict_reductions = 6
    if strict_instruction_reductions < minimum_strict_reductions:
        raise RuntimeError(
            "hardware redux did not meet the frozen matrix instruction-reduction floor: "
            f"{strict_instruction_reductions} vs {minimum_strict_reductions}"
        )
    missing_dtype_reductions = sorted(
        dtype
        for dtype in ("float32", "float16", "int32")
        if strict_instruction_reductions_by_dtype[dtype] == 0
    )
    if missing_dtype_reductions:
        raise RuntimeError(
            "hardware redux has no strict instruction reduction for dtypes: "
            + ", ".join(missing_dtype_reductions)
        )
    missing_op_reductions = sorted(
        op
        for op in {case["op"] for case in CASES}
        if strict_instruction_reductions_by_op[op] == 0
    )
    if missing_op_reductions:
        raise RuntimeError(
            "hardware redux has no strict instruction reduction for ops: "
            + ", ".join(missing_op_reductions)
        )
    acceptance = {
        "eligible_comparisons": eligible_comparisons,
        "shuffle_reductions": shuffle_reductions,
        "strict_instruction_reductions": strict_instruction_reductions,
        "minimum_strict_instruction_reductions": minimum_strict_reductions,
        "strict_instruction_reductions_by_dtype": dict(
            sorted(strict_instruction_reductions_by_dtype.items())
        ),
        "strict_instruction_reductions_by_op": dict(
            sorted(strict_instruction_reductions_by_op.items())
        ),
    }
    return comparisons, acceptance


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Warp-aggregate hardware redux CUBIN/SASS trace",
        "",
        f"- Source: `{payload['source_sha']}`",
        f"- Matrix: `{len(CASES)}` kernels x `{len(ARCHES)}` targets x 2 same-build modes.",
        "- GPU execution: `false`.",
        f"- Eligible comparisons with fewer aggregate shuffles: "
        f"`{payload['acceptance']['shuffle_reductions']}/"
        f"{payload['acceptance']['eligible_comparisons']}`.",
        f"- Strict instruction-count reductions: "
        f"`{payload['acceptance']['strict_instruction_reductions']}` "
        f"(required: `{payload['acceptance']['minimum_strict_instruction_reductions']}`).",
        "",
        "| Case | Target | Instructions | Redux | Shuffle | Registers |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["comparisons"]:
        default = item["default"]
        rollback = item["rollback"]

        def max_regs(case: dict[str, Any]) -> int:
            return max(value["n_regs"] for value in case["resources"].values())

        lines.append(
            f"| `{item['name']}` | `{item['arch']}` | "
            f"{rollback['instruction_count']}→{default['instruction_count']} | "
            f"{rollback['groups']['redux']}→{default['groups']['redux']} | "
            f"{rollback['groups']['shuffle']}→{default['groups']['shuffle']} | "
            f"{max_regs(rollback)}→{max_regs(default)} |"
        )
    lines.extend(
        [
            "",
            "The default and rollback modes lower the same PrimFunc from one installed build. "
            "The rollback sets `tl.disable_warp_aggregate_redux=True`. Static CUBIN/SASS "
            "differences are an acceptance screen; latency and end-to-end speedup require real GPU timing.",
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
        raise ValueError("warp-aggregate-redux architectures must be distinct")
    if MAX_WORKERS < 1:
        raise ValueError("warp-aggregate-redux worker count must be positive")
    if RAW_DIR.exists():
        raise RuntimeError(f"raw output directory already exists: {RAW_DIR}")
    started = time.time()
    records, compile_inputs = lower_cases()
    compile_all(compile_inputs)
    comparisons, acceptance = build_comparisons(records)
    payload = {
        "schema": "tilelang-warp-aggregate-redux-v1",
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
