"""Compile frozen real-workload vectorization variants without a GPU.

This diagnostic elaborates the exact regression configurations behind the two
largest public wins associated with register-only ``T.Parallel`` vectorization:
DeepSeek mHC prefill's large fused kernel and the 4096^3 W4A8 dequantization
GEMM.  Each byte-identical PrimFunc is lowered with the planner and legacy gate,
compiled with TileLang's CUDA callback, and disassembled for explicit NVIDIA
architectures.  It never initializes or executes a GPU.
"""

from __future__ import annotations

import concurrent.futures
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Any
import zipfile

import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.contrib.cuda_resource_info import pop_recorded, reset_recorder
from tilelang.cuda.backend import tilelang_callback_cuda_compile


REPOSITORY = "nya-a-cat/tilelang"
CONFIG_KEY = "tl.vectorize_local_parallel"
INPLACE_CONFIG_KEY = "tl.storage_rewrite_detect_inplace"
REGISTER_USAGE_CONFIG_KEY = "tl.ptxas_register_usage_level"
DISABLE_REINTERPRET_CONFIG_KEY = "tl.disable_reinterpret_vectorization"
DISABLE_INT4X2_UNPACK_CONFIG_KEY = "tl.disable_int4x2_unpack_peephole"
MODES = tuple(
    mode.strip()
    for mode in os.environ.get(
        "TILELANG_REAL_VECTOR_MODES",
        "planner,legacy",
    ).split(",")
    if mode.strip()
)
WORKLOAD_NAMES = tuple(
    name.strip()
    for name in os.environ.get("TILELANG_REAL_VECTOR_WORKLOADS", "").split(",")
    if name.strip()
)
BASE_MODES = {
    "planner",
    "legacy",
    "planner_inplace",
    "planner_reinterpret_legacy",
    "planner_int4x2_unpack_legacy",
}
REGISTER_USAGE_MODE_RE = re.compile(r"planner_ru(?P<level>\d+)$")
LAUNCH_BOUNDS_MODE_RE = re.compile(r"planner_lb(?P<blocks>\d+)$")
DEFAULT_ARCHES = "sm_75,sm_80,sm_90a,sm_100a,sm_120a"
RESULT_PATH = Path(
    os.environ.get(
        "TILELANG_REAL_VECTOR_RESULT",
        "tilelang-real-vectorization-workloads.json",
    )
)
REPORT_PATH = Path(
    os.environ.get(
        "TILELANG_REAL_VECTOR_REPORT",
        "tilelang-real-vectorization-workloads.md",
    )
)
RAW_ARCHIVE_PATH = Path(
    os.environ.get(
        "TILELANG_REAL_VECTOR_RAW_ARCHIVE",
        "tilelang-real-vectorization-workloads-raw.zip",
    )
)
RAW_DIR = Path(
    os.environ.get(
        "TILELANG_REAL_VECTOR_RAW_DIR",
        "tilelang-real-vectorization-workloads-raw",
    )
)
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")
ARCHES = tuple(
    arch.strip()
    for arch in os.environ.get("TILELANG_REAL_VECTOR_ARCHES", DEFAULT_ARCHES).split(",")
    if arch.strip()
)
MAX_WORKERS = int(os.environ.get("TILELANG_REAL_VECTOR_COMPILE_WORKERS", "4"))
SASS_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)",
    re.MULTILINE,
)
PACKED_TYPE_RE = re.compile(
    r"\b(?:float|double|int|uint|short|ushort|char|uchar|half|bfloat)[234]\b"
)

if not ARCHES or len(set(ARCHES)) != len(ARCHES):
    raise ValueError("TILELANG_REAL_VECTOR_ARCHES must contain distinct architectures")
if len(set(WORKLOAD_NAMES)) != len(WORKLOAD_NAMES):
    raise ValueError("TILELANG_REAL_VECTOR_WORKLOADS must contain distinct workload names")
invalid_modes = [
    mode
    for mode in MODES
    if mode not in BASE_MODES
    and REGISTER_USAGE_MODE_RE.fullmatch(mode) is None
    and LAUNCH_BOUNDS_MODE_RE.fullmatch(mode) is None
]
invalid_register_levels = [
    mode
    for mode in MODES
    if (match := REGISTER_USAGE_MODE_RE.fullmatch(mode)) is not None
    and not 0 <= int(match.group("level")) <= 10
]
invalid_launch_bounds = [
    mode
    for mode in MODES
    if (match := LAUNCH_BOUNDS_MODE_RE.fullmatch(mode)) is not None
    and not 2 <= int(match.group("blocks")) <= 8
]
if (
    not MODES
    or len(set(MODES)) != len(MODES)
    or not {"planner", "legacy"} <= set(MODES)
    or invalid_modes
    or invalid_register_levels
    or invalid_launch_bounds
):
    raise ValueError(
        "TILELANG_REAL_VECTOR_MODES must contain planner and legacy; optional modes are "
        "planner_inplace, planner_reinterpret_legacy, planner_int4x2_unpack_legacy, "
        "planner_ru0 through planner_ru10, and planner_lb2 through planner_lb8"
    )
if MAX_WORKERS < 1:
    raise ValueError("TILELANG_REAL_VECTOR_COMPILE_WORKERS must be positive")


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


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_workloads(repo_root: Path) -> list[dict[str, Any]]:
    mhc_path = repo_root / "examples/deepseek_mhc/example_mhc_pre.py"
    w4a8_path = repo_root / "examples/dequantize_gemm/example_dequant_gemm_w4a8.py"
    for path in (mhc_path, w4a8_path):
        if not path.is_file():
            raise RuntimeError(f"frozen example entrypoint is missing: {path}")

    mhc = load_module("tilelang_trace_example_mhc_pre", mhc_path)
    w4a8 = load_module("tilelang_trace_example_dequant_gemm_w4a8", w4a8_path)

    tokens = 2048
    hidden_size = 4096
    hc_mult = 4
    n_splits = 1
    hc_mult3 = hc_mult * (2 + hc_mult)
    mhc_jit = mhc.mhc_pre_big_fuse_tilelang
    mhc_prim = mhc_jit.get_tir(
        T.Tensor((n_splits, tokens, hc_mult3), T.float32),
        T.Tensor((n_splits, tokens), T.float32),
        T.Tensor((3,), T.float32),
        T.Tensor((hc_mult3,), T.float32),
        T.Tensor((tokens, hc_mult, hidden_size), T.bfloat16),
        T.Tensor((tokens, hc_mult), T.float32),
        T.Tensor((tokens, hc_mult * hc_mult), T.float32),
        T.Tensor((tokens, hidden_size), T.bfloat16),
        hidden_size=hidden_size,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=10,
        n_splits=n_splits,
        hc_mult=hc_mult,
    )

    w4a8_jit = w4a8.matmul_int8xint4.jit_impl
    w4a8_prim = w4a8_jit.get_tir(
        4096,
        4096,
        4096,
        "int8",
        "int32",
        "int32",
        num_bits=4,
        block_M=32,
        block_N=32,
        block_K=128,
        num_stages=1,
        threads=128,
    )

    return [
        {
            "name": "deepseek_mhc_pre_big_fuse",
            "example_path": mhc_path,
            "prim_func": mhc_prim,
            "pass_configs": dict(mhc_jit.pass_configs or {}),
            "configuration": {
                "tokens": tokens,
                "hidden_size": hidden_size,
                "hc_mult": hc_mult,
                "n_splits": n_splits,
                "sinkhorn_repeat": 10,
                "rms_eps": 1e-6,
                "hc_pre_eps": 1e-6,
                "hc_sinkhorn_eps": 1e-6,
                "hc_post_mult_value": 1.0,
            },
            "scope_note": "large fused kernel from the sequential mHC pre regression",
        },
        {
            "name": "dequant_gemm_w4a8_4096",
            "example_path": w4a8_path,
            "prim_func": w4a8_prim,
            "pass_configs": dict(w4a8_jit.pass_configs or {}),
            "configuration": {
                "M": 4096,
                "N": 4096,
                "K": 4096,
                "in_dtype": "int8",
                "out_dtype": "int32",
                "accum_dtype": "int32",
                "num_bits": 4,
                "block_M": 32,
                "block_N": 32,
                "block_K": 128,
                "num_stages": 1,
                "threads": 128,
            },
            "scope_note": "exact fixed configuration used by run_regression_perf",
        },
    ]


def packed_types(source: str) -> dict[str, int]:
    return dict(sorted(Counter(PACKED_TYPE_RE.findall(source)).items()))


def register_usage_level(mode: str) -> int | None:
    match = REGISTER_USAGE_MODE_RE.fullmatch(mode)
    return int(match.group("level")) if match is not None else None


def launch_bounds_blocks(mode: str) -> int | None:
    match = LAUNCH_BOUNDS_MODE_RE.fullmatch(mode)
    return int(match.group("blocks")) if match is not None else None


def rewrite_launch_bounds(source: str, blocks: int) -> tuple[str, int]:
    """Change only CUDA's min-blocks launch-bound argument.

    The lowering pipeline is kept byte-identical across the scan.  Rewriting
    generated source isolates ptxas register allocation from every TileLang IR
    and schedule decision.
    """

    rewritten, count = re.subn(
        r"(__launch_bounds__\(\s*\d+\s*,\s*)1(\s*\))",
        rf"\g<1>{blocks}\g<2>",
        source,
    )
    if count == 0:
        raise RuntimeError("launch-bound scan found no CUDA min-blocks argument")
    return rewritten, count


def lower_sources(workloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workload_records: list[dict[str, Any]] = []
    compile_inputs: list[dict[str, Any]] = []
    for workload in workloads:
        prim_func = workload["prim_func"]
        primfunc_text = str(prim_func)
        record = {
            "name": workload["name"],
            "example_path": workload["example_path"].relative_to(Path.cwd()).as_posix(),
            "example_sha256": sha256_file(workload["example_path"]),
            "primfunc_sha256": sha256_text(primfunc_text),
            "configuration": workload["configuration"],
            "scope_note": workload["scope_note"],
            "function_pass_configs": {
                str(key): value for key, value in workload["pass_configs"].items()
            },
            "cases": [],
        }
        for arch in ARCHES:
            for mode in MODES:
                target = tvm.target.Target({"kind": "cuda", "arch": arch})
                pass_configs = dict(workload["pass_configs"])
                pass_configs[CONFIG_KEY] = mode != "legacy"
                pass_configs[INPLACE_CONFIG_KEY] = mode == "planner_inplace"
                pass_configs[DISABLE_REINTERPRET_CONFIG_KEY] = (
                    mode == "planner_reinterpret_legacy"
                )
                pass_configs[DISABLE_INT4X2_UNPACK_CONFIG_KEY] = (
                    mode == "planner_int4x2_unpack_legacy"
                )
                usage_level = register_usage_level(mode)
                if usage_level is not None:
                    pass_configs[REGISTER_USAGE_CONFIG_KEY] = usage_level
                started = time.perf_counter()
                with tvm.transform.PassContext(opt_level=3, config=pass_configs), target:
                    artifact = tl.lower(
                        prim_func,
                        target=target,
                        enable_device_compile=False,
                    )
                source = str(artifact.kernel_source or "")
                if not source.strip():
                    raise RuntimeError(f"empty CUDA source for {workload['name']}/{arch}/{mode}")
                lowered_source_sha256 = sha256_text(source)
                launch_blocks = launch_bounds_blocks(mode)
                launch_bounds_rewrites = 0
                if launch_blocks is not None:
                    source, launch_bounds_rewrites = rewrite_launch_bounds(
                        source, launch_blocks
                    )
                case_dir = RAW_DIR / workload["name"] / arch / mode
                case_dir.mkdir(parents=True, exist_ok=True)
                source_path = case_dir / "kernel.cu"
                source_path.write_text(source, encoding="utf-8")
                case = {
                    "arch": arch,
                    "mode": mode,
                    "lower_seconds": time.perf_counter() - started,
                    "source_sha256": sha256_text(source),
                    "lowered_source_sha256": lowered_source_sha256,
                    "source_bytes": len(source.encode()),
                    "packed_types": packed_types(source),
                    "packed_type_occurrences": len(PACKED_TYPE_RE.findall(source)),
                    "int4x2_unpack_occurrences": source.count("tl_unpack_int4x2("),
                    "launch_bounds_min_blocks": launch_blocks or 1,
                    "launch_bounds_rewrites": launch_bounds_rewrites,
                    "source_path": source_path.relative_to(RAW_DIR).as_posix(),
                }
                record["cases"].append(case)
                compile_inputs.append(
                    {
                        "workload": workload["name"],
                        "arch": arch,
                        "mode": mode,
                        "source": source,
                        "source_path": source_path,
                        "pass_configs": pass_configs,
                        "case": case,
                    }
                )
        workload_records.append(record)
    return workload_records, compile_inputs


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
            "warp_sync": count_prefixes("WARPSYNC"),
            "shuffle": count_prefixes("SHFL"),
            "redux": count_prefixes("REDUX"),
            "shared_load": count_prefixes("LDS", "LDSM"),
            "shared_store": count_prefixes("STS"),
            "global_load": count_prefixes("LDG"),
            "global_store": count_prefixes("STG"),
            "local_load": count_prefixes("LDL"),
            "local_store": count_prefixes("STL"),
        },
        "opcodes": dict(opcodes.most_common()),
    }


def compile_case(item: dict[str, Any], nvdisasm: str) -> dict[str, Any]:
    target = tvm.target.Target({"kind": "cuda", "arch": item["arch"]})
    reset_recorder()
    started = time.perf_counter()
    cubin = bytes(
        tilelang_callback_cuda_compile(
            item["source"],
            target,
            item["pass_configs"],
        )
    )
    compile_seconds = time.perf_counter() - started
    resources = {
        name: asdict(usage)
        for name, usage in sorted(pop_recorded().items())
    }
    case_dir = item["source_path"].parent
    cubin_path = case_dir / "kernel.cubin"
    sass_path = case_dir / "kernel.sass"
    cubin_path.write_bytes(cubin)
    disassembly = subprocess.run(
        [nvdisasm, str(cubin_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    ).stdout
    sass_path.write_text(disassembly, encoding="utf-8")
    return {
        "compile_seconds": compile_seconds,
        "cubin_sha256": sha256_bytes(cubin),
        "cubin_bytes": len(cubin),
        "cubin_path": cubin_path.relative_to(RAW_DIR).as_posix(),
        "sass_path": sass_path.relative_to(RAW_DIR).as_posix(),
        "resources": resources,
        **parse_sass(disassembly),
    }


def compile_all(compile_inputs: list[dict[str, Any]]) -> None:
    nvdisasm = os.environ.get("NVDISASM") or shutil.which("nvdisasm")
    if nvdisasm is not None and not Path(nvdisasm).is_file():
        raise RuntimeError(f"configured NVDISASM does not exist: {nvdisasm}")
    if nvdisasm is None:
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            candidate = Path(cuda_home) / "bin/nvdisasm"
            if candidate.is_file():
                nvdisasm = str(candidate)
    if nvdisasm is None:
        raise RuntimeError("nvdisasm is required for the real-workload trace")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(compile_case, item, nvdisasm): item
            for item in compile_inputs
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            compiled = future.result()
            item["case"].update(compiled)
            print(
                f"compiled {item['workload']}/{item['arch']}/{item['mode']}: "
                f"{compiled['instruction_count']} instructions, "
                f"{compiled['cubin_bytes']} cubin bytes"
            )


def enrich_comparisons(workload_records: list[dict[str, Any]]) -> dict[str, Any]:
    source_changed = 0
    sass_changed = 0
    planner_packed_gain = 0
    comparisons = 0
    inplace_source_changed = 0
    inplace_sass_changed = 0
    inplace_instruction_reduced = 0
    inplace_register_reduced = 0
    usage_modes = [mode for mode in MODES if register_usage_level(mode) is not None]
    usage_register_reduced = 0
    usage_spill_free = 0
    usage_instruction_reduced = 0
    reinterpret_source_changed = 0
    reinterpret_sass_changed = 0
    reinterpret_instruction_reduced = 0
    reinterpret_register_reduced = 0
    int4x2_source_changed = 0
    int4x2_sass_changed = 0
    int4x2_instruction_reduced = 0
    int4x2_register_reduced = 0
    for workload in workload_records:
        by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
        workload["comparisons"] = []
        for arch in ARCHES:
            planner = by_key[(arch, "planner")]
            legacy = by_key[(arch, "legacy")]
            source_delta = planner["source_sha256"] != legacy["source_sha256"]
            sass_delta = planner["sass_sha256"] != legacy["sass_sha256"]
            packed_delta = planner["packed_type_occurrences"] - legacy["packed_type_occurrences"]
            group_delta = {
                group: planner["groups"].get(group, 0) - legacy["groups"].get(group, 0)
                for group in sorted(set(planner["groups"]) | set(legacy["groups"]))
            }
            workload["comparisons"].append(
                {
                    "arch": arch,
                    "source_changed": source_delta,
                    "sass_changed": sass_delta,
                    "planner_minus_legacy": {
                        "source_bytes": planner["source_bytes"] - legacy["source_bytes"],
                        "packed_type_occurrences": packed_delta,
                        "cubin_bytes": planner["cubin_bytes"] - legacy["cubin_bytes"],
                        "instruction_count": planner["instruction_count"] - legacy["instruction_count"],
                        "groups": group_delta,
                    },
                }
            )
            source_changed += int(source_delta)
            sass_changed += int(sass_delta)
            planner_packed_gain += int(packed_delta > 0)
            comparisons += 1
        if "planner_inplace" in MODES:
            workload["inplace_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                inplace = by_key[(arch, "planner_inplace")]
                source_delta = inplace["source_sha256"] != planner["source_sha256"]
                sass_delta = inplace["sass_sha256"] != planner["sass_sha256"]
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner["resources"].values()),
                    default=0,
                )
                inplace_regs = max(
                    (int(item["n_regs"]) for item in inplace["resources"].values()),
                    default=0,
                )
                workload["inplace_comparisons"].append(
                    {
                        "arch": arch,
                        "source_changed": source_delta,
                        "sass_changed": sass_delta,
                        "inplace_minus_planner": {
                            "source_bytes": inplace["source_bytes"] - planner["source_bytes"],
                            "cubin_bytes": inplace["cubin_bytes"] - planner["cubin_bytes"],
                            "instruction_count": inplace["instruction_count"] - planner["instruction_count"],
                            "max_registers": inplace_regs - planner_regs,
                        },
                    }
                )
                inplace_source_changed += int(source_delta)
                inplace_sass_changed += int(sass_delta)
                inplace_instruction_reduced += int(
                    inplace["instruction_count"] < planner["instruction_count"]
                )
                inplace_register_reduced += int(inplace_regs < planner_regs)
        if usage_modes:
            workload["register_usage_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                planner_resources = list(planner["resources"].values())
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner_resources),
                    default=0,
                )
                for mode in usage_modes:
                    candidate = by_key[(arch, mode)]
                    resources = list(candidate["resources"].values())
                    candidate_regs = max(
                        (int(item["n_regs"]) for item in resources),
                        default=0,
                    )
                    spill_stores = sum(int(item["spill_store_bytes"]) for item in resources)
                    spill_loads = sum(int(item["spill_load_bytes"]) for item in resources)
                    comparison = {
                        "arch": arch,
                        "mode": mode,
                        "register_usage_level": register_usage_level(mode),
                        "source_changed": candidate["source_sha256"] != planner["source_sha256"],
                        "candidate_minus_planner": {
                            "cubin_bytes": candidate["cubin_bytes"] - planner["cubin_bytes"],
                            "instruction_count": candidate["instruction_count"]
                            - planner["instruction_count"],
                            "max_registers": candidate_regs - planner_regs,
                            "spill_store_bytes": spill_stores,
                            "spill_load_bytes": spill_loads,
                        },
                    }
                    workload["register_usage_comparisons"].append(comparison)
                    usage_register_reduced += int(candidate_regs < planner_regs)
                    usage_spill_free += int(spill_stores == 0 and spill_loads == 0)
                    usage_instruction_reduced += int(
                        candidate["instruction_count"] < planner["instruction_count"]
                    )
        if "planner_reinterpret_legacy" in MODES:
            workload["reinterpret_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                legacy_reinterpret = by_key[(arch, "planner_reinterpret_legacy")]
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner["resources"].values()),
                    default=0,
                )
                legacy_regs = max(
                    (
                        int(item["n_regs"])
                        for item in legacy_reinterpret["resources"].values()
                    ),
                    default=0,
                )
                comparison = {
                    "arch": arch,
                    "source_changed": planner["source_sha256"]
                    != legacy_reinterpret["source_sha256"],
                    "sass_changed": planner["sass_sha256"]
                    != legacy_reinterpret["sass_sha256"],
                    "planner_minus_legacy_reinterpret": {
                        "source_bytes": planner["source_bytes"]
                        - legacy_reinterpret["source_bytes"],
                        "cubin_bytes": planner["cubin_bytes"]
                        - legacy_reinterpret["cubin_bytes"],
                        "instruction_count": planner["instruction_count"]
                        - legacy_reinterpret["instruction_count"],
                        "max_registers": planner_regs - legacy_regs,
                    },
                }
                workload["reinterpret_comparisons"].append(comparison)
                reinterpret_source_changed += int(comparison["source_changed"])
                reinterpret_sass_changed += int(comparison["sass_changed"])
                reinterpret_instruction_reduced += int(
                    planner["instruction_count"] < legacy_reinterpret["instruction_count"]
                )
                reinterpret_register_reduced += int(planner_regs < legacy_regs)
        if "planner_int4x2_unpack_legacy" in MODES:
            workload["int4x2_unpack_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                legacy_unpack = by_key[(arch, "planner_int4x2_unpack_legacy")]
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner["resources"].values()),
                    default=0,
                )
                legacy_regs = max(
                    (int(item["n_regs"]) for item in legacy_unpack["resources"].values()),
                    default=0,
                )
                comparison = {
                    "arch": arch,
                    "source_changed": planner["source_sha256"]
                    != legacy_unpack["source_sha256"],
                    "sass_changed": planner["sass_sha256"]
                    != legacy_unpack["sass_sha256"],
                    "planner_minus_legacy_int4x2_unpack": {
                        "source_bytes": planner["source_bytes"]
                        - legacy_unpack["source_bytes"],
                        "cubin_bytes": planner["cubin_bytes"]
                        - legacy_unpack["cubin_bytes"],
                        "instruction_count": planner["instruction_count"]
                        - legacy_unpack["instruction_count"],
                        "max_registers": planner_regs - legacy_regs,
                        "helper_occurrences": planner["int4x2_unpack_occurrences"]
                        - legacy_unpack["int4x2_unpack_occurrences"],
                    },
                }
                workload["int4x2_unpack_comparisons"].append(comparison)
                int4x2_source_changed += int(comparison["source_changed"])
                int4x2_sass_changed += int(comparison["sass_changed"])
                int4x2_instruction_reduced += int(
                    planner["instruction_count"] < legacy_unpack["instruction_count"]
                )
                int4x2_register_reduced += int(planner_regs < legacy_regs)
    if source_changed == 0 or sass_changed == 0 or planner_packed_gain == 0:
        raise RuntimeError(
            "planner/legacy trace produced no material source, SASS, or packed-type difference"
        )
    if "planner_int4x2_unpack_legacy" in MODES:
        w4a8 = next(
            workload
            for workload in workload_records
            if workload["name"] == "dequant_gemm_w4a8_4096"
        )
        by_key = {(case["arch"], case["mode"]): case for case in w4a8["cases"]}
        for arch in ARCHES:
            planner = by_key[(arch, "planner")]
            legacy_unpack = by_key[(arch, "planner_int4x2_unpack_legacy")]
            sm_version = int(re.match(r"sm_(\d+)", arch).group(1))
            if sm_version >= 100 and planner["int4x2_unpack_occurrences"] <= 0:
                raise RuntimeError(f"int4x2 unpack peephole did not trigger for {arch}")
            if sm_version < 100 and planner["int4x2_unpack_occurrences"] != 0:
                raise RuntimeError(f"int4x2 unpack peephole unexpectedly triggered for {arch}")
            if legacy_unpack["int4x2_unpack_occurrences"] != 0:
                raise RuntimeError(f"int4x2 unpack rollback still emitted helper for {arch}")
    return {
        "comparisons": comparisons,
        "source_changed": source_changed,
        "sass_changed": sass_changed,
        "planner_packed_gain": planner_packed_gain,
        "inplace_comparisons": len(ARCHES) * len(workload_records)
        if "planner_inplace" in MODES
        else 0,
        "inplace_source_changed": inplace_source_changed,
        "inplace_sass_changed": inplace_sass_changed,
        "inplace_instruction_reduced": inplace_instruction_reduced,
        "inplace_register_reduced": inplace_register_reduced,
        "register_usage_comparisons": len(ARCHES) * len(workload_records) * len(usage_modes),
        "register_usage_register_reduced": usage_register_reduced,
        "register_usage_spill_free": usage_spill_free,
        "register_usage_instruction_reduced": usage_instruction_reduced,
        "reinterpret_comparisons": len(ARCHES) * len(workload_records)
        if "planner_reinterpret_legacy" in MODES
        else 0,
        "reinterpret_source_changed": reinterpret_source_changed,
        "reinterpret_sass_changed": reinterpret_sass_changed,
        "reinterpret_instruction_reduced": reinterpret_instruction_reduced,
        "reinterpret_register_reduced": reinterpret_register_reduced,
        "int4x2_unpack_comparisons": len(ARCHES) * len(workload_records)
        if "planner_int4x2_unpack_legacy" in MODES
        else 0,
        "int4x2_unpack_source_changed": int4x2_source_changed,
        "int4x2_unpack_sass_changed": int4x2_sass_changed,
        "int4x2_unpack_instruction_reduced": int4x2_instruction_reduced,
        "int4x2_unpack_register_reduced": int4x2_register_reduced,
    }


def enrich_launch_bounds_scan(
    workload_records: list[dict[str, Any]], aggregate: dict[str, Any]
) -> None:
    launch_modes = [mode for mode in MODES if launch_bounds_blocks(mode) is not None]
    if not launch_modes:
        aggregate.update(
            {
                "launch_bounds_comparisons": 0,
                "launch_bounds_register_reduced": 0,
                "launch_bounds_instruction_nonregressed": 0,
                "launch_bounds_spill_free": 0,
                "launch_bounds_static_candidates": 0,
            }
        )
        return

    register_reduced = 0
    instruction_nonregressed = 0
    spill_free = 0
    static_candidates = 0
    comparisons = 0

    def max_resource(case: dict[str, Any], field: str) -> int:
        return max(
            (int(resource[field]) for resource in case["resources"].values()),
            default=0,
        )

    def sum_resource(case: dict[str, Any], field: str) -> int:
        return sum(int(resource[field]) for resource in case["resources"].values())

    for workload in workload_records:
        by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
        workload["launch_bounds_comparisons"] = []
        for arch in ARCHES:
            planner = by_key[(arch, "planner")]
            planner_regs = max_resource(planner, "n_regs")
            for mode in launch_modes:
                candidate = by_key[(arch, mode)]
                candidate_regs = max_resource(candidate, "n_regs")
                spill_stores = sum_resource(candidate, "spill_store_bytes")
                spill_loads = sum_resource(candidate, "spill_load_bytes")
                source_isolated = (
                    candidate["lowered_source_sha256"]
                    == planner["lowered_source_sha256"]
                )
                if not source_isolated:
                    raise RuntimeError(
                        f"launch-bound mode changed TileLang lowering for "
                        f"{workload['name']}/{arch}/{mode}"
                    )
                reduced = candidate_regs < planner_regs
                nonregressed = (
                    candidate["instruction_count"] <= planner["instruction_count"]
                )
                zero_spill = spill_stores == 0 and spill_loads == 0
                viable = reduced and nonregressed and zero_spill
                comparison = {
                    "arch": arch,
                    "mode": mode,
                    "min_blocks_per_sm": launch_bounds_blocks(mode),
                    "source_isolated": source_isolated,
                    "static_candidate": viable,
                    "candidate_minus_planner": {
                        "cubin_bytes": candidate["cubin_bytes"]
                        - planner["cubin_bytes"],
                        "instruction_count": candidate["instruction_count"]
                        - planner["instruction_count"],
                        "max_registers": candidate_regs - planner_regs,
                        "spill_store_bytes": spill_stores,
                        "spill_load_bytes": spill_loads,
                    },
                }
                workload["launch_bounds_comparisons"].append(comparison)
                register_reduced += int(reduced)
                instruction_nonregressed += int(nonregressed)
                spill_free += int(zero_spill)
                static_candidates += int(viable)
                comparisons += 1

    aggregate.update(
        {
            "launch_bounds_comparisons": comparisons,
            "launch_bounds_register_reduced": register_reduced,
            "launch_bounds_instruction_nonregressed": instruction_nonregressed,
            "launch_bounds_spill_free": spill_free,
            "launch_bounds_static_candidates": static_candidates,
        }
    )


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Frozen real-workload vectorization trace",
        "",
        f"- Source commit: `{payload['source_sha']}`",
        f"- Architectures: `{', '.join(payload['architectures'])}`",
        f"- Modes: `{', '.join(payload['modes'])}` from one installed TileLang build",
        "- Device compilation: yes, through TileLang's CUDA callback",
        "- GPU execution: no",
        "",
        "| Workload | Arch | packed types legacy→planner | instructions legacy→planner | registers legacy→planner | local load/store legacy→planner |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for workload in payload["workloads"]:
        by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
        for arch in payload["architectures"]:
            planner = by_key[(arch, "planner")]
            legacy = by_key[(arch, "legacy")]

            def register_count(case: dict[str, Any]) -> str:
                values = sorted({int(item["n_regs"]) for item in case["resources"].values()})
                return "/".join(map(str, values)) if values else "n/a"

            lines.append(
                f"| `{workload['name']}` | `{arch}` | "
                f"{legacy['packed_type_occurrences']}→{planner['packed_type_occurrences']} | "
                f"{legacy['instruction_count']}→{planner['instruction_count']} | "
                f"{register_count(legacy)}→{register_count(planner)} | "
                f"{legacy['groups']['local_load']}/{legacy['groups']['local_store']}→"
                f"{planner['groups']['local_load']}/{planner['groups']['local_store']} |"
            )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "This report proves exact frozen-program elaboration, CUDA source changes, device compilation, CUBIN/SASS changes, and compiler resource usage. Runtime latency and end-to-end speedup require real GPU execution.",
            "",
        ]
    )
    if "planner_inplace" in payload["modes"]:
        lines.extend(
            [
                "## Experimental inplace detector",
                "",
                "| Workload | Arch | instructions planner→inplace | registers planner→inplace | source/SASS changed |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                inplace = by_key[(arch, "planner_inplace")]

                def max_registers(case: dict[str, Any]) -> str:
                    values = [int(item["n_regs"]) for item in case["resources"].values()]
                    return str(max(values)) if values else "n/a"

                lines.append(
                    f"| `{workload['name']}` | `{arch}` | "
                    f"{planner['instruction_count']}→{inplace['instruction_count']} | "
                    f"{max_registers(planner)}→{max_registers(inplace)} | "
                    f"{planner['source_sha256'] != inplace['source_sha256']}/"
                    f"{planner['sass_sha256'] != inplace['sass_sha256']} |"
                )
        lines.append("")
    usage_modes = [mode for mode in payload["modes"] if register_usage_level(mode) is not None]
    if usage_modes:
        lines.extend(
            [
                "## PTXAS register-usage scan",
                "",
                "| Workload | Arch | level | instructions planner→candidate | registers planner→candidate | spill stores/loads (bytes) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner["resources"].values()),
                    default=0,
                )
                for mode in usage_modes:
                    candidate = by_key[(arch, mode)]
                    resources = list(candidate["resources"].values())
                    candidate_regs = max(
                        (int(item["n_regs"]) for item in resources),
                        default=0,
                    )
                    spill_stores = sum(int(item["spill_store_bytes"]) for item in resources)
                    spill_loads = sum(int(item["spill_load_bytes"]) for item in resources)
                    lines.append(
                        f"| `{workload['name']}` | `{arch}` | {register_usage_level(mode)} | "
                        f"{planner['instruction_count']}→{candidate['instruction_count']} | "
                        f"{planner_regs}→{candidate_regs} | {spill_stores}/{spill_loads} |"
                    )
        lines.append("")
    launch_modes = [
        mode for mode in payload["modes"] if launch_bounds_blocks(mode) is not None
    ]
    if launch_modes:
        lines.extend(
            [
                "## CUDA launch-bound scan",
                "",
                "| Workload | Arch | min blocks/SM | instructions planner→candidate | registers planner→candidate | spill stores/loads (bytes) | static candidate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {
                (case["arch"], case["mode"]): case for case in workload["cases"]
            }
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                planner_regs = max(
                    (
                        int(resource["n_regs"])
                        for resource in planner["resources"].values()
                    ),
                    default=0,
                )
                comparisons = {
                    comparison["mode"]: comparison
                    for comparison in workload["launch_bounds_comparisons"]
                    if comparison["arch"] == arch
                }
                for mode in launch_modes:
                    candidate = by_key[(arch, mode)]
                    candidate_regs = max(
                        (
                            int(resource["n_regs"])
                            for resource in candidate["resources"].values()
                        ),
                        default=0,
                    )
                    delta = comparisons[mode]["candidate_minus_planner"]
                    lines.append(
                        f"| `{workload['name']}` | `{arch}` | "
                        f"{launch_bounds_blocks(mode)} | "
                        f"{planner['instruction_count']}→{candidate['instruction_count']} | "
                        f"{planner_regs}→{candidate_regs} | "
                        f"{delta['spill_store_bytes']}/{delta['spill_load_bytes']} | "
                        f"{comparisons[mode]['static_candidate']} |"
                    )
        lines.append("")
    if "planner_reinterpret_legacy" in payload["modes"]:
        lines.extend(
            [
                "## Transparent reinterpret planning",
                "",
                "| Workload | Arch | packed types legacy→transparent | instructions legacy→transparent | registers legacy→transparent |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                legacy_reinterpret = by_key[(arch, "planner_reinterpret_legacy")]

                def register_count(case: dict[str, Any]) -> str:
                    values = sorted(
                        {int(item["n_regs"]) for item in case["resources"].values()}
                    )
                    return "/".join(map(str, values)) if values else "n/a"

                lines.append(
                    f"| `{workload['name']}` | `{arch}` | "
                    f"{legacy_reinterpret['packed_type_occurrences']}→"
                    f"{planner['packed_type_occurrences']} | "
                    f"{legacy_reinterpret['instruction_count']}→"
                    f"{planner['instruction_count']} | "
                    f"{register_count(legacy_reinterpret)}→{register_count(planner)} |"
                )
        lines.append("")
    if "planner_int4x2_unpack_legacy" in payload["modes"]:
        lines.extend(
            [
                "## Packed signed-int4x2 unpack",
                "",
                "| Workload | Arch | helper legacy→peephole | instructions legacy→peephole | registers legacy→peephole |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                legacy_unpack = by_key[(arch, "planner_int4x2_unpack_legacy")]

                def register_count(case: dict[str, Any]) -> str:
                    values = sorted(
                        {int(item["n_regs"]) for item in case["resources"].values()}
                    )
                    return "/".join(map(str, values)) if values else "n/a"

                lines.append(
                    f"| `{workload['name']}` | `{arch}` | "
                    f"{legacy_unpack['int4x2_unpack_occurrences']}→"
                    f"{planner['int4x2_unpack_occurrences']} | "
                    f"{legacy_unpack['instruction_count']}→"
                    f"{planner['instruction_count']} | "
                    f"{register_count(legacy_unpack)}→{register_count(planner)} |"
                )
        lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def archive_raw() -> None:
    RAW_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(RAW_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(RAW_DIR).as_posix())


def main() -> int:
    started = time.perf_counter()
    repo_root = Path.cwd().resolve()
    RAW_DIR.mkdir(parents=True, exist_ok=False)
    workloads = build_workloads(repo_root)
    if WORKLOAD_NAMES:
        by_name = {workload["name"]: workload for workload in workloads}
        missing = [name for name in WORKLOAD_NAMES if name not in by_name]
        if missing:
            raise ValueError(f"unknown TILELANG_REAL_VECTOR_WORKLOADS: {', '.join(missing)}")
        workloads = [by_name[name] for name in WORKLOAD_NAMES]
    workload_records, compile_inputs = lower_sources(workloads)
    compile_all(compile_inputs)
    aggregate = enrich_comparisons(workload_records)
    enrich_launch_bounds_scan(workload_records, aggregate)
    payload = {
        "schema": "tilelang-real-vectorization-workloads-v1",
        "repository": REPOSITORY,
        "source_sha": SOURCE_SHA,
        "python": platform.python_version(),
        "architectures": list(ARCHES),
        "modes": list(MODES),
        "selected_workloads": [workload["name"] for workload in workloads],
        "config_key": CONFIG_KEY,
        "device_compile": True,
        "gpu_execution": False,
        "compile_workers": MAX_WORKERS,
        "aggregate": aggregate,
        "workloads": workload_records,
        "duration_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    archive_raw()
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    print(f"result={RESULT_PATH}")
    print(f"report={REPORT_PATH}")
    print(f"raw_archive={RAW_ARCHIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
