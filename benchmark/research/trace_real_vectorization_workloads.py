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
MODES = tuple(
    mode.strip()
    for mode in os.environ.get(
        "TILELANG_REAL_VECTOR_MODES",
        "planner,legacy",
    ).split(",")
    if mode.strip()
)
VALID_MODES = {"planner", "legacy", "planner_inplace"}
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
if (
    not MODES
    or len(set(MODES)) != len(MODES)
    or not set(MODES) <= VALID_MODES
    or not {"planner", "legacy"} <= set(MODES)
):
    raise ValueError(
        "TILELANG_REAL_VECTOR_MODES must contain planner and legacy, with optional planner_inplace"
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
                case_dir = RAW_DIR / workload["name"] / arch / mode
                case_dir.mkdir(parents=True, exist_ok=True)
                source_path = case_dir / "kernel.cu"
                source_path.write_text(source, encoding="utf-8")
                case = {
                    "arch": arch,
                    "mode": mode,
                    "lower_seconds": time.perf_counter() - started,
                    "source_sha256": sha256_text(source),
                    "source_bytes": len(source.encode()),
                    "packed_types": packed_types(source),
                    "packed_type_occurrences": len(PACKED_TYPE_RE.findall(source)),
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
    if source_changed == 0 or sass_changed == 0 or planner_packed_gain == 0:
        raise RuntimeError(
            "planner/legacy trace produced no material source, SASS, or packed-type difference"
        )
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
    }


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
    workload_records, compile_inputs = lower_sources(workloads)
    compile_all(compile_inputs)
    aggregate = enrich_comparisons(workload_records)
    payload = {
        "schema": "tilelang-real-vectorization-workloads-v1",
        "repository": REPOSITORY,
        "source_sha": SOURCE_SHA,
        "python": platform.python_version(),
        "architectures": list(ARCHES),
        "modes": list(MODES),
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
