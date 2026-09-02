"""Compile single-warp shared-memory synchronization A/Bs without a GPU.

Each frozen PrimFunc is lowered twice from one installed build.  The default
path narrows a full-CTA barrier to a warp barrier when the CUDA CTA contains
exactly one warp; the rollback path keeps the historical CTA barrier.  CUDA
source, CUBIN, SASS, and ptxas resources are retained for every target.
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
CONFIG_KEY = "tl.disable_single_warp_storage_sync"
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")
ARCHES = tuple(
    value.strip()
    for value in os.environ.get(
        "TILELANG_SINGLE_WARP_SYNC_ARCHES",
        "sm_75,sm_80,sm_90a,sm_100a,sm_120a",
    ).split(",")
    if value.strip()
)
MAX_WORKERS = int(os.environ.get("TILELANG_SINGLE_WARP_SYNC_WORKERS", "4"))
RESULT_PATH = Path(
    os.environ.get("TILELANG_SINGLE_WARP_SYNC_RESULT", "tilelang-single-warp-sync.json")
)
REPORT_PATH = Path(
    os.environ.get("TILELANG_SINGLE_WARP_SYNC_REPORT", "tilelang-single-warp-sync.md")
)
RAW_DIR = Path(
    os.environ.get("TILELANG_SINGLE_WARP_SYNC_RAW_DIR", "tilelang-single-warp-sync-raw")
)
RAW_ARCHIVE_PATH = Path(
    os.environ.get(
        "TILELANG_SINGLE_WARP_SYNC_RAW_ARCHIVE",
        "tilelang-single-warp-sync-raw.zip",
    )
)
MODES = ("default", "rollback")
CASES = (
    {"name": "shared_exchange_f16", "dtype": "float16"},
    {"name": "shared_exchange_f32", "dtype": "float32"},
    {"name": "shared_exchange_i32", "dtype": "int32"},
)
SASS_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)",
    re.MULTILINE,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def make_kernel(dtype: str):
    @T.prim_func
    def kernel(A: T.Tensor((32,), dtype), B: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((32,), dtype)
            tx = T.get_thread_binding()
            shared[tx] = A[tx]
            B[tx] = shared[31 - tx]

    return kernel


def count_source_syncs(source: str) -> dict[str, int]:
    body = source[source.rfind("__global__") :] if "__global__" in source else source
    return {
        "warp": body.count("__syncwarp("),
        "cta": body.count("__syncthreads("),
        "named": body.count("tl::__bar_sync("),
    }


def parse_sass(sass: str) -> dict[str, Any]:
    opcodes = Counter(match.group("opcode").upper() for match in SASS_INSTRUCTION_RE.finditer(sass))
    if not opcodes:
        raise RuntimeError("nvdisasm output contains no recognized instructions")

    warp_barriers = sum(
        count for opcode, count in opcodes.items() if opcode.startswith("BAR.WARP")
    )
    cta_barriers = sum(
        count
        for opcode, count in opcodes.items()
        if opcode.startswith("BAR") and not opcode.startswith("BAR.WARP")
    )
    return {
        "sass_sha256": sha256_text(sass),
        "sass_chars": len(sass),
        "instruction_count": sum(opcodes.values()),
        "groups": {
            "warp_barrier": warp_barriers,
            "cta_barrier": cta_barriers,
            "shared_load": sum(
                count for opcode, count in opcodes.items() if opcode.startswith(("LDS", "LDSM"))
            ),
            "shared_store": sum(
                count for opcode, count in opcodes.items() if opcode.startswith("STS")
            ),
            "local_load": sum(
                count for opcode, count in opcodes.items() if opcode.startswith("LDL")
            ),
            "local_store": sum(
                count for opcode, count in opcodes.items() if opcode.startswith("STL")
            ),
        },
        "opcodes": dict(opcodes.most_common()),
    }


def lower_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    compile_inputs: list[dict[str, Any]] = []
    for spec in CASES:
        prim_func = make_kernel(spec["dtype"])
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
                source_syncs = count_source_syncs(source)
                expected = {"default": (1, 0), "rollback": (0, 1)}[mode]
                if (source_syncs["warp"], source_syncs["cta"]) != expected:
                    raise RuntimeError(
                        f"unexpected source syncs for {spec['name']}/{arch}/{mode}: {source_syncs}"
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
                    "source_sha256": sha256_text(source),
                    "source_bytes": len(source.encode()),
                    "source_path": source_path.relative_to(RAW_DIR).as_posix(),
                    "source_syncs": source_syncs,
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
        raise RuntimeError("nvdisasm is required for the single-warp-sync trace")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(compile_case, item, nvdisasm): item for item in inputs}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            compiled = future.result()
            item["record"].update(compiled)
            print(
                f"compiled {item['record']['name']}/{item['record']['arch']}/{item['record']['mode']}: "
                f"{compiled['instruction_count']} instructions, "
                f"warp={compiled['groups']['warp_barrier']}, "
                f"cta={compiled['groups']['cta_barrier']}"
            )


def build_comparisons(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(item["name"], item["arch"], item["mode"]): item for item in records}
    comparisons = []
    machine_code_narrowings = 0
    machine_code_eliminations = 0
    explicit_warp_barriers = 0
    spill_free = 0
    for spec in CASES:
        for arch in ARCHES:
            default = by_key[(spec["name"], arch, "default")]
            rollback = by_key[(spec["name"], arch, "rollback")]
            default_resources = list(default["resources"].values())
            rollback_resources = list(rollback["resources"].values())
            spills = sum(
                int(resource["spill_store_bytes"]) + int(resource["spill_load_bytes"])
                for resource in default_resources + rollback_resources
            )
            machine_narrowing = (
                default["groups"]["cta_barrier"] + 1 == rollback["groups"]["cta_barrier"]
                and default["groups"]["warp_barrier"]
                in (
                    rollback["groups"]["warp_barrier"],
                    rollback["groups"]["warp_barrier"] + 1,
                )
            )
            machine_elimination = (
                default["groups"]["warp_barrier"] == rollback["groups"]["warp_barrier"]
            )
            explicit_warp_barrier = (
                default["groups"]["warp_barrier"] == rollback["groups"]["warp_barrier"] + 1
            )
            machine_code_narrowings += int(machine_narrowing)
            machine_code_eliminations += int(machine_elimination)
            explicit_warp_barriers += int(explicit_warp_barrier)
            spill_free += int(spills == 0)
            comparisons.append(
                {
                    "name": spec["name"],
                    "dtype": spec["dtype"],
                    "arch": arch,
                    "machine_code_narrowing": machine_narrowing,
                    "machine_code_elimination": machine_elimination,
                    "explicit_warp_barrier": explicit_warp_barrier,
                    "spill_bytes": spills,
                    "default": default,
                    "rollback": rollback,
                    "default_minus_rollback": {
                        "instructions": default["instruction_count"] - rollback["instruction_count"],
                        "cubin_bytes": default["cubin_bytes"] - rollback["cubin_bytes"],
                        "warp_barrier": default["groups"]["warp_barrier"]
                        - rollback["groups"]["warp_barrier"],
                        "cta_barrier": default["groups"]["cta_barrier"]
                        - rollback["groups"]["cta_barrier"],
                    },
                }
            )

    total = len(CASES) * len(ARCHES)
    summary = {
        "comparisons": total,
        "machine_code_barrier_narrowings": machine_code_narrowings,
        "machine_code_barrier_eliminations": machine_code_eliminations,
        "explicit_warp_barriers": explicit_warp_barriers,
        "spill_free_comparisons": spill_free,
    }
    if machine_code_narrowings != total:
        raise RuntimeError(
            f"expected {total} machine-code barrier narrowings, got {machine_code_narrowings}"
        )
    if spill_free != total:
        raise RuntimeError(f"expected {total} spill-free comparisons, got {spill_free}")
    return comparisons, summary


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Single-warp shared-memory synchronization CUBIN gate",
        "",
        f"- Source: `{payload['repository']}@{payload['source_sha']}`",
        f"- Targets: `{', '.join(payload['arches'])}`",
        f"- Comparisons: {payload['summary']['comparisons']}",
        f"- Machine-code CTA barrier narrowings: {payload['summary']['machine_code_barrier_narrowings']}",
        f"- Machine-code barrier eliminations: {payload['summary']['machine_code_barrier_eliminations']}",
        f"- Explicit warp barriers retained by ptxas: {payload['summary']['explicit_warp_barriers']}",
        f"- Spill-free comparisons: {payload['summary']['spill_free_comparisons']}",
        "",
        "| Case | Arch | SASS barrier rollback to default | Instructions rollback to default | Spill bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison in payload["comparisons"]:
        default = comparison["default"]
        rollback = comparison["rollback"]
        lines.append(
            f"| `{comparison['name']}` | `{comparison['arch']}` | "
            f"CTA {rollback['groups']['cta_barrier']} to {default['groups']['cta_barrier']}; "
            f"warp {rollback['groups']['warp_barrier']} to {default['groups']['warp_barrier']} | "
            f"{rollback['instruction_count']} to {default['instruction_count']} | "
            f"{comparison['spill_bytes']} |"
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_raw() -> None:
    RAW_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RAW_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(RAW_DIR).as_posix())


def main() -> None:
    if not SOURCE_SHA:
        raise RuntimeError("TILELANG_SOURCE_SHA is required")
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    records, compile_inputs = lower_cases()
    compile_all(compile_inputs)
    comparisons, summary = build_comparisons(records)
    payload = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "source_sha": SOURCE_SHA,
        "config_key": CONFIG_KEY,
        "arches": list(ARCHES),
        "modes": list(MODES),
        "cases": list(CASES),
        "summary": summary,
        "comparisons": comparisons,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    archive_raw()
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
