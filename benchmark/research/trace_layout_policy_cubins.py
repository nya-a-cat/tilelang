"""Compile frozen layout-policy choices to CUBIN/SASS without a GPU.

The layout cost trace proves that several policies choose different layouts,
but its modeled scores cannot establish which choice produces leaner machine
code.  This companion diagnostic lowers the same byte-identity-guarded 18
PrimFuncs under three policies for explicit CUDA architectures.  Identical
CUDA sources are compiled once per architecture and the result is shared by
hash, keeping the free-CI matrix small while preserving exact comparisons.

This script performs CUDA device compilation and disassembly.  It never
initializes or executes a GPU, so latency and correctness remain outside its
evidence boundary.
"""

from __future__ import annotations

import concurrent.futures
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
from typing import Any
import zipfile

import tilelang as tl
from tilelang import tvm
from tilelang.contrib.cuda_resource_info import pop_recorded, reset_recorder
from tilelang.cuda.backend import tilelang_callback_cuda_compile

from trace_layout_cost_features import (
    EXPECTED_PRIMFUNC_SHA256,
    build_cases,
    sha256_text,
)


REPOSITORY = "nya-a-cat/tilelang"
CONFIG_KEY = "tl.layout_cost_model"
POLICIES = tuple(
    value.strip()
    for value in os.environ.get(
        "TILELANG_LAYOUT_POLICY_CUBIN_POLICIES",
        "target-default,register-count,io-aware",
    ).split(",")
    if value.strip()
)
ARCHES = tuple(
    value.strip()
    for value in os.environ.get(
        "TILELANG_LAYOUT_POLICY_CUBIN_ARCHES",
        "sm_75,sm_80,sm_90a,sm_100a,sm_120a",
    ).split(",")
    if value.strip()
)
MAX_WORKERS = int(os.environ.get("TILELANG_LAYOUT_POLICY_CUBIN_WORKERS", "4"))
RESULT_PATH = Path(
    os.environ.get(
        "TILELANG_LAYOUT_POLICY_CUBIN_RESULT",
        "tilelang-layout-policy-cubins.json",
    )
)
REPORT_PATH = Path(
    os.environ.get(
        "TILELANG_LAYOUT_POLICY_CUBIN_REPORT",
        "tilelang-layout-policy-cubins.md",
    )
)
RAW_DIR = Path(
    os.environ.get(
        "TILELANG_LAYOUT_POLICY_CUBIN_RAW_DIR",
        "tilelang-layout-policy-cubins-raw",
    )
)
RAW_ARCHIVE_PATH = Path(
    os.environ.get(
        "TILELANG_LAYOUT_POLICY_CUBIN_RAW_ARCHIVE",
        "tilelang-layout-policy-cubins-raw.zip",
    )
)
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")
SASS_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)",
    re.MULTILINE,
)

if not POLICIES or POLICIES[0] != "target-default" or len(set(POLICIES)) != len(POLICIES):
    raise ValueError("policies must be distinct and start with target-default")
if not ARCHES or len(set(ARCHES)) != len(ARCHES):
    raise ValueError("architectures must be distinct")
if MAX_WORKERS < 1:
    raise ValueError("compile worker count must be positive")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_nvdisasm() -> str:
    candidate = os.environ.get("NVDISASM") or shutil.which("nvdisasm")
    if candidate and Path(candidate).is_file():
        return candidate
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidate = str(Path(cuda_home) / "bin/nvdisasm")
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError("nvdisasm is required for the layout-policy CUBIN trace")


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
            "global_load": count_prefixes("LDG"),
            "global_store": count_prefixes("STG"),
            "local_load": count_prefixes("LDL"),
            "local_store": count_prefixes("STL"),
            "shared_load": count_prefixes("LDS", "LDSM"),
            "shared_store": count_prefixes("STS"),
            "shuffle": count_prefixes("SHFL"),
        },
        "opcodes": dict(opcodes.most_common()),
    }


def maximum_resource(resources: dict[str, dict[str, Any]], field: str) -> int:
    return max((int(record.get(field) or 0) for record in resources.values()), default=0)


def lower_matrix() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    cases = build_cases()
    lowered: list[dict[str, Any]] = []
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for case in cases:
        actual_hash = sha256_text(str(case.program))
        expected_hash = EXPECTED_PRIMFUNC_SHA256[case.name]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"canonical PrimFunc hash drift for {case.name}: expected {expected_hash}, got {actual_hash}"
            )
        for arch in ARCHES:
            target = tvm.target.Target({"kind": "cuda", "arch": arch})
            for policy in POLICIES:
                pass_configs = {CONFIG_KEY: policy}
                started = time.perf_counter()
                with tvm.transform.PassContext(opt_level=3, config=pass_configs), target:
                    artifact = tl.lower(
                        case.program,
                        target=target,
                        enable_device_compile=False,
                    )
                source = str(artifact.kernel_source or "")
                if not source.strip():
                    raise RuntimeError(f"empty source for {case.name}/{arch}/{policy}")
                source_hash = sha256_text(source)
                source_path = RAW_DIR / case.name / arch / policy / "kernel.cu"
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(source, encoding="utf-8")
                record = {
                    "name": case.name,
                    "family": case.family,
                    "primfunc_sha256": actual_hash,
                    "arch": arch,
                    "policy": policy,
                    "lower_seconds": time.perf_counter() - started,
                    "source_sha256": source_hash,
                    "source_bytes": len(source.encode()),
                    "source_path": source_path.relative_to(RAW_DIR).as_posix(),
                }
                lowered.append(record)
                key = (arch, source_hash)
                if key not in unique:
                    unique[key] = {
                        "arch": arch,
                        "source": source,
                        "source_sha256": source_hash,
                        "pass_configs": pass_configs,
                        "compile_dir": RAW_DIR / "_unique" / arch / source_hash,
                    }
    return lowered, unique


def compile_unique(item: dict[str, Any], nvdisasm: str) -> dict[str, Any]:
    target = tvm.target.Target({"kind": "cuda", "arch": item["arch"]})
    reset_recorder()
    started = time.perf_counter()
    cubin = bytes(tilelang_callback_cuda_compile(item["source"], target, item["pass_configs"]))
    compile_seconds = time.perf_counter() - started
    resources = {name: asdict(usage) for name, usage in sorted(pop_recorded().items())}
    compile_dir = item["compile_dir"]
    compile_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = compile_dir / "kernel.cubin"
    sass_path = compile_dir / "kernel.sass"
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


def compile_unique_matrix(unique: dict[tuple[str, str], dict[str, Any]]) -> None:
    nvdisasm = find_nvdisasm()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_key = {
            executor.submit(compile_unique, item, nvdisasm): key for key, item in unique.items()
        }
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            compiled = future.result()
            unique[key].update(compiled)
            print(
                f"compiled {key[0]}/{key[1][:12]}: "
                f"{compiled['instruction_count']} instructions, {compiled['cubin_bytes']} bytes",
                flush=True,
            )


def attach_compiled(
    lowered: list[dict[str, Any]], unique: dict[tuple[str, str], dict[str, Any]]
) -> None:
    shared_fields = (
        "compile_seconds",
        "cubin_sha256",
        "cubin_bytes",
        "cubin_path",
        "sass_path",
        "sass_sha256",
        "sass_chars",
        "instruction_count",
        "groups",
        "opcodes",
        "resources",
    )
    for record in lowered:
        compiled = unique[(record["arch"], record["source_sha256"])]
        record.update({field: compiled[field] for field in shared_fields})


def make_comparisons(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(record["name"], record["arch"], record["policy"]): record for record in records}
    comparisons: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, int]] = {
        policy: {
            "comparisons": 0,
            "source_changed": 0,
            "sass_changed": 0,
            "strict_instruction_reductions": 0,
            "equal_instructions": 0,
            "instruction_regressions": 0,
            "register_reductions": 0,
            "register_regressions": 0,
            "spill_cases": 0,
        }
        for policy in POLICIES[1:]
    }
    family_aggregates: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(
            lambda: {"comparisons": 0, "reductions": 0, "ties": 0, "regressions": 0}
        )
    )

    for record in records:
        if record["policy"] != "target-default":
            continue
        default = record
        default_regs = maximum_resource(default["resources"], "n_regs")
        default_spills = maximum_resource(default["resources"], "n_spills")
        for policy in POLICIES[1:]:
            candidate = by_key[(record["name"], record["arch"], policy)]
            candidate_regs = maximum_resource(candidate["resources"], "n_regs")
            candidate_spills = maximum_resource(candidate["resources"], "n_spills")
            instruction_delta = candidate["instruction_count"] - default["instruction_count"]
            group_delta = {
                group: candidate["groups"].get(group, 0) - default["groups"].get(group, 0)
                for group in sorted(set(candidate["groups"]) | set(default["groups"]))
            }
            comparison = {
                "name": record["name"],
                "family": record["family"],
                "arch": record["arch"],
                "candidate_policy": policy,
                "source_changed": candidate["source_sha256"] != default["source_sha256"],
                "sass_changed": candidate["sass_sha256"] != default["sass_sha256"],
                "candidate_minus_default": {
                    "instructions": instruction_delta,
                    "registers": candidate_regs - default_regs,
                    "spills": candidate_spills - default_spills,
                    "cubin_bytes": candidate["cubin_bytes"] - default["cubin_bytes"],
                    "groups": group_delta,
                },
                "default": {
                    "instructions": default["instruction_count"],
                    "registers": default_regs,
                    "spills": default_spills,
                    "source_sha256": default["source_sha256"],
                    "sass_sha256": default["sass_sha256"],
                },
                "candidate": {
                    "instructions": candidate["instruction_count"],
                    "registers": candidate_regs,
                    "spills": candidate_spills,
                    "source_sha256": candidate["source_sha256"],
                    "sass_sha256": candidate["sass_sha256"],
                },
            }
            comparisons.append(comparison)
            aggregate = aggregates[policy]
            aggregate["comparisons"] += 1
            aggregate["source_changed"] += int(comparison["source_changed"])
            aggregate["sass_changed"] += int(comparison["sass_changed"])
            aggregate["strict_instruction_reductions"] += int(instruction_delta < 0)
            aggregate["equal_instructions"] += int(instruction_delta == 0)
            aggregate["instruction_regressions"] += int(instruction_delta > 0)
            aggregate["register_reductions"] += int(candidate_regs < default_regs)
            aggregate["register_regressions"] += int(candidate_regs > default_regs)
            aggregate["spill_cases"] += int(candidate_spills > 0)
            family = family_aggregates[policy][record["family"]]
            family["comparisons"] += 1
            family["reductions"] += int(instruction_delta < 0)
            family["ties"] += int(instruction_delta == 0)
            family["regressions"] += int(instruction_delta > 0)

    return comparisons, {
        "by_policy": aggregates,
        "by_policy_family": {
            policy: dict(sorted(families.items()))
            for policy, families in sorted(family_aggregates.items())
        },
    }


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Layout policy CUBIN/SASS diagnostic",
        "",
        f"- Source: `{payload['source_sha']}`",
        f"- Frozen PrimFuncs: `{payload['case_count']}`",
        f"- Architectures: `{', '.join(payload['architectures'])}`",
        f"- Logical policy cases: `{payload['logical_case_count']}`",
        f"- Unique CUDA sources compiled: `{payload['unique_compile_count']}`",
        f"- Device execution: `{str(payload['gpu_execution']).lower()}`",
        "",
        "## Policy summary",
        "",
        "| Candidate vs target-default | comparisons | source changed | "
        "instruction reductions | ties | regressions | register reductions | "
        "register regressions | spill cases |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, aggregate in payload["aggregate"]["by_policy"].items():
        lines.append(
            f"| {policy} | {aggregate['comparisons']} | {aggregate['source_changed']} | "
            f"{aggregate['strict_instruction_reductions']} | {aggregate['equal_instructions']} | "
            f"{aggregate['instruction_regressions']} | {aggregate['register_reductions']} | "
            f"{aggregate['register_regressions']} | {aggregate['spill_cases']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            payload["evidence_boundary"],
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def archive_raw() -> None:
    RAW_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RAW_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(RAW_DIR).as_posix())


def main() -> int:
    started = time.time()
    lowered, unique = lower_matrix()
    compile_unique_matrix(unique)
    attach_compiled(lowered, unique)
    comparisons, aggregate = make_comparisons(lowered)
    payload = {
        "schema": "tilelang-layout-policy-cubin-trace-v1",
        "repository": REPOSITORY,
        "source_sha": SOURCE_SHA,
        "python": platform.python_version(),
        "tilelang": tl.__version__,
        "architectures": list(ARCHES),
        "policies": list(POLICIES),
        "case_count": len(EXPECTED_PRIMFUNC_SHA256),
        "logical_case_count": len(lowered),
        "unique_compile_count": len(unique),
        "compile_dedup_saved": len(lowered) - len(unique),
        "gpu_execution": False,
        "device_compile": True,
        "duration_seconds": time.time() - started,
        "evidence_boundary": (
            "Static CUDA source, CUBIN, SASS, and ptxas-resource comparison for byte-identity-guarded "
            "PrimFuncs. It does not execute kernels or establish correctness, latency, throughput, or an "
            "end-to-end speedup. A policy change still requires same-input runtime A/B on real GPUs."
        ),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "cases": lowered,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    archive_raw()
    print(
        json.dumps(
            {
                "logical_cases": len(lowered),
                "unique_compiles": len(unique),
                "dedup_saved": len(lowered) - len(unique),
                "result": str(RESULT_PATH),
                "report": str(REPORT_PATH),
                "raw_archive": str(RAW_ARCHIVE_PATH),
                "raw_archive_sha256": sha256_file(RAW_ARCHIVE_PATH),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
