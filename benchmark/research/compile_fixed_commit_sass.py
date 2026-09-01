"""Compile frozen fixed-commit CUDA sources to CUBIN/SASS without a GPU.

This diagnostic consumes the published free-T4 reduction A/B result, extracts
the exact generated CUDA source from both isolated TileLang versions, and
compiles each side against its matching verified wheel headers.  It is intended
for a GitHub-hosted CPU runner with a cached CUDA toolchain.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from typing import Any
import zipfile


REPOSITORY = "nya-a-cat/tilelang"
BASELINE_COMMIT = "958d6d3bd24a31874a2bb189a9791347e855eecd"
BASELINE_BUILD_COMMIT = "a66f3860fdc3c1e5f6f78d96ee02d2e89953eda2"
CANDIDATE_COMMIT = "2ac416ac8e2c60be29bba792d6bf6ded8315467e"
CANDIDATE_HEADER_COMMIT = "4df968c3a85e723dc1870c62a0745284660bffd3"
RUNNER_SOURCE_COMMIT = "2505e78fad5a6631f97c3c86631e9424aa63b15a"
INPUT_RELEASE_TAG = f"colab-fixed-commit-ab-reduction-{RUNNER_SOURCE_COMMIT}"
INPUT_RESULT_NAME = "tilelang-fixed-commit-ab-reduction-t4.json"
INPUT_RESULT_SHA256 = "884d593c11d9111ed19023ea3e7c81a3152ede774ac43bf5f3c67bcce35de5c3"
BASELINE_WHEEL_NAME = "tilelang-0.1.13+cu130.gita66f3860-cp39-abi3-linux_x86_64.whl"
BASELINE_WHEEL_SHA256 = "058138c5b6ece9c0c7b6b20ebbbc572f510e210a6a65f5ef974e118c04f65cc8"
CANDIDATE_WHEEL_NAME = "tilelang-0.1.13+cu130.git4df968c3-cp39-abi3-linux_x86_64.whl"
CANDIDATE_WHEEL_SHA256 = "f3ddcaa79cb10a5b61f4c8ed2eb6941fb30a3f53ddb45aeff2008b6824835f82"
ARCH = "sm_75"
CASE_COUNT = 12
CASE_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
SASS_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)",
    re.MULTILINE,
)
SASS_METADATA_PATTERNS = {
    "registers": re.compile(r"SHI_REGISTERS=(\d+)"),
    "barriers": re.compile(r"SHI_BARRIERS=(\d+)"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return (completed.stdout + completed.stderr).strip()


def verify_file(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")


def safe_extract_wheel(wheel: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"wheel path escapes extraction root: {member.filename!r}")
        archive.extractall(destination)


def locate_include_roots(extracted: Path) -> dict[str, Path]:
    reduce_header = extracted / "tilelang/src/tl_templates/cuda/reduce.h"
    cutlass_header = extracted / "tilelang/3rdparty/cutlass/include/cutlass/cutlass.h"
    for path in (reduce_header, cutlass_header):
        if not path.is_file():
            raise RuntimeError(f"verified wheel is missing required header: {path.relative_to(extracted)}")
    template_root = reduce_header.parents[2]
    cutlass_include = cutlass_header.parents[1]
    if not (template_root / "tl_templates/cuda/common.h").is_file():
        raise RuntimeError(f"invalid template include root: {template_root}")
    return {
        "template_root": template_root,
        "cutlass_include": cutlass_include,
        "reduce_header": reduce_header,
    }


def parse_sass(sass: str) -> dict[str, Any]:
    opcodes = Counter(match.group("opcode").upper() for match in SASS_INSTRUCTION_RE.finditer(sass))
    if not opcodes:
        raise RuntimeError("nvdisasm output contains no recognized instructions")

    def count_prefixes(*prefixes: str) -> int:
        return sum(count for opcode, count in opcodes.items() if opcode.startswith(prefixes))

    metadata: dict[str, int | None] = {}
    for name, pattern in SASS_METADATA_PATTERNS.items():
        match = pattern.search(sass)
        metadata[name] = int(match.group(1)) if match is not None else None
    if metadata["registers"] is None:
        raise RuntimeError("nvdisasm output does not contain SHI_REGISTERS metadata")
    return {
        "sass_sha256": sha256_text(sass),
        "sass_chars": len(sass),
        "instruction_count": sum(opcodes.values()),
        "metadata": metadata,
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


def run_logged(command: list[str], log_path: Path, timeout: int = 180) -> None:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    output = completed.stdout + completed.stderr
    log_path.write_text(
        f"command={json.dumps(command)}\nreturncode={completed.returncode}\nduration_seconds={time.perf_counter() - started}\n{output}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}; see {log_path}")


def compile_source(
    *,
    label: str,
    case_name: str,
    source: str,
    include_roots: dict[str, Path],
    cuda_home: Path,
    nvcc: Path,
    nvdisasm: Path,
    raw_dir: Path,
) -> dict[str, Any]:
    case_dir = raw_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)
    source_path = case_dir / f"{case_name}.cu"
    cubin_path = case_dir / f"{case_name}.cubin"
    sass_path = case_dir / f"{case_name}.sass"
    nvcc_log_path = case_dir / f"{case_name}.nvcc.log"
    source_path.write_text(source, encoding="utf-8")
    command = [
        str(nvcc),
        "-ccbin=g++",
        "--cubin",
        "-O3",
        "-lineinfo",
        f"-arch={ARCH}",
        "-std=c++20",
        f"-I{include_roots['template_root']}",
        f"-I{include_roots['cutlass_include']}",
        f"-I{cuda_home / 'include'}",
        "-o",
        str(cubin_path),
        str(source_path),
    ]
    started = time.perf_counter()
    run_logged(command, nvcc_log_path)
    compile_seconds = time.perf_counter() - started
    disassembly = subprocess.run(
        [str(nvdisasm), str(cubin_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    sass_path.write_text(disassembly, encoding="utf-8")
    parsed = parse_sass(disassembly)
    return {
        "compile_seconds": compile_seconds,
        "source_sha256": sha256_file(source_path),
        "source_bytes": source_path.stat().st_size,
        "cubin_sha256": sha256_file(cubin_path),
        "cubin_bytes": cubin_path.stat().st_size,
        "nvcc_log_sha256": sha256_file(nvcc_log_path),
        "sass_path": sass_path.relative_to(raw_dir).as_posix(),
        **parsed,
    }


def summarize_case(
    baseline_case: dict[str, Any],
    candidate_case: dict[str, Any],
    compiled: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    group_names = sorted(set(compiled["baseline"]["groups"]) | set(compiled["candidate"]["groups"]))
    baseline_registers = compiled["baseline"]["metadata"]["registers"]
    candidate_registers = compiled["candidate"]["metadata"]["registers"]
    return {
        "name": baseline_case["name"],
        "family": baseline_case["family"],
        "canonical_primfunc_sha256": baseline_case["canonical_primfunc_sha256"],
        "input_sha256": baseline_case["input_sha256"],
        "generated_source_changed": (baseline_case["generated_source_sha256"] != candidate_case["generated_source_sha256"]),
        "sass_changed": compiled["baseline"]["sass_sha256"] != compiled["candidate"]["sass_sha256"],
        "baseline": compiled["baseline"],
        "candidate": compiled["candidate"],
        "candidate_minus_baseline": {
            "instruction_count": (compiled["candidate"]["instruction_count"] - compiled["baseline"]["instruction_count"]),
            "registers": candidate_registers - baseline_registers,
            "groups": {
                group: compiled["candidate"]["groups"].get(group, 0) - compiled["baseline"]["groups"].get(group, 0) for group in group_names
            },
        },
    }


def aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    group_names = sorted({group for case in cases for label in ("baseline", "candidate") for group in case[label]["groups"]})
    family_totals: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: {"baseline": defaultdict(int), "candidate": defaultdict(int)})
    for case in cases:
        for label in ("baseline", "candidate"):
            family_totals[case["family"]][label]["instructions"] += case[label]["instruction_count"]
            for group, count in case[label]["groups"].items():
                family_totals[case["family"]][label][group] += count
    return {
        "complete_cases": len(cases),
        "total_cases": CASE_COUNT,
        "source_changed_cases": sum(case["generated_source_changed"] for case in cases),
        "sass_changed_cases": sum(case["sass_changed"] for case in cases),
        "instruction_count": {label: sum(case[label]["instruction_count"] for case in cases) for label in ("baseline", "candidate")},
        "group_totals": {
            label: {group: sum(case[label]["groups"].get(group, 0) for case in cases) for group in group_names}
            for label in ("baseline", "candidate")
        },
        "family_totals": {
            family: {label: dict(values) for label, values in labels.items()} for family, labels in sorted(family_totals.items())
        },
        "barrier_reduced_cases": [case["name"] for case in cases if case["candidate_minus_baseline"]["groups"]["barrier"] < 0],
        "barrier_increased_cases": [case["name"] for case in cases if case["candidate_minus_baseline"]["groups"]["barrier"] > 0],
    }


def report_markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    baseline_groups = aggregate["group_totals"]["baseline"]
    candidate_groups = aggregate["group_totals"]["candidate"]
    lines = [
        "# TileLang fixed-commit reduction SASS diagnostic",
        "",
        f"- Status: `{payload['status']}`; compiled cases: `{CASE_COUNT}/{CASE_COUNT}` for both versions.",
        f"- Target: `{ARCH}`; compiler: `{payload['toolchain']['nvcc_version'].splitlines()[-1]}`.",
        f"- Baseline compiler/runtime: `{BASELINE_COMMIT}`; verified wheel scaffold: `{BASELINE_BUILD_COMMIT}`.",
        f"- Candidate compiler/runtime: `{CANDIDATE_COMMIT}`; verified header wheel: `{CANDIDATE_HEADER_COMMIT}`.",
        f"- Generated-source changes: `{aggregate['source_changed_cases']}/{CASE_COUNT}`; "
        f"SASS changes: `{aggregate['sass_changed_cases']}/{CASE_COUNT}`.",
        f"- Total static instructions: `{aggregate['instruction_count']['baseline']}` → `{aggregate['instruction_count']['candidate']}`.",
        f"- Total barrier instructions: `{baseline_groups.get('barrier', 0)}` → `{candidate_groups.get('barrier', 0)}`.",
        f"- Total shuffle instructions: `{baseline_groups.get('shuffle', 0)}` → `{candidate_groups.get('shuffle', 0)}`.",
        "",
        "## Per-case static metrics",
        "",
        "| Case | Barriers | Shuffles | Shared load/store | Instructions | Registers | Local load/store |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        baseline = case["baseline"]
        candidate = case["candidate"]
        baseline_groups = baseline["groups"]
        candidate_groups = candidate["groups"]
        lines.append(
            f"| `{case['name']}` | "
            f"{baseline_groups.get('barrier', 0)}→{candidate_groups.get('barrier', 0)} | "
            f"{baseline_groups.get('shuffle', 0)}→{candidate_groups.get('shuffle', 0)} | "
            f"{baseline_groups.get('shared_load', 0)}/{baseline_groups.get('shared_store', 0)}→"
            f"{candidate_groups.get('shared_load', 0)}/{candidate_groups.get('shared_store', 0)} | "
            f"{baseline['instruction_count']}→{candidate['instruction_count']} | "
            f"{baseline['metadata']['registers']}→{candidate['metadata']['registers']} | "
            f"{baseline_groups.get('local_load', 0)}/{baseline_groups.get('local_store', 0)}→"
            f"{candidate_groups.get('local_load', 0)}/{candidate_groups.get('local_store', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"The workflow downloads the published `{INPUT_RESULT_NAME}` with SHA-256 `{INPUT_RESULT_SHA256}` and "
            "the two verified wheels. It extracts each frozen generated CUDA source and compiles it with the same "
            "`JITKernel._get_sass` option shape: CUBIN, `-O3`, line info, C++20, exact SM target, and matching "
            "TileLang/CUTLASS headers. Complete source, CUBIN, nvcc log, and nvdisasm output are retained in the raw "
            "archive.",
            "",
            "## Evidence boundary",
            "",
            payload["evidence_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def validate_input(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    expected = {
        "status": "complete",
        "repository": REPOSITORY,
        "suite": "reduction",
        "suite_case_count": CASE_COUNT,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_build_commit": BASELINE_BUILD_COMMIT,
        "candidate_commit": CANDIDATE_COMMIT,
        "runner_source_sha": RUNNER_SOURCE_COMMIT,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"unexpected input {name}: expected {value!r}, got {payload.get(name)!r}")
    manifest = payload.get("environment", {}).get("candidate_overlay_manifest", {})
    if manifest.get("native_base_sha") != CANDIDATE_HEADER_COMMIT or manifest.get("source_sha") != CANDIDATE_COMMIT:
        raise RuntimeError("input candidate environment does not identify the frozen header/compiler commits")
    baseline_cases = {case["name"]: case for case in payload["baseline_inventory"]["cases"]}
    candidate_cases = {case["name"]: case for case in payload["candidate_inventory"]["cases"]}
    if list(baseline_cases) != list(candidate_cases) or len(baseline_cases) != CASE_COUNT:
        raise RuntimeError("input case identity or count mismatch")
    for name, baseline_case in baseline_cases.items():
        candidate_case = candidate_cases[name]
        for field in ("family", "canonical_primfunc_sha256", "input_sha256", "output_shape", "output_dtype"):
            if baseline_case[field] != candidate_case[field]:
                raise RuntimeError(f"case {name} differs in {field}")
        for case in (baseline_case, candidate_case):
            if sha256_text(case["generated_source"]) != case["generated_source_sha256"]:
                raise RuntimeError(f"case {name} generated source hash mismatch")
    return baseline_cases, candidate_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-result", type=Path, required=True)
    parser.add_argument("--baseline-wheel", type=Path, required=True)
    parser.add_argument("--candidate-wheel", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--nvdisasm", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "tilelang-fixed-commit-sass-sm75-actions.json"
    report_path = args.output_dir / "tilelang-fixed-commit-sass-sm75-actions-report.md"
    payload: dict[str, Any] = {
        "schema": "tilelang-fixed-commit-offline-sass-v1",
        "status": "failed",
        "repository": REPOSITORY,
        "arch": ARCH,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_build_commit": BASELINE_BUILD_COMMIT,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_header_commit": CANDIDATE_HEADER_COMMIT,
        "runner_commit": os.environ.get("GITHUB_SHA", "unknown"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "source_evidence": {
            "release_tag": INPUT_RELEASE_TAG,
            "asset": INPUT_RESULT_NAME,
            "sha256": INPUT_RESULT_SHA256,
        },
        "evidence_boundary": (
            "This is a GPU-free static CUBIN/SASS diagnostic for sm_75 using the cached GitHub Actions CUDA 13.0 "
            "toolchain. It explains instruction-level differences in the already measured T4 kernels. Runtime "
            "latency remains governed by the published 30-cycle free-T4 result. Exact CUDA 12.8 T4 disassembly and "
            "A100, H100, B200, RTX 5090, and MI300X gates remain open."
        ),
    }
    try:
        for path in (args.input_result, args.baseline_wheel, args.candidate_wheel, args.nvcc, args.nvdisasm):
            if not path.is_file():
                raise RuntimeError(f"required input is missing: {path}")
        if not (args.cuda_home / "include/cuda.h").is_file():
            raise RuntimeError(f"CUDA include tree is missing under {args.cuda_home}")
        verify_file(args.input_result, INPUT_RESULT_SHA256)
        verify_file(args.baseline_wheel, BASELINE_WHEEL_SHA256)
        verify_file(args.candidate_wheel, CANDIDATE_WHEEL_SHA256)
        input_payload = json.loads(args.input_result.read_text(encoding="utf-8"))
        baseline_cases, candidate_cases = validate_input(input_payload)

        if args.work_dir.exists():
            raise RuntimeError(f"work directory already exists: {args.work_dir}")
        args.work_dir.mkdir(parents=True)
        baseline_extract = args.work_dir / "baseline-wheel"
        candidate_extract = args.work_dir / "candidate-wheel"
        raw_dir = args.work_dir / "raw"
        safe_extract_wheel(args.baseline_wheel, baseline_extract)
        safe_extract_wheel(args.candidate_wheel, candidate_extract)
        include_roots = {
            "baseline": locate_include_roots(baseline_extract),
            "candidate": locate_include_roots(candidate_extract),
        }
        payload["wheel_headers"] = {
            label: {
                "wheel": str(wheel.name),
                "wheel_sha256": sha256_file(wheel),
                "reduce_header_sha256": sha256_file(roots["reduce_header"]),
                "cutlass_header_sha256": sha256_file(roots["cutlass_include"] / "cutlass/cutlass.h"),
            }
            for label, wheel, roots in (
                ("baseline", args.baseline_wheel, include_roots["baseline"]),
                ("candidate", args.candidate_wheel, include_roots["candidate"]),
            )
        }
        payload["toolchain"] = {
            "platform": platform.platform(),
            "python": sys.version,
            "nvcc_path": str(args.nvcc),
            "nvcc_sha256": sha256_file(args.nvcc),
            "nvcc_version": command_output([str(args.nvcc), "--version"]),
            "nvdisasm_path": str(args.nvdisasm),
            "nvdisasm_sha256": sha256_file(args.nvdisasm),
            "nvdisasm_version": command_output([str(args.nvdisasm), "--version"]),
            "gxx_version": command_output(["g++", "--version"]),
        }

        cases: list[dict[str, Any]] = []
        for name, baseline_case in baseline_cases.items():
            if CASE_NAME_RE.fullmatch(name) is None:
                raise RuntimeError(f"unsafe case name: {name!r}")
            candidate_case = candidate_cases[name]
            compiled = {
                label: compile_source(
                    label=label,
                    case_name=name,
                    source=case["generated_source"],
                    include_roots=include_roots[label],
                    cuda_home=args.cuda_home,
                    nvcc=args.nvcc,
                    nvdisasm=args.nvdisasm,
                    raw_dir=raw_dir,
                )
                for label, case in (("baseline", baseline_case), ("candidate", candidate_case))
            }
            cases.append(summarize_case(baseline_case, candidate_case, compiled))
        aggregate = aggregate_cases(cases)
        if aggregate["complete_cases"] != CASE_COUNT or aggregate["source_changed_cases"] != CASE_COUNT:
            raise RuntimeError(f"case completion or source-change invariant failed: {aggregate}")
        payload.update(
            {
                "status": "complete",
                "cases": cases,
                "aggregate": aggregate,
                "raw_dir": str(raw_dir),
            }
        )
    except Exception as error:  # noqa: BLE001 - always preserve a forensic result
        payload.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        payload["duration_seconds"] = time.time() - started
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if payload["status"] == "complete":
            report_path.write_text(report_markdown(payload), encoding="utf-8")
    print(f"TILELANG_FIXED_COMMIT_SASS_RESULT={result_path}")
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
