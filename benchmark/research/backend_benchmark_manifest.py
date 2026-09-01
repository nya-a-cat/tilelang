"""Validate and select the frozen TileLang backend benchmark matrix.

This module has no TileLang, Torch, or third-party dependency. It is safe to
run in the quick CPU-only CI job before any compiler build or GPU allocation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


MANIFEST_PATH = Path(__file__).with_name("backend_benchmark_manifest.json")
EXPECTED_SCHEMA = "tilelang-backend-benchmark-manifest-v1"
EXPECTED_LAYERS = {"operator", "subgraph", "model", "external_suite"}
EXPECTED_REPOSITORY = "nya-a-cat/tilelang"
EXPECTED_BASELINE_COMMIT = "958d6d3bd24a31874a2bb189a9791347e855eecd"
EXPECTED_REQUIRED_HARDWARE = {
    "a100_sm80",
    "h100_sm90a",
    "b200_sm100a",
    "rtx5090_sm120",
    "mi300x_gfx942",
}
EXPECTED_WORKLOAD_GROUP_COUNT = 11
EXPECTED_CASE_COUNT = 112
IMPLEMENTATION_STATUSES = {
    "in_tree",
    "partial_in_tree",
    "adapter_required",
    "integration_required",
}
HEX_COMMIT = re.compile(r"[0-9a-f]{40}")


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_ids(items: list[dict[str, Any]], label: str, errors: list[str]) -> set[str]:
    values = [item.get("id") for item in items]
    missing = [index for index, value in enumerate(values) if not isinstance(value, str) or not value]
    if missing:
        errors.append(f"{label} entries without a non-empty id: {missing}")
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate {label} ids: {duplicates}")
    return {value for value in values if isinstance(value, str) and value}


def validate_manifest(manifest: dict[str, Any], *, manifest_path: Path = MANIFEST_PATH) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != EXPECTED_SCHEMA:
        errors.append(f"schema must be {EXPECTED_SCHEMA!r}")
    if manifest.get("status") != "frozen":
        errors.append("manifest status must be frozen")
    if manifest.get("repository") != EXPECTED_REPOSITORY:
        errors.append(f"repository must remain {EXPECTED_REPOSITORY!r}")

    baseline = manifest.get("baseline", {})
    if baseline.get("commit") != EXPECTED_BASELINE_COMMIT:
        errors.append(f"baseline.commit must remain {EXPECTED_BASELINE_COMMIT}")
    if "byte-identical TileLang programs" not in baseline.get("comparison_contract", ""):
        errors.append("baseline comparison contract must preserve identical TileLang programs")

    tiers = manifest.get("tiers", [])
    hardware = manifest.get("hardware", [])
    sources = manifest.get("sources", [])
    workloads = manifest.get("workloads", [])
    tier_ids = unique_ids(tiers, "tier", errors)
    hardware_ids = unique_ids(hardware, "hardware", errors)
    source_ids = unique_ids(sources, "source", errors)
    unique_ids(workloads, "workload", errors)

    required_hardware = {item.get("id") for item in hardware if item.get("required_for_primary_score")}
    if required_hardware != EXPECTED_REQUIRED_HARDWARE:
        errors.append(f"required primary hardware must remain {sorted(EXPECTED_REQUIRED_HARDWARE)}")
    for target in hardware:
        if target.get("backend") not in {"cuda", "rocm"}:
            errors.append(f"hardware {target.get('id')}: unsupported backend")
        if target.get("role") not in {"free_screen", "main", "expansion"}:
            errors.append(f"hardware {target.get('id')}: unsupported role")

    for source in sources:
        if not str(source.get("url", "")).startswith("https://"):
            errors.append(f"source {source.get('id')}: url must use HTTPS")
        if "repository" in str(source.get("kind", "")) and not HEX_COMMIT.fullmatch(str(source.get("commit", ""))):
            errors.append(f"source {source.get('id')}: repository source needs a full commit")

    seen_case_ids: set[str] = set()
    covered_layers: set[str] = set()
    for workload in workloads:
        workload_id = workload.get("id")
        layer = workload.get("layer")
        covered_layers.add(layer)
        if layer not in EXPECTED_LAYERS:
            errors.append(f"workload {workload_id}: invalid layer {layer!r}")
        if not workload.get("family"):
            errors.append(f"workload {workload_id}: family is required")
        if workload.get("implementation_status") not in IMPLEMENTATION_STATUSES:
            errors.append(f"workload {workload_id}: invalid implementation_status {workload.get('implementation_status')!r}")
        driver = workload.get("driver")
        if driver is not None:
            driver_path = manifest_path.resolve().parents[2] / driver
            if not driver_path.is_file():
                errors.append(f"workload {workload_id}: missing driver {driver!r}")
        if workload.get("implementation_status") == "in_tree" and driver is None:
            errors.append(f"workload {workload_id}: in_tree status requires a driver")

        unknown_tiers = set(workload.get("tiers", [])) - tier_ids
        unknown_hardware = set(workload.get("hardware", [])) - hardware_ids
        unknown_sources = set(workload.get("source_ids", [])) - source_ids
        if unknown_tiers:
            errors.append(f"workload {workload_id}: unknown tiers {sorted(unknown_tiers)}")
        if unknown_hardware:
            errors.append(f"workload {workload_id}: unknown hardware {sorted(unknown_hardware)}")
        if unknown_sources:
            errors.append(f"workload {workload_id}: unknown sources {sorted(unknown_sources)}")
        for required_field in ("tiers", "hardware", "source_ids", "dtypes", "modes", "cases"):
            if not workload.get(required_field):
                errors.append(f"workload {workload_id}: {required_field} must be non-empty")

        local_case_ids = unique_ids(workload.get("cases", []), f"case in {workload_id}", errors)
        overlap = sorted(local_case_ids & seen_case_ids)
        if overlap:
            errors.append(f"workload {workload_id}: globally duplicate case ids {overlap}")
        seen_case_ids.update(local_case_ids)
        for case in workload.get("cases", []):
            params = case.get("params", {})
            if isinstance(params, dict):
                for commit_field in ("revision", "source_commit"):
                    commit = params.get(commit_field)
                    if commit is not None and not HEX_COMMIT.fullmatch(str(commit)):
                        errors.append(f"case {case.get('id')}: {commit_field} must be a full commit")
            case_hardware = set(case.get("hardware", workload.get("hardware", [])))
            if not case_hardware:
                errors.append(f"case {case.get('id')}: effective hardware must be non-empty")
            unknown_case_hardware = case_hardware - set(workload.get("hardware", []))
            if unknown_case_hardware:
                errors.append(f"case {case.get('id')}: hardware is outside its workload {sorted(unknown_case_hardware)}")

    if covered_layers != EXPECTED_LAYERS:
        errors.append(f"primary layer coverage mismatch: {sorted(covered_layers)}")
    if len(workloads) != EXPECTED_WORKLOAD_GROUP_COUNT:
        errors.append(f"frozen workload group count must remain {EXPECTED_WORKLOAD_GROUP_COUNT}")
    if len(seen_case_ids) != EXPECTED_CASE_COUNT:
        errors.append(f"frozen case count must remain {EXPECTED_CASE_COUNT}")

    protocol = manifest.get("protocol", {})
    aggregation = protocol.get("aggregation", {})
    acceptance = protocol.get("acceptance", {})
    if set(aggregation.get("primary_layers", [])) != EXPECTED_LAYERS:
        errors.append("aggregation.primary_layers must cover all four frozen layers")
    if acceptance.get("overall_hierarchical_geomean_min") != 1.5:
        errors.append("overall hierarchical geomean gate must remain exactly 1.5")
    if acceptance.get("correctness_pass_fraction") != 1.0:
        errors.append("correctness gate must remain exactly 100%")
    if acceptance.get("required_architecture_completion_fraction") != 1.0:
        errors.append("required architecture completion gate must remain exactly 100%")
    return errors


def select_jobs(
    manifest: dict[str, Any],
    *,
    tier: str | None,
    hardware: str | None,
    layer: str | None,
    ready_only: bool,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for workload in manifest["workloads"]:
        if tier is not None and tier not in workload["tiers"]:
            continue
        if hardware is not None and hardware not in workload["hardware"]:
            continue
        if layer is not None and layer != workload["layer"]:
            continue
        if ready_only and workload["implementation_status"] != "in_tree":
            continue
        selected_tiers = [tier] if tier is not None else workload["tiers"]
        for case in workload["cases"]:
            case_hardware = case.get("hardware", workload["hardware"])
            if hardware is not None and hardware not in case_hardware:
                continue
            selected_hardware = [hardware] if hardware is not None else case_hardware
            for target in selected_hardware:
                jobs.append(
                    {
                        "job_id": f"{workload['id']}::{case['id']}::{target}",
                        "workload": workload["id"],
                        "case": case["id"],
                        "layer": workload["layer"],
                        "family": workload["family"],
                        "hardware": target,
                        "tiers": selected_tiers,
                        "dtypes": workload["dtypes"],
                        "modes": workload["modes"],
                        "implementation_status": workload["implementation_status"],
                        "driver": workload.get("driver"),
                        "params": case.get("params", {}),
                    }
                )
    return jobs


def summary(manifest: dict[str, Any], jobs: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    return {
        "schema": "tilelang-backend-benchmark-selection-v1",
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "baseline_commit": manifest["baseline"]["commit"],
        "workload_group_count": len(manifest["workloads"]),
        "case_count": sum(len(workload["cases"]) for workload in manifest["workloads"]),
        "selected_job_count": len(jobs),
        "selected_by_layer": dict(sorted(Counter(job["layer"] for job in jobs).items())),
        "selected_by_hardware": dict(sorted(Counter(job["hardware"] for job in jobs).items())),
        "selected_by_status": dict(sorted(Counter(job["implementation_status"] for job in jobs).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    select = subparsers.add_parser("select")
    select.add_argument("--tier")
    select.add_argument("--hardware")
    select.add_argument("--layer", choices=sorted(EXPECTED_LAYERS))
    select.add_argument("--ready-only", action="store_true")
    select.add_argument("--include-jobs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    errors = validate_manifest(manifest, manifest_path=args.manifest)
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2, sort_keys=True))
        return 2

    if args.command == "validate":
        jobs = select_jobs(
            manifest,
            tier=None,
            hardware=None,
            layer=None,
            ready_only=False,
        )
        payload = {"status": "complete", **summary(manifest, jobs, args.manifest)}
    else:
        tier_ids = {tier["id"] for tier in manifest["tiers"]}
        hardware_ids = {target["id"] for target in manifest["hardware"]}
        if args.tier is not None and args.tier not in tier_ids:
            raise SystemExit(f"unknown tier: {args.tier}")
        if args.hardware is not None and args.hardware not in hardware_ids:
            raise SystemExit(f"unknown hardware: {args.hardware}")
        jobs = select_jobs(
            manifest,
            tier=args.tier,
            hardware=args.hardware,
            layer=args.layer,
            ready_only=args.ready_only,
        )
        payload = {
            "status": "complete",
            "filters": {
                "tier": args.tier,
                "hardware": args.hardware,
                "layer": args.layer,
                "ready_only": args.ready_only,
            },
            **summary(manifest, jobs, args.manifest),
        }
        if args.include_jobs:
            payload["jobs"] = jobs
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
