"""Install an exact native overlay and run the regularized-policy A/B on Colab T4.

The Colab CLI transmits this controller as one cell. Upload the four files in
``SCRIPT_SHA256`` to ``WORK_DIR`` before execution. Every executable or Python
input downloaded or uploaded for the run is hash checked before use.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any
import urllib.request


REPOSITORY = "nya-a-cat/tilelang"
OVERLAY_SOURCE_SHA = "2ac416ac8e2c60be29bba792d6bf6ded8315467e"
NATIVE_BASE_SHA = "4df968c3a85e723dc1870c62a0745284660bffd3"
NATIVE_BASE_TAG = f"colab-fast-{NATIVE_BASE_SHA}"
OVERLAY_RELEASE_TAG = f"colab-native-{OVERLAY_SOURCE_SHA}"
BASE_WHEEL_NAME = "tilelang-0.1.13+cu130.git4df968c3-cp39-abi3-linux_x86_64.whl"
BASE_WHEEL_SHA256 = "f3ddcaa79cb10a5b61f4c8ed2eb6941fb30a3f53ddb45aeff2008b6824835f82"
BASE_WHEEL_URL = (
    f"https://github.com/{REPOSITORY}/releases/download/{NATIVE_BASE_TAG}/tilelang-0.1.13%2Bcu130.git4df968c3-cp39-abi3-linux_x86_64.whl"
)
OVERLAY_ASSETS = {
    "apply_colab_native_overlay.sh": "945da1e58f8384ca09a8e5983e0c283bc716ba16372eb8f269de87deb7d96d98",
    "native-overlay-manifest.json": "7cc1bd1fcf3e247d1d0f9193eaadc67d5138fd9de46fb02338725086dc5b28f2",
    "tilelang-native-overlay.tar.gz": "d8cefd2cb79b583c18808ab9fef90b6c08c63435508f6c296686e6feb176680a",
}
SCRIPT_SHA256 = {
    "benchmark_layout_cost_models_t4.py": "5da1e8fb602f84e3d7bb17ad7c357412935eca379e2e21b261e70e07207d1a5f",
    "benchmark_layout_normalization_policies_t4.py": "a424500185fbc76a32901527ee917e9116b743a5ec820c32cfb8d67a8117009a",
    "scan_layout_policy_divergence.py": "dd002155fa32d023d810965f816fde49232127c38975db4762db48ca7e4d3da6",
    "benchmark_layout_divergent_cases_t4.py": "c279f1496a48d1b595ffef8c9bef7531cad1f9e58f8cb0d8329bb1ed923c36f9",
}
POLICIES = ("register-count", "io-aware-regularized")
WORK_DIR = Path("/tmp/tilelang-layout-regularized")
OUTPUT_DIR = Path("/tmp/tilelang-layout-regularized-evidence")
BENCHMARK_RESULT = WORK_DIR / "benchmark-result.json"
RESULT_PATH = OUTPUT_DIR / "layout-regularized-t4.json"
GZIP_PATH = OUTPUT_DIR / "layout-regularized-t4.json.gz"
REPORT_PATH = OUTPUT_DIR / "layout-regularized-t4-report.md"
CHECKSUM_PATH = OUTPUT_DIR / "SHA256SUMS"
HEX_40 = re.compile(r"[0-9a-f]{40}")
HEX_64 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - retain optional provenance failures
        return f"{type(error).__name__}: {error}"


def run_checked(
    command: list[str],
    records: list[dict[str, Any]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> str:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = completed.stdout + completed.stderr
    records.append(
        {
            "command": command,
            "returncode": completed.returncode,
            "duration_seconds": time.perf_counter() - started,
            "output_tail": output[-12000:],
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {command}")
    return output


def download(url: str, output: Path, expected_sha256: str) -> dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "tilelang-research"})
    with urllib.request.urlopen(request, timeout=120) as response, output.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    actual_sha256 = sha256_file(output)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"download hash mismatch for {output.name}: expected {expected_sha256}, got {actual_sha256}")
    return {
        "name": output.name,
        "url": url,
        "sha256": actual_sha256,
        "bytes": output.stat().st_size,
        "duration_seconds": time.perf_counter() - started,
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def verify_uploaded_scripts() -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in SCRIPT_SHA256.items():
        path = WORK_DIR / name
        if not path.is_file():
            raise RuntimeError(f"missing uploaded benchmark input: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"uploaded script hash mismatch for {name}: expected {expected_sha256}, got {actual_sha256}")
        verified[name] = {
            "sha256": actual_sha256,
            "bytes": path.stat().st_size,
        }
    return verified


def prepare_overlay(records: list[dict[str, Any]]) -> dict[str, Any]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    downloads: list[dict[str, Any]] = []
    base_wheel = WORK_DIR / BASE_WHEEL_NAME
    downloads.append(download(BASE_WHEEL_URL, base_wheel, BASE_WHEEL_SHA256))
    for name, expected_sha256 in OVERLAY_ASSETS.items():
        url = f"https://github.com/{REPOSITORY}/releases/download/{OVERLAY_RELEASE_TAG}/{name}"
        downloads.append(download(url, WORK_DIR / name, expected_sha256))

    manifest_path = WORK_DIR / "native-overlay-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema": "tilelang-colab-native-overlay-v1",
        "repository": REPOSITORY,
        "source_sha": OVERLAY_SOURCE_SHA,
        "native_base_sha": NATIVE_BASE_SHA,
        "native_base_tag": NATIVE_BASE_TAG,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"unexpected overlay manifest {key}: {manifest.get(key)!r}")
    if manifest.get("overlay_sha256") != OVERLAY_ASSETS["tilelang-native-overlay.tar.gz"]:
        raise RuntimeError("overlay manifest archive hash differs from the release asset")
    base_manifest = manifest.get("base_wheel", {})
    if base_manifest.get("name") != BASE_WHEEL_NAME or base_manifest.get("sha256") != BASE_WHEEL_SHA256:
        raise RuntimeError("overlay manifest identifies a different base wheel")

    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            "apache-tvm-ffi==0.1.12",
            "torch-c-dlpack-ext==0.1.5",
            "z3-solver==4.15.4.0",
        ],
        records,
        timeout=300,
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            "--no-deps",
            str(base_wheel),
        ],
        records,
        timeout=300,
    )
    installer = WORK_DIR / "apply_colab_native_overlay.sh"
    installer.chmod(0o755)
    install_env = os.environ.copy()
    install_env["PYTHON"] = sys.executable
    run_checked(
        [
            str(installer),
            str(WORK_DIR / "tilelang-native-overlay.tar.gz"),
            str(manifest_path),
        ],
        records,
        env=install_env,
        timeout=300,
    )
    return {"downloads": downloads, "manifest": manifest}


def run_benchmark(records: list[dict[str, Any]]) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(WORK_DIR),
            "TILELANG_LAYOUT_AB_POLICIES": ",".join(POLICIES),
            "TILELANG_LAYOUT_SCAN_POLICIES": ",".join(POLICIES),
            "TILELANG_LAYOUT_AB_CYCLES": "30",
            "TILELANG_LAYOUT_AB_WARM_SECONDS": "1",
            "TILELANG_LAYOUT_AB_MIN_BATCH_MS": "50",
            "TILELANG_LAYOUT_AB_RESULT": str(BENCHMARK_RESULT),
            "TILELANG_LAYOUT_AB_SCHEMA": "tilelang-layout-regularized-runtime-ab-v1",
            "TILELANG_SOURCE_SHA": OVERLAY_SOURCE_SHA,
            "TILELANG_NATIVE_BASE_SHA": NATIVE_BASE_SHA,
            "TILELANG_LAYOUT_AB_EVIDENCE_BOUNDARY": (
                "One free Colab T4 measures 18 unchanged layout-divergent PrimFuncs under register-count and "
                "the provisional io-aware-regularized policy. Correctness, generated source, ptxas resources, "
                "paired eager timing, and CUDA Graph replay are covered. The frozen cross-architecture, "
                "operator, subgraph, model, and external-suite gates remain open."
            ),
        }
    )
    run_checked(
        [sys.executable, str(WORK_DIR / "benchmark_layout_divergent_cases_t4.py")],
        records,
        cwd=WORK_DIR,
        env=env,
        timeout=1200,
    )
    if not BENCHMARK_RESULT.is_file():
        raise RuntimeError("benchmark produced no result JSON")
    payload = json.loads(BENCHMARK_RESULT.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    if payload.get("status") != "complete":
        raise RuntimeError(f"benchmark status is {payload.get('status')!r}")
    if payload.get("policies") != list(POLICIES):
        raise RuntimeError(f"benchmark policies drifted: {payload.get('policies')!r}")
    if aggregate.get("complete_cases") != 18 or aggregate.get("total_cases") != 18:
        raise RuntimeError(f"benchmark case completion drifted: {aggregate!r}")
    graph_cases = [case for case in payload.get("cases", []) if "cuda_graph" in case]
    if len(graph_cases) != 18:
        raise RuntimeError(f"CUDA Graph completion drifted: {len(graph_cases)}/18")
    if payload.get("system", {}).get("gpu_capability") != [7, 5]:
        raise RuntimeError(f"expected SM75 T4, got {payload.get('system')!r}")
    return payload


def report_markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    family_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"eager": [], "cuda_graph": []})
    rows: list[tuple[str, str, float, float]] = []
    for case in payload["cases"]:
        eager = float(case["eager"]["candidate_speedup_over_baseline"])
        graph = float(case["cuda_graph"]["candidate_speedup_over_baseline"])
        family_values[case["family"]]["eager"].append(eager)
        family_values[case["family"]]["cuda_graph"].append(graph)
        rows.append((case["name"], case["family"], eager, graph))

    eager_sorted = sorted(rows, key=lambda row: row[2])
    graph_sorted = sorted(rows, key=lambda row: row[3])
    lines = [
        "# TileLang layout regularization: free T4 A/B",
        "",
        f"- Status: `{payload['status']}`; correctness-complete cases: `18/18`.",
        f"- Overlay source: `{OVERLAY_SOURCE_SHA}`; native base: `{NATIVE_BASE_SHA}`.",
        f"- Runner source: `{payload['remote_provenance']['runner_source_sha']}`.",
        f"- GPU: `{payload['system']['gpu_name']}`; capability: `{payload['system']['gpu_capability']}`.",
        f"- Eager geometric mean: `{aggregate['eager_candidate_speedup_geomean']:.6f}x`.",
        f"- CUDA Graph geometric mean: `{aggregate['cuda_graph_candidate_speedup_geomean']:.6f}x`.",
        f"- Eager range: `{eager_sorted[0][2]:.6f}x` to `{eager_sorted[-1][2]:.6f}x`.",
        f"- CUDA Graph range: `{graph_sorted[0][3]:.6f}x` to `{graph_sorted[-1][3]:.6f}x`.",
        "",
        "## Family geometric means",
        "",
        "| Family | Cases | Eager | CUDA Graph |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family in sorted(family_values):
        values = family_values[family]
        lines.append(
            f"| {family} | {len(values['eager'])} | {geometric_mean(values['eager']):.6f}x | {geometric_mean(values['cuda_graph']):.6f}x |"
        )
    lines.extend(
        [
            "",
            "## Per-case results",
            "",
            "| Case | Family | Eager | CUDA Graph |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for name, family, eager, graph in rows:
        lines.append(f"| `{name}` | {family} | {eager:.6f}x | {graph:.6f}x |")
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "Each case compiled the same PrimFunc under the two selected policies. The run used 30 paired cycles, "
            "alternating baseline/candidate order, one second of warm-up per policy and mode, and batches calibrated "
            "to at least 50 ms. The raw JSON includes all samples, generated source, compiler resource usage, "
            "environment data, setup commands, public asset URLs, and SHA-256 identities.",
            "",
            "## Evidence boundary",
            "",
            payload["evidence_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any]) -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    RESULT_PATH.write_bytes(result_bytes)
    GZIP_PATH.write_bytes(gzip.compress(result_bytes, compresslevel=9, mtime=0))
    REPORT_PATH.write_text(report_markdown(payload), encoding="utf-8")
    checksums = {path.name: sha256_file(path) for path in (RESULT_PATH, GZIP_PATH, REPORT_PATH)}
    CHECKSUM_PATH.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    checksums[CHECKSUM_PATH.name] = sha256_file(CHECKSUM_PATH)
    return checksums


def main() -> int:
    started = time.time()
    records: list[dict[str, Any]] = []
    runner_source_sha = os.environ.get("TILELANG_RUNNER_SOURCE_SHA", "")
    controller_sha256 = os.environ.get("TILELANG_CONTROLLER_SHA256", "")
    payload: dict[str, Any] = {
        "schema": "tilelang-layout-regularized-colab-evidence-v1",
        "status": "failed",
        "repository": REPOSITORY,
        "started_unix": started,
        "system_controller": {
            "platform": platform.platform(),
            "python": sys.version,
            "nvidia_smi": command_output(["nvidia-smi"]),
        },
    }
    exit_code = 0
    try:
        if not HEX_40.fullmatch(runner_source_sha):
            raise RuntimeError("TILELANG_RUNNER_SOURCE_SHA must be a full lowercase Git commit")
        if not HEX_64.fullmatch(controller_sha256):
            raise RuntimeError("TILELANG_CONTROLLER_SHA256 must be a lowercase SHA-256 digest")
        scripts = verify_uploaded_scripts()
        overlay = prepare_overlay(records)
        payload = run_benchmark(records)
        payload["remote_provenance"] = {
            "runner_source_sha": runner_source_sha,
            "controller_sha256": controller_sha256,
            "uploaded_scripts": scripts,
            "overlay_release_tag": OVERLAY_RELEASE_TAG,
            "overlay_source_sha": OVERLAY_SOURCE_SHA,
            "native_base_sha": NATIVE_BASE_SHA,
            "downloads": overlay["downloads"],
            "overlay_manifest": overlay["manifest"],
            "setup_records": records,
        }
        payload["controller_duration_seconds"] = time.time() - started
    except Exception as error:  # noqa: BLE001 - always preserve a remote forensic record
        exit_code = 2
        payload.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "runner_source_sha": runner_source_sha,
                "controller_sha256": controller_sha256,
                "setup_records": records,
                "controller_duration_seconds": time.time() - started,
            }
        )

    if payload.get("status") == "complete":
        checksums = write_outputs(payload)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checksums = {RESULT_PATH.name: sha256_file(RESULT_PATH)}
    print(
        "TILELANG_LAYOUT_REGULARIZED_COLAB_RESULT="
        + json.dumps(
            {
                "status": payload.get("status"),
                "paths": {
                    "result": str(RESULT_PATH),
                    "gzip": str(GZIP_PATH),
                    "report": str(REPORT_PATH),
                    "checksums": str(CHECKSUM_PATH),
                },
                "sha256": checksums,
                "controller_duration_seconds": payload["controller_duration_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
