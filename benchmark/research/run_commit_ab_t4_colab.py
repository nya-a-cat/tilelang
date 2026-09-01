"""Run a fixed-baseline versus candidate TileLang A/B suite on one Colab T4.

Upload ``benchmark_commit_ab_t4.py`` beside this controller in ``WORK_DIR``.
The controller hash-checks every input, creates isolated Python environments
and TileLang caches, then alternates persistent baseline/candidate workers for
30 paired cycles. The installed TileLang compiler/runtime version is the sole
experimental variable. Candidate identity and overlay digests may be supplied
as exact environment values so a new compiler candidate does not require a
controller source edit.
"""

from __future__ import annotations

from collections import defaultdict, deque
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any
import urllib.request


REPOSITORY = "nya-a-cat/tilelang"


def exact_hex_env(name: str, default: str, length: int) -> str:
    value = os.environ.get(name, default)
    if re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{name} must contain exactly {length} lowercase hexadecimal characters")
    return value


BASELINE_COMMIT = "958d6d3bd24a31874a2bb189a9791347e855eecd"
BASELINE_BUILD_COMMIT = "a66f3860fdc3c1e5f6f78d96ee02d2e89953eda2"
CANDIDATE_COMMIT = exact_hex_env(
    "TILELANG_CANDIDATE_COMMIT",
    "2ac416ac8e2c60be29bba792d6bf6ded8315467e",
    40,
)
CANDIDATE_NATIVE_BASE = "4df968c3a85e723dc1870c62a0745284660bffd3"

BASELINE_TAG = f"colab-fast-{BASELINE_BUILD_COMMIT}"
BASELINE_WHEEL_NAME = "tilelang-0.1.13+cu130.gita66f3860-cp39-abi3-linux_x86_64.whl"
BASELINE_WHEEL_SHA256 = "058138c5b6ece9c0c7b6b20ebbbc572f510e210a6a65f5ef974e118c04f65cc8"
BASELINE_REVISIONS_SHA256 = "d96c6b0650c2f917f879255fdc9224a6c00420d627532941824bfbe3ebde8d68"

CANDIDATE_BASE_TAG = f"colab-fast-{CANDIDATE_NATIVE_BASE}"
CANDIDATE_BASE_WHEEL_NAME = "tilelang-0.1.13+cu130.git4df968c3-cp39-abi3-linux_x86_64.whl"
CANDIDATE_BASE_WHEEL_SHA256 = "f3ddcaa79cb10a5b61f4c8ed2eb6941fb30a3f53ddb45aeff2008b6824835f82"
CANDIDATE_OVERLAY_TAG = f"colab-native-{CANDIDATE_COMMIT}"
CANDIDATE_OVERLAY_ASSETS = {
    "apply_colab_native_overlay.sh": exact_hex_env(
        "TILELANG_CANDIDATE_OVERLAY_INSTALLER_SHA256",
        "945da1e58f8384ca09a8e5983e0c283bc716ba16372eb8f269de87deb7d96d98",
        64,
    ),
    "native-overlay-manifest.json": exact_hex_env(
        "TILELANG_CANDIDATE_OVERLAY_MANIFEST_SHA256",
        "7cc1bd1fcf3e247d1d0f9193eaadc67d5138fd9de46fb02338725086dc5b28f2",
        64,
    ),
    "tilelang-native-overlay.tar.gz": exact_hex_env(
        "TILELANG_CANDIDATE_OVERLAY_ARCHIVE_SHA256",
        "d8cefd2cb79b583c18808ab9fef90b6c08c63435508f6c296686e6feb176680a",
        64,
    ),
}

WORKER_NAME = "benchmark_commit_ab_t4.py"
WORKER_SHA256 = "8d9e4abefb841c1f70100741521dd243e9300f85b021bd844119a03d976156b3"
RPC_PREFIX = "TILELANG_COMMIT_AB_RPC="
SUITE_CONFIGS = {
    "layout": {
        "case_count": 18,
        "workload": "layout/elementwise",
        "evidence_boundary": (
            "This is an advisory free-T4 SM75 screen of the frozen 18-case layout/elementwise workload. "
            "It attributes observed differences to fixed baseline versus candidate TileLang compiler/runtime "
            "commits under default settings. Reduction/normalization, subgraphs, models, external suites, cold-L2, "
            "A100, H100, B200, RTX 5090, and MI300X primary gates remain open."
        ),
    },
    "reduction": {
        "case_count": 12,
        "workload": "reduction/normalization",
        "evidence_boundary": (
            "This is an advisory free-T4 SM75 screen of the frozen 12-case reduction/normalization workload: "
            "direct sum, max, min, absolute-sum, and bitwise-or reductions plus Softmax, RMSNorm, and LayerNorm. "
            "It attributes observed differences to fixed baseline versus candidate TileLang compiler/runtime "
            "commits under default settings. Subgraphs, models, external suites, cold-L2, A100, H100, B200, "
            "RTX 5090, and MI300X primary gates remain open."
        ),
    },
}
SUITE = os.environ.get("TILELANG_COMMIT_AB_SUITE", "layout")
if SUITE not in SUITE_CONFIGS:
    raise ValueError(f"unsupported TILELANG_COMMIT_AB_SUITE: {SUITE!r}")
SUITE_CONFIG = SUITE_CONFIGS[SUITE]
CASE_COUNT = int(SUITE_CONFIG["case_count"])
CAPTURE_SASS = os.environ.get("TILELANG_COMMIT_AB_CAPTURE_SASS", "0") == "1"
SASS_ONLY = os.environ.get("TILELANG_COMMIT_AB_SASS_ONLY", "0") == "1"
if SASS_ONLY and not CAPTURE_SASS:
    raise ValueError("TILELANG_COMMIT_AB_SASS_ONLY requires TILELANG_COMMIT_AB_CAPTURE_SASS=1")
RUN_NAME = f"{SUITE}-sass" if SASS_ONLY else SUITE
WORK_DIR = Path(f"/tmp/tilelang-fixed-commit-ab-{RUN_NAME}")
WORKER_UPLOAD_PATH = Path(os.environ.get("TILELANG_WORKER_UPLOAD_PATH", f"/tmp/{WORKER_NAME}"))
OUTPUT_DIR = Path(f"/tmp/tilelang-fixed-commit-ab-{RUN_NAME}-evidence")
RESULT_PATH = OUTPUT_DIR / f"tilelang-fixed-commit-ab-{RUN_NAME}-t4.json"
GZIP_PATH = OUTPUT_DIR / f"tilelang-fixed-commit-ab-{RUN_NAME}-t4.json.gz"
REPORT_PATH = OUTPUT_DIR / f"tilelang-fixed-commit-ab-{RUN_NAME}-t4-report.md"
CHECKSUM_PATH = OUTPUT_DIR / "SHA256SUMS"
BASELINE_ENV = WORK_DIR / "venv-baseline"
CANDIDATE_ENV = WORK_DIR / "venv-candidate"
BASELINE_CACHE = WORK_DIR / "cache-baseline"
CANDIDATE_CACHE = WORK_DIR / "cache-candidate"
BASELINE_INVENTORY = WORK_DIR / "baseline-inventory.json"
CANDIDATE_INVENTORY = WORK_DIR / "candidate-inventory.json"

CYCLES = 30
WARM_SECONDS = 1.0
MIN_BATCH_MS = 100.0
MAX_BATCH_ITERS = 65536
HEX_40 = re.compile(r"[0-9a-f]{40}")
HEX_64 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "minimum": min(values),
        "maximum": max(values),
    }


def ratio_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p10": percentile(values, 0.10),
        "p50": statistics.median(values),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - optional provenance remains forensic
        return f"{type(error).__name__}: {error}"


def run_checked(
    command: list[str],
    records: list[dict[str, Any]],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> str:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
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
            "output_tail": output[-16000:],
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


class WorkerClient:
    def __init__(
        self,
        label: str,
        python: Path,
        worker: Path,
        source_commit: str,
        cache_dir: Path,
        inventory_path: Path,
    ) -> None:
        self.label = label
        self.python = python
        self.worker = worker
        self.source_commit = source_commit
        self.cache_dir = cache_dir
        self.inventory_path = inventory_path
        self.process: subprocess.Popen[bytes] | None = None
        self.log_tail: deque[str] = deque(maxlen=400)
        self.ready: dict[str, Any] | None = None
        self.read_buffer = b""

    def start(self, timeout: float = 1200.0) -> dict[str, Any]:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "TILELANG_CACHE_DIR": str(self.cache_dir),
                "TILELANG_COMMIT_AB_LABEL": self.label,
                "TILELANG_COMMIT_AB_SOURCE_COMMIT": self.source_commit,
                "TILELANG_COMMIT_AB_SUITE": SUITE,
                "TILELANG_COMMIT_AB_CAPTURE_SASS": "1" if CAPTURE_SASS else "0",
                "TILELANG_COMMIT_AB_INVENTORY": str(self.inventory_path),
            }
        )
        self.process = subprocess.Popen(
            [str(self.python), str(self.worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
        )
        response = self._read_response(timeout)
        if response.get("status") != "ready":
            raise RuntimeError(f"{self.label} worker failed to start: {response}")
        self.ready = response
        return response

    def _read_response(self, timeout: float) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError(f"{self.label} worker is not running")
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            while True:
                while b"\n" in self.read_buffer:
                    line_bytes, self.read_buffer = self.read_buffer.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace")
                    self.log_tail.append(line)
                    if line.startswith(RPC_PREFIX):
                        return json.loads(line[len(RPC_PREFIX) :])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for {self.label} worker")
                events = selector.select(min(remaining, 5.0))
                if not events:
                    if self.process.poll() is not None:
                        raise RuntimeError(f"{self.label} worker exited {self.process.returncode}; tail={list(self.log_tail)[-20:]}")
                    continue
                chunk = os.read(self.process.stdout.fileno(), 1 << 16)
                if not chunk:
                    raise RuntimeError(f"{self.label} worker closed output; exit={self.process.poll()}; tail={list(self.log_tail)[-20:]}")
                self.read_buffer += chunk
        finally:
            selector.close()

    def request(self, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError(f"{self.label} worker is not running")
        self.process.stdin.write((json.dumps(payload, sort_keys=True) + "\n").encode())
        self.process.stdin.flush()
        response = self._read_response(timeout)
        if response.get("status") != "ok":
            raise RuntimeError(f"{self.label} worker command failed: {response}")
        return response

    def shutdown(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write((json.dumps({"command": "shutdown"}) + "\n").encode())
                self.process.stdin.flush()
                self._read_response(20.0)
            except Exception as error:  # noqa: BLE001 - bounded cleanup follows
                self.log_tail.append(f"shutdown_error={type(error).__name__}: {error}")
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)


def release_url(tag: str, name: str) -> str:
    encoded = name.replace("+", "%2B")
    return f"https://github.com/{REPOSITORY}/releases/download/{tag}/{encoded}"


def verify_worker() -> Path:
    worker = WORK_DIR / WORKER_NAME
    source = worker if worker.is_file() else WORKER_UPLOAD_PATH
    if not source.is_file():
        raise RuntimeError(f"missing uploaded worker: checked {worker} and {WORKER_UPLOAD_PATH}")
    actual_sha256 = sha256_file(source)
    if actual_sha256 != WORKER_SHA256:
        raise RuntimeError(f"worker hash mismatch: expected {WORKER_SHA256}, got {actual_sha256}")
    if source != worker:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, worker)
    actual_sha256 = sha256_file(worker)
    if actual_sha256 != WORKER_SHA256:
        raise RuntimeError(f"worker hash mismatch: expected {WORKER_SHA256}, got {actual_sha256}")
    return worker


def prepare_environments(records: list[dict[str, Any]]) -> dict[str, Any]:
    worker_source = (WORK_DIR / WORKER_NAME).read_bytes()
    for path in (BASELINE_ENV, CANDIDATE_ENV, BASELINE_CACHE, CANDIDATE_CACHE):
        if path.exists():
            shutil.rmtree(path)
    downloads_dir = WORK_DIR / "downloads"
    if downloads_dir.exists():
        shutil.rmtree(downloads_dir)
    downloads_dir.mkdir(parents=True)
    (WORK_DIR / WORKER_NAME).write_bytes(worker_source)

    downloads: list[dict[str, Any]] = []
    baseline_wheel = downloads_dir / BASELINE_WHEEL_NAME
    downloads.append(
        download(
            release_url(BASELINE_TAG, BASELINE_WHEEL_NAME),
            baseline_wheel,
            BASELINE_WHEEL_SHA256,
        )
    )
    baseline_revisions = downloads_dir / "baseline-source-revisions.txt"
    downloads.append(
        download(
            release_url(BASELINE_TAG, "source-revisions.txt"),
            baseline_revisions,
            BASELINE_REVISIONS_SHA256,
        )
    )
    candidate_base_wheel = downloads_dir / CANDIDATE_BASE_WHEEL_NAME
    downloads.append(
        download(
            release_url(CANDIDATE_BASE_TAG, CANDIDATE_BASE_WHEEL_NAME),
            candidate_base_wheel,
            CANDIDATE_BASE_WHEEL_SHA256,
        )
    )
    for name, expected_sha256 in CANDIDATE_OVERLAY_ASSETS.items():
        downloads.append(
            download(
                release_url(CANDIDATE_OVERLAY_TAG, name),
                downloads_dir / name,
                expected_sha256,
            )
        )

    venv_flags = ["-m", "venv", "--system-site-packages", "--without-pip"]
    run_checked([sys.executable, *venv_flags, str(BASELINE_ENV)], records)
    run_checked([sys.executable, *venv_flags, str(CANDIDATE_ENV)], records)
    baseline_python = BASELINE_ENV / "bin" / "python"
    candidate_python = CANDIDATE_ENV / "bin" / "python"
    dependencies = [
        "apache-tvm-ffi==0.1.12",
        "torch-c-dlpack-ext==0.1.5",
        "z3-solver==4.15.4.0",
    ]
    for python in (baseline_python, candidate_python):
        run_checked(
            [str(python), "-c", "import pip; print(pip.__version__, pip.__file__)"],
            records,
        )
        run_checked(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                *dependencies,
            ],
            records,
            timeout=300,
        )
    run_checked(
        [
            str(baseline_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            "--ignore-installed",
            "--no-deps",
            str(baseline_wheel),
        ],
        records,
    )
    run_checked(
        [
            str(candidate_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            "--ignore-installed",
            "--no-deps",
            str(candidate_base_wheel),
        ],
        records,
    )

    for label, python, environment_root in (
        ("baseline", baseline_python, BASELINE_ENV),
        ("candidate", candidate_python, CANDIDATE_ENV),
    ):
        package_output = run_checked(
            [
                str(python),
                "-c",
                "from importlib import metadata; print(metadata.distribution('tilelang').locate_file('tilelang').resolve())",
            ],
            records,
        )
        package_path = Path(package_output.strip())
        if not package_path.is_relative_to(environment_root.resolve()):
            raise RuntimeError(f"{label} TileLang resolved outside its isolated environment: {package_path}")

    manifest_path = downloads_dir / "native-overlay-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema": "tilelang-colab-native-overlay-v1",
        "repository": REPOSITORY,
        "source_sha": CANDIDATE_COMMIT,
        "native_base_sha": CANDIDATE_NATIVE_BASE,
        "native_base_tag": CANDIDATE_BASE_TAG,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"unexpected candidate overlay manifest {key}: {manifest.get(key)!r}")
    if manifest.get("overlay_sha256") != CANDIDATE_OVERLAY_ASSETS["tilelang-native-overlay.tar.gz"]:
        raise RuntimeError("candidate overlay archive hash differs from its manifest")
    installer = downloads_dir / "apply_colab_native_overlay.sh"
    installer.chmod(0o755)
    install_env = os.environ.copy()
    install_env["PYTHON"] = str(candidate_python)
    run_checked(
        [
            str(installer),
            str(downloads_dir / "tilelang-native-overlay.tar.gz"),
            str(manifest_path),
        ],
        records,
        env=install_env,
    )

    revisions_text = baseline_revisions.read_text(encoding="utf-8")
    if f"commit={BASELINE_BUILD_COMMIT}" not in revisions_text:
        raise RuntimeError("baseline provenance identifies a different build commit")
    return {
        "downloads": downloads,
        "baseline_revisions": revisions_text,
        "candidate_overlay_manifest": manifest,
        "baseline_python": str(baseline_python),
        "candidate_python": str(candidate_python),
    }


def load_and_compare_inventories() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = json.loads(BASELINE_INVENTORY.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE_INVENTORY.read_text(encoding="utf-8"))
    if baseline.get("status") != "ready" or candidate.get("status") != "ready":
        raise RuntimeError("a worker inventory is incomplete")
    if baseline.get("suite") != SUITE or candidate.get("suite") != SUITE:
        raise RuntimeError(f"worker inventory suite mismatch for {SUITE}")
    if baseline.get("capture_sass") is not CAPTURE_SASS or candidate.get("capture_sass") is not CAPTURE_SASS:
        raise RuntimeError(f"worker SASS capture mismatch: expected {CAPTURE_SASS}")
    baseline_cases = {case["name"]: case for case in baseline["cases"]}
    candidate_cases = {case["name"]: case for case in candidate["cases"]}
    if list(baseline_cases) != list(candidate_cases) or len(baseline_cases) != CASE_COUNT:
        raise RuntimeError(f"baseline and candidate case lists differ from the frozen {CASE_COUNT}-case {SUITE} suite")
    identity_fields = (
        "family",
        "canonical_primfunc_sha256",
        "input_sha256",
        "output_shape",
        "output_dtype",
        "atol",
        "rtol",
    )
    for name in baseline_cases:
        for field in identity_fields:
            if baseline_cases[name].get(field) != candidate_cases[name].get(field):
                raise RuntimeError(f"workload drift for {name} field {field}")
    baseline_gpu = baseline["system"].get("gpu_capability")
    candidate_gpu = candidate["system"].get("gpu_capability")
    if baseline_gpu != [7, 5] or candidate_gpu != [7, 5]:
        raise RuntimeError(f"expected T4 SM75 workers, got {baseline_gpu!r} and {candidate_gpu!r}")
    return baseline, candidate


def summarize_sass_capture(capture: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(capture, dict):
        raise RuntimeError("requested SASS capture is missing")
    required = ("sass_sha256", "sass_chars", "instruction_count", "groups", "opcodes", "duration_seconds")
    missing = [field for field in required if field not in capture]
    if missing:
        raise RuntimeError(f"SASS capture fields are missing: {missing}")
    return {
        "sass_sha256": capture["sass_sha256"],
        "sass_chars": capture["sass_chars"],
        "registers": capture.get("registers"),
        "instruction_count": capture["instruction_count"],
        "groups": capture["groups"],
        "opcodes": capture["opcodes"],
        "duration_seconds": capture["duration_seconds"],
    }


def build_sass_diagnostic_cases(
    baseline_inventory: dict[str, Any],
    candidate_inventory: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_cases = {case["name"]: case for case in baseline_inventory["cases"]}
    candidate_cases = {case["name"]: case for case in candidate_inventory["cases"]}
    results: list[dict[str, Any]] = []
    for name, baseline_case in baseline_cases.items():
        candidate_case = candidate_cases[name]
        baseline_sass = summarize_sass_capture(baseline_case.get("sass_capture"))
        candidate_sass = summarize_sass_capture(candidate_case.get("sass_capture"))
        group_names = sorted(set(baseline_sass["groups"]) | set(candidate_sass["groups"]))
        group_delta = {
            group: int(candidate_sass["groups"].get(group, 0)) - int(baseline_sass["groups"].get(group, 0)) for group in group_names
        }
        baseline_registers = baseline_sass.get("registers")
        candidate_registers = candidate_sass.get("registers")
        register_delta = (
            int(candidate_registers) - int(baseline_registers)
            if baseline_registers is not None and candidate_registers is not None
            else None
        )
        results.append(
            {
                "name": name,
                "family": baseline_case["family"],
                "status": "complete",
                "canonical_primfunc_sha256": baseline_case["canonical_primfunc_sha256"],
                "generated_source_changed": (baseline_case["generated_source_sha256"] != candidate_case["generated_source_sha256"]),
                "sass_changed": baseline_sass["sass_sha256"] != candidate_sass["sass_sha256"],
                "baseline": {
                    "dynamic_shared_bytes": baseline_case["dynamic_shared_bytes"],
                    "sass": baseline_sass,
                },
                "candidate": {
                    "dynamic_shared_bytes": candidate_case["dynamic_shared_bytes"],
                    "sass": candidate_sass,
                },
                "candidate_minus_baseline": {
                    "dynamic_shared_bytes": (int(candidate_case["dynamic_shared_bytes"]) - int(baseline_case["dynamic_shared_bytes"])),
                    "registers": register_delta,
                    "instruction_count": (int(candidate_sass["instruction_count"]) - int(baseline_sass["instruction_count"])),
                    "groups": group_delta,
                },
            }
        )
    return results


def aggregate_sass_results(cases: list[dict[str, Any]]) -> dict[str, Any]:
    group_names = sorted({group for case in cases for label in ("baseline", "candidate") for group in case[label]["sass"]["groups"]})
    group_totals = {
        label: {group: sum(int(case[label]["sass"]["groups"].get(group, 0)) for case in cases) for group in group_names}
        for label in ("baseline", "candidate")
    }
    return {
        "complete_cases": len(cases),
        "total_cases": CASE_COUNT,
        "source_changed_cases": sum(bool(case["generated_source_changed"]) for case in cases),
        "sass_changed_cases": sum(bool(case["sass_changed"]) for case in cases),
        "instruction_count": {
            label: sum(int(case[label]["sass"]["instruction_count"]) for case in cases) for label in ("baseline", "candidate")
        },
        "group_totals": group_totals,
        "barrier_reduced_cases": [case["name"] for case in cases if int(case["candidate_minus_baseline"]["groups"].get("barrier", 0)) < 0],
        "barrier_increased_cases": [
            case["name"] for case in cases if int(case["candidate_minus_baseline"]["groups"].get("barrier", 0)) > 0
        ],
    }


def measure_mode(
    name: str,
    mode: str,
    workers: dict[str, WorkerClient],
    warm_reverse: bool,
) -> dict[str, Any]:
    warm_order = ["candidate", "baseline"] if warm_reverse else ["baseline", "candidate"]
    warm_records: dict[str, Any] = {}
    for label in warm_order:
        response = workers[label].request(
            {"command": "warm", "case": name, "mode": mode, "seconds": WARM_SECONDS},
            timeout=WARM_SECONDS + 60.0,
        )
        warm_records[label] = response["result"]

    iterations = 64
    probes: list[dict[str, Any]] = []
    while True:
        probe_record: dict[str, Any] = {"iterations": iterations}
        for label in ("baseline", "candidate"):
            response = workers[label].request({"command": "probe", "case": name, "mode": mode, "iterations": iterations})
            probe_record[label] = response["result"]
        probes.append(probe_record)
        minimum_ms = min(float(probe_record[label]["event_batch_ms"]) for label in ("baseline", "candidate"))
        if minimum_ms >= MIN_BATCH_MS or iterations >= MAX_BATCH_ITERS:
            break
        iterations = min(iterations * 2, MAX_BATCH_ITERS)

    samples: dict[str, dict[str, list[float]]] = {
        label: {"event_per_launch_us": [], "wall_per_launch_us": []} for label in ("baseline", "candidate")
    }
    cycle_records: list[dict[str, Any]] = []
    for cycle in range(CYCLES):
        order = ["baseline", "candidate", "candidate", "baseline"] if cycle % 2 == 0 else ["candidate", "baseline", "baseline", "candidate"]
        cycle_values: dict[str, dict[str, list[float]]] = {
            label: {"event_per_launch_us": [], "wall_per_launch_us": []} for label in ("baseline", "candidate")
        }
        for label in order:
            response = workers[label].request({"command": "measure", "case": name, "mode": mode, "iterations": iterations})
            result = response["result"]
            for metric in ("event_per_launch_us", "wall_per_launch_us"):
                value = float(result[metric])
                samples[label][metric].append(value)
                cycle_values[label][metric].append(value)
        cycle_p50 = {
            label: {metric: statistics.median(cycle_values[label][metric]) for metric in ("event_per_launch_us", "wall_per_launch_us")}
            for label in ("baseline", "candidate")
        }
        cycle_records.append(
            {
                "cycle": cycle,
                "order": order,
                "p50_us": cycle_p50,
                "event_candidate_speedup_over_baseline": (
                    cycle_p50["baseline"]["event_per_launch_us"] / cycle_p50["candidate"]["event_per_launch_us"]
                ),
                "wall_candidate_speedup_over_baseline": (
                    cycle_p50["baseline"]["wall_per_launch_us"] / cycle_p50["candidate"]["wall_per_launch_us"]
                ),
            }
        )

    sample_seconds = {label: sum(samples[label]["event_per_launch_us"]) * iterations / 1e6 for label in ("baseline", "candidate")}
    if min(sample_seconds.values()) < 3.0:
        raise RuntimeError(f"sample duration below three seconds for {name}/{mode}: {sample_seconds}")
    event_ratios = [record["event_candidate_speedup_over_baseline"] for record in cycle_records]
    wall_ratios = [record["wall_candidate_speedup_over_baseline"] for record in cycle_records]
    return {
        "iterations_per_sample": iterations,
        "cycles": CYCLES,
        "warmup": warm_records,
        "calibration_probes": probes,
        "sample_seconds": sample_seconds,
        "samples_us": samples,
        "cycle_records": cycle_records,
        "summary_us": {label: {metric: summary(values) for metric, values in metrics.items()} for label, metrics in samples.items()},
        "event_candidate_speedup_over_baseline": statistics.median(event_ratios),
        "event_paired_speedup": ratio_summary(event_ratios),
        "wall_candidate_speedup_over_baseline": statistics.median(wall_ratios),
        "wall_paired_speedup": ratio_summary(wall_ratios),
    }


def run_benchmark(
    workers: dict[str, WorkerClient],
    baseline_inventory: dict[str, Any],
    candidate_inventory: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_cases = {case["name"]: case for case in baseline_inventory["cases"]}
    candidate_cases = {case["name"]: case for case in candidate_inventory["cases"]}
    results: list[dict[str, Any]] = []
    for case_index, name in enumerate(baseline_cases):
        baseline_case = baseline_cases[name]
        candidate_case = candidate_cases[name]
        if not baseline_case["cuda_graph_captured"] or not candidate_case["cuda_graph_captured"]:
            raise RuntimeError(f"CUDA Graph capture incomplete for {name}")
        result = {
            "name": name,
            "family": baseline_case["family"],
            "canonical_primfunc_sha256": baseline_case["canonical_primfunc_sha256"],
            "input_sha256": baseline_case["input_sha256"],
            "generated_source_changed": (baseline_case["generated_source_sha256"] != candidate_case["generated_source_sha256"]),
            "baseline_generated_source_sha256": baseline_case["generated_source_sha256"],
            "candidate_generated_source_sha256": candidate_case["generated_source_sha256"],
            "baseline_resource_usage": baseline_case["resource_usage"],
            "candidate_resource_usage": candidate_case["resource_usage"],
            "eager": measure_mode(name, "eager", workers, bool(case_index % 2)),
            "cuda_graph": measure_mode(name, "cuda_graph", workers, not bool(case_index % 2)),
            "status": "complete",
        }
        results.append(result)
        print(
            "TILELANG_COMMIT_AB_PROGRESS="
            + json.dumps(
                {
                    "completed": len(results),
                    "total": len(baseline_cases),
                    "case": name,
                    "eager_speedup": result["eager"]["event_candidate_speedup_over_baseline"],
                    "graph_speedup": result["cuda_graph"]["event_candidate_speedup_over_baseline"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return results


def aggregate_results(cases: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"eager": [], "cuda_graph": []})
    for case in cases:
        for mode in ("eager", "cuda_graph"):
            families[case["family"]][mode].append(float(case[mode]["event_candidate_speedup_over_baseline"]))
    return {
        "complete_cases": len(cases),
        "total_cases": CASE_COUNT,
        "eager_event_speedup_geomean": geometric_mean([float(case["eager"]["event_candidate_speedup_over_baseline"]) for case in cases]),
        "cuda_graph_event_speedup_geomean": geometric_mean(
            [float(case["cuda_graph"]["event_candidate_speedup_over_baseline"]) for case in cases]
        ),
        "source_changed_cases": sum(bool(case["generated_source_changed"]) for case in cases),
        "critical_regressions_below_0_97": [
            {"case": case["name"], "mode": mode, "speedup": case[mode]["event_candidate_speedup_over_baseline"]}
            for case in cases
            for mode in ("eager", "cuda_graph")
            if float(case[mode]["event_candidate_speedup_over_baseline"]) < 0.97
        ],
        "families": {
            family: {mode: geometric_mean(values) for mode, values in modes.items()} for family, modes in sorted(families.items())
        },
    }


def sass_report_markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    baseline_groups = aggregate["group_totals"]["baseline"]
    candidate_groups = aggregate["group_totals"]["candidate"]
    lines = [
        f"# TileLang fixed-commit A/B: free T4 {SUITE} SASS diagnostic",
        "",
        f"- Status: `{payload['status']}`; disassembled cases: `{CASE_COUNT}/{CASE_COUNT}`.",
        f"- Fixed baseline: `{BASELINE_COMMIT}` (build-only scaffold `{BASELINE_BUILD_COMMIT}`).",
        f"- Candidate compiler/runtime: `{CANDIDATE_COMMIT}`.",
        f"- Generated-source changes: `{aggregate['source_changed_cases']}/{CASE_COUNT}` cases.",
        f"- SASS changes: `{aggregate['sass_changed_cases']}/{CASE_COUNT}` cases.",
        f"- Total static instructions: `{aggregate['instruction_count']['baseline']}` → `{aggregate['instruction_count']['candidate']}`.",
        f"- Total barrier instructions: `{baseline_groups.get('barrier', 0)}` → `{candidate_groups.get('barrier', 0)}`.",
        f"- Total shuffle instructions: `{baseline_groups.get('shuffle', 0)}` → `{candidate_groups.get('shuffle', 0)}`.",
        "",
        "## Per-case static metrics",
        "",
        "| Case | Barriers | Shuffles | Shared load/store | Instructions | Registers | Dynamic shared bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        baseline = case["baseline"]
        candidate = case["candidate"]
        baseline_groups = baseline["sass"]["groups"]
        candidate_groups = candidate["sass"]["groups"]
        lines.append(
            f"| `{case['name']}` | "
            f"{baseline_groups.get('barrier', 0)}→{candidate_groups.get('barrier', 0)} | "
            f"{baseline_groups.get('shuffle', 0)}→{candidate_groups.get('shuffle', 0)} | "
            f"{baseline_groups.get('shared_load', 0)}/{baseline_groups.get('shared_store', 0)}→"
            f"{candidate_groups.get('shared_load', 0)}/{candidate_groups.get('shared_store', 0)} | "
            f"{baseline['sass']['instruction_count']}→{candidate['sass']['instruction_count']} | "
            f"{baseline['sass'].get('registers')}→{candidate['sass'].get('registers')} | "
            f"{baseline['dynamic_shared_bytes']}→{candidate['dynamic_shared_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "Each isolated worker compiles and correctness-checks the same frozen PrimFuncs, then recompiles its "
            "generated CUDA source through the installed TileLang `JITKernel._get_sass` path for the resident T4 "
            "target. The result retains complete SASS, opcode counts, compiler resources, dynamic shared memory, "
            "generated source, inputs, environments, and setup records. This diagnostic performs no runtime timing.",
            "",
            "## Evidence boundary",
            "",
            payload["evidence_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def report_markdown(payload: dict[str, Any]) -> str:
    if payload.get("sass_only"):
        return sass_report_markdown(payload)
    aggregate = payload["aggregate"]
    lines = [
        f"# TileLang fixed-commit A/B: free T4 {SUITE} screen",
        "",
        f"- Status: `{payload['status']}`; correctness-complete cases: `{CASE_COUNT}/{CASE_COUNT}`.",
        f"- Frozen suite: `{SUITE}` ({SUITE_CONFIG['workload']}).",
        f"- Fixed baseline: `{BASELINE_COMMIT}` (build-only scaffold `{BASELINE_BUILD_COMMIT}`).",
        f"- Candidate compiler/runtime: `{CANDIDATE_COMMIT}`.",
        "- Comparison: byte-identical PrimFuncs and inputs, isolated environments and caches, default backend settings.",
        f"- Eager CUDA-event geometric mean: `{aggregate['eager_event_speedup_geomean']:.6f}x`.",
        f"- CUDA Graph CUDA-event geometric mean: `{aggregate['cuda_graph_event_speedup_geomean']:.6f}x`.",
        f"- Generated-source changes: `{aggregate['source_changed_cases']}/{CASE_COUNT}` cases.",
        f"- Critical slices below 0.97x: `{len(aggregate['critical_regressions_below_0_97'])}`.",
        "",
        "## Per-case results",
        "",
        "| Case | Family | Eager | CUDA Graph | Source changed |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| `{case['name']}` | {case['family']} | "
            f"{case['eager']['event_candidate_speedup_over_baseline']:.6f}x | "
            f"{case['cuda_graph']['event_candidate_speedup_over_baseline']:.6f}x | "
            f"{str(case['generated_source_changed']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "Two persistent workers use separate Python environments, TileLang installations, and cache directories. "
            "Each case uses one shared iteration count. Thirty paired cycles alternate baseline, candidate, candidate, "
            "baseline, with the reverse order on odd cycles. Every variant and launch mode receives at least one second "
            "of warm-up and at least three seconds of CUDA-event sampling. CUDA-event and synchronized wall samples, "
            "p50/p90/p99, generated source, compiler resources, inputs, environment, and setup records are retained.",
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
    paths = [RESULT_PATH]
    if payload.get("status") == "complete":
        GZIP_PATH.write_bytes(gzip.compress(result_bytes, compresslevel=9, mtime=0))
        REPORT_PATH.write_text(report_markdown(payload), encoding="utf-8")
        paths.extend((GZIP_PATH, REPORT_PATH))
    checksums = {path.name: sha256_file(path) for path in paths}
    CHECKSUM_PATH.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    checksums[CHECKSUM_PATH.name] = sha256_file(CHECKSUM_PATH)
    return checksums


def main() -> int:
    started = time.time()
    records: list[dict[str, Any]] = []
    workers: dict[str, WorkerClient] = {}
    runner_source_sha = os.environ.get("TILELANG_RUNNER_SOURCE_SHA", "")
    controller_sha256 = os.environ.get("TILELANG_CONTROLLER_SHA256", "")
    payload: dict[str, Any] = {
        "schema": "tilelang-fixed-commit-colab-ab-v3",
        "status": "failed",
        "repository": REPOSITORY,
        "suite": SUITE,
        "suite_case_count": CASE_COUNT,
        "capture_sass": CAPTURE_SASS,
        "sass_only": SASS_ONLY,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_build_commit": BASELINE_BUILD_COMMIT,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_overlay_tag": CANDIDATE_OVERLAY_TAG,
        "candidate_overlay_asset_sha256": CANDIDATE_OVERLAY_ASSETS,
        "started_unix": started,
        "controller_system": {
            "platform": platform.platform(),
            "python": sys.version,
            "nvidia_smi": command_output(["nvidia-smi"]),
        },
        "evidence_boundary": (
            "This is a compile-and-disassemble diagnostic on one free Colab T4 SM75. It compares generated source, "
            "SASS, static instruction mix, registers, and dynamic shared memory for the fixed TileLang commits. "
            "Runtime conclusions remain in the paired 30-cycle reduction release; this diagnostic contains no "
            "runtime timing. A100, H100, B200, RTX 5090, and MI300X machine-code gates remain open."
            if SASS_ONLY
            else SUITE_CONFIG["evidence_boundary"]
        ),
    }
    exit_code = 0
    try:
        if not HEX_40.fullmatch(runner_source_sha):
            raise RuntimeError("TILELANG_RUNNER_SOURCE_SHA must be a full lowercase Git commit")
        if not HEX_64.fullmatch(controller_sha256):
            raise RuntimeError("TILELANG_CONTROLLER_SHA256 must be a lowercase SHA-256 digest")
        actual_controller_sha256 = sha256_file(Path(__file__).resolve())
        if actual_controller_sha256 != controller_sha256:
            raise RuntimeError(f"controller hash mismatch: expected {controller_sha256}, got {actual_controller_sha256}")
        worker = verify_worker()
        environment = prepare_environments(records)
        baseline_python = Path(environment["baseline_python"])
        candidate_python = Path(environment["candidate_python"])
        workers = {
            "baseline": WorkerClient(
                "baseline",
                baseline_python,
                worker,
                BASELINE_COMMIT,
                BASELINE_CACHE,
                BASELINE_INVENTORY,
            ),
            "candidate": WorkerClient(
                "candidate",
                candidate_python,
                worker,
                CANDIDATE_COMMIT,
                CANDIDATE_CACHE,
                CANDIDATE_INVENTORY,
            ),
        }
        ready = {
            "baseline": workers["baseline"].start(),
            "candidate": workers["candidate"].start(),
        }
        if ready["baseline"].get("suite") != SUITE or ready["candidate"].get("suite") != SUITE:
            raise RuntimeError(f"worker ready suite mismatch for {SUITE}")
        if ready["baseline"].get("capture_sass") is not CAPTURE_SASS or ready["candidate"].get("capture_sass") is not CAPTURE_SASS:
            raise RuntimeError(f"worker ready SASS capture mismatch: expected {CAPTURE_SASS}")
        baseline_inventory, candidate_inventory = load_and_compare_inventories()
        if SASS_ONLY:
            cases = build_sass_diagnostic_cases(baseline_inventory, candidate_inventory)
            aggregate = aggregate_sass_results(cases)
            protocol = {
                "suite": SUITE,
                "case_count": CASE_COUNT,
                "mode": "compile-and-disassemble",
                "runtime_measurement": False,
                "sass_method": "JITKernel._get_sass generated-source recompile",
                "pass_configs": None,
            }
        else:
            cases = run_benchmark(workers, baseline_inventory, candidate_inventory)
            aggregate = aggregate_results(cases)
            protocol = {
                "suite": SUITE,
                "case_count": CASE_COUNT,
                "cycles": CYCLES,
                "paired_order_even": ["baseline", "candidate", "candidate", "baseline"],
                "paired_order_odd": ["candidate", "baseline", "baseline", "candidate"],
                "warm_seconds_per_variant_mode": WARM_SECONDS,
                "minimum_batch_ms": MIN_BATCH_MS,
                "minimum_sample_seconds_per_variant_mode": 3.0,
                "pass_configs": None,
                "cache_mode": "hot_l2",
            }
        if aggregate["complete_cases"] != CASE_COUNT:
            raise RuntimeError(f"case completion drifted: {aggregate}")
        payload.update(
            {
                "status": "complete",
                "runner_source_sha": runner_source_sha,
                "controller_sha256": controller_sha256,
                "worker_sha256": WORKER_SHA256,
                "protocol": protocol,
                "environment": environment,
                "worker_ready": ready,
                "baseline_inventory": baseline_inventory,
                "candidate_inventory": candidate_inventory,
                "cases": cases,
                "aggregate": aggregate,
            }
        )
    except Exception as error:  # noqa: BLE001 - always write a forensic result
        exit_code = 2
        payload.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "runner_source_sha": runner_source_sha,
                "controller_sha256": controller_sha256,
            }
        )
    finally:
        for worker in workers.values():
            worker.shutdown()
        payload["worker_log_tails"] = {label: list(worker.log_tail) for label, worker in workers.items()}
        payload["setup_records"] = records
        payload["duration_seconds"] = time.time() - started
        payload["finished_nvidia_smi"] = command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,memory.used",
                "--format=csv,noheader",
            ]
        )

    checksums = write_outputs(payload)
    print(
        "TILELANG_FIXED_COMMIT_COLAB_RESULT="
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
                "duration_seconds": payload["duration_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
