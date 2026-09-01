"""One-shot Vast.ai runtime A/B for consumer Blackwell layout divergence cases.

The runner is intentionally self-contained and hash-pins every downloaded
TileLang artifact.  It verifies that the allocated GPU is compute capability
12.0 before spending time on installation or compilation, then writes all
results under ``/workspace/evidence`` for collection before instance teardown.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import urllib.request


BENCHMARK_SOURCE_SHA = "8c0367aa70431026da0a51486edf74c9717c7796"
RUNTIME_SOURCE_SHA = "cffe79004aab69c04510acdb89cf0775df20dc9a"
NATIVE_BASE_SHA = "dd39e5a8e8ec33a75b9eb1d984e8a7db9f8656e4"
RUNTIME_RELEASE_TAG = f"colab-layout-auto-{RUNTIME_SOURCE_SHA}"
RUNTIME_RELEASE_ROOT = f"https://github.com/nya-a-cat/tilelang/releases/download/{RUNTIME_RELEASE_TAG}"
WHEEL_NAME = "tilelang-0.1.13+cu130.gitdd39e5a8-cp39-abi3-linux_x86_64.whl"
WHEEL_SHA256 = "dfd084a99b3a6c4a7caf36a85e1629398fe3e45da8440c6c85fc3496f992c93c"
WHEEL_URL = (
    "https://github.com/nya-a-cat/tilelang/releases/download/"
    "colab-fast-dd39e5a8e8ec33a75b9eb1d984e8a7db9f8656e4/"
    "tilelang-0.1.13%2Bcu130.gitdd39e5a8-cp39-abi3-linux_x86_64.whl"
)
RUNTIME_ASSETS = {
    "tilelang-python-overlay.tar.gz": "28e13007bcf23e3ec58a380505a76a1bbec9909d7ed21358791fe46428b47dd8",
    "overlay-manifest.json": "faf6f21bf29e48646f287ebc1793fa9ff0f72fd11fb6d99b521f198aaa5e574f",
    "apply_colab_python_overlay.sh": "168758a166315d373fb2b73c03ac4dda4d12f9133333ad60c3121d5f0a1d17b1",
}
BENCHMARK_FILES = {
    "benchmark_layout_cost_models_t4.py": "71b5dd6db0f9335f774426a59de5d0c7264183a9bf52f726ab6acc2be7dfdbfa",
    "benchmark_layout_normalization_policies_t4.py": "a424500185fbc76a32901527ee917e9116b743a5ec820c32cfb8d67a8117009a",
    "scan_layout_policy_divergence.py": "145027fa187905485cc7fca26472414d6996f573b678d2078c45c22f728bc9cb",
    "benchmark_layout_divergent_cases_t4.py": "ad2a7b64710997b4b51b8170aad29e36ae94a314d778ec9aad3de997765ffdfc",
}
RAW_ROOT = f"https://raw.githubusercontent.com/nya-a-cat/tilelang/{BENCHMARK_SOURCE_SHA}/benchmark/research"
EVIDENCE_ROOT = Path(os.environ.get("TILELANG_EVIDENCE_ROOT", "/workspace/evidence"))
WORK_ROOT = Path(os.environ.get("TILELANG_WORK_ROOT", "/workspace/tilelang-work"))
RAW_RESULT = EVIDENCE_ROOT / "layout-divergent-runtime-raw.json"
RESULT = EVIDENCE_ROOT / "layout-divergent-blackwell-run.json"
LOG = EVIDENCE_ROOT / "layout-divergent-blackwell-execution.log"
MANIFEST = EVIDENCE_ROOT / "SHA256SUMS"
EXPECTED_CAPABILITY = (12, 0)
EXPECTED_GPU_NAME = os.environ.get("TILELANG_EXPECTED_GPU_NAME", "RTX 5060 Ti")
RUN_LOGS: list[str] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str],
    *,
    timeout: int = 600,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=cwd,
    )
    RUN_LOGS.append(
        "\n".join(
            [
                f"$ {command!r}",
                f"returncode={completed.returncode} duration_seconds={time.perf_counter() - started:.9f}",
                "[stdout]",
                completed.stdout,
                "[stderr]",
                completed.stderr,
            ]
        )
    )
    if completed.returncode != 0:
        output = (completed.stdout + "\n" + completed.stderr)[-12000:]
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {command!r}\n{output}")
    return completed


def command_output(command: list[str]) -> str:
    try:
        return run(command, timeout=30).stdout.strip()
    except Exception as error:  # noqa: BLE001 - optional provenance must survive failures
        return f"{type(error).__name__}: {error}"


def download_exact(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    actual = sha256(destination)
    RUN_LOGS.append(f"download {url}\npath={destination}\nsha256={actual}\n")
    if actual != expected_sha256:
        raise RuntimeError(f"download SHA-256 mismatch for {destination.name}: {actual}")


def torch_identity() -> dict[str, object]:
    import torch

    return {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }


def verify_blackwell() -> dict[str, object]:
    identity = torch_identity()
    if not identity["cuda_available"]:
        raise RuntimeError(f"CUDA is unavailable: {identity}")
    capability = tuple(identity["gpu_capability"])
    if capability != EXPECTED_CAPABILITY:
        raise RuntimeError(f"expected compute capability {EXPECTED_CAPABILITY}, got {capability}: {identity}")
    expected_name = EXPECTED_GPU_NAME.lower().replace("nvidia", "").strip()
    actual_name = str(identity["gpu_name"]).lower().replace("nvidia", "").strip()
    if expected_name and expected_name not in actual_name:
        raise RuntimeError(f"expected GPU name containing {EXPECTED_GPU_NAME!r}, got {identity['gpu_name']!r}")
    return identity


def install_exact_runtime(initial_torch: dict[str, object]) -> dict[str, object]:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    downloaded_runtime_assets: dict[str, str] = {}
    for name, expected_hash in RUNTIME_ASSETS.items():
        destination = WORK_ROOT / name
        download_exact(f"{RUNTIME_RELEASE_ROOT}/{name}", destination, expected_hash)
        downloaded_runtime_assets[name] = sha256(destination)

    manifest = json.loads((WORK_ROOT / "overlay-manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha"] != RUNTIME_SOURCE_SHA or manifest["native_base_sha"] != NATIVE_BASE_SHA:
        raise RuntimeError(f"overlay manifest identity mismatch: {manifest}")
    if manifest["overlay_sha256"] != RUNTIME_ASSETS["tilelang-python-overlay.tar.gz"]:
        raise RuntimeError(f"overlay manifest digest mismatch: {manifest['overlay_sha256']}")

    wheel = WORK_ROOT / WHEEL_NAME
    download_exact(WHEEL_URL, wheel, WHEEL_SHA256)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            str(wheel),
        ],
        timeout=900,
    )
    installed_torch = torch_identity()
    if installed_torch["version"] != initial_torch["version"]:
        raise RuntimeError(f"TileLang install changed PyTorch: before={initial_torch}, after={installed_torch}")

    overlay_env = dict(os.environ)
    overlay_env["PYTHON"] = sys.executable
    overlay_apply = run(
        ["bash", str(WORK_ROOT / "apply_colab_python_overlay.sh"), str(WORK_ROOT / "tilelang-python-overlay.tar.gz")],
        timeout=120,
        env=overlay_env,
    )
    identity_code = (
        "import json,pathlib,tilelang; "
        "p=pathlib.Path(tilelang.__file__).resolve().parent; "
        "print(json.dumps({'package':str(p),'version':tilelang.__version__,"
        "'identity':json.loads((p/'_python_overlay_identity.json').read_text())},sort_keys=True))"
    )
    identity = json.loads(run([sys.executable, "-c", identity_code], timeout=60).stdout.strip())
    if identity["identity"]["source_sha"] != RUNTIME_SOURCE_SHA:
        raise RuntimeError(f"overlay source mismatch: {identity['identity']}")
    if identity["identity"]["native_base_sha"] != NATIVE_BASE_SHA:
        raise RuntimeError(f"native base mismatch: {identity['identity']}")

    downloaded_benchmarks: dict[str, str] = {}
    for name, expected_hash in BENCHMARK_FILES.items():
        destination = WORK_ROOT / name
        download_exact(f"{RAW_ROOT}/{name}", destination, expected_hash)
        downloaded_benchmarks[name] = sha256(destination)

    return {
        "wheel_name": WHEEL_NAME,
        "wheel_sha256": sha256(wheel),
        "runtime_source_sha": RUNTIME_SOURCE_SHA,
        "benchmark_source_sha": BENCHMARK_SOURCE_SHA,
        "downloaded_runtime_assets": downloaded_runtime_assets,
        "downloaded_benchmarks": downloaded_benchmarks,
        "overlay_apply_output": overlay_apply.stdout.strip(),
        "overlay_manifest": manifest,
        "torch_before_install": initial_torch,
        "torch_after_install": installed_torch,
        **identity,
    }


def execute_benchmark() -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    env = dict(os.environ)
    env.update(
        {
            "TILELANG_SOURCE_SHA": BENCHMARK_SOURCE_SHA,
            "TILELANG_NATIVE_BASE_SHA": NATIVE_BASE_SHA,
            "TILELANG_LAYOUT_AB_RESULT": str(RAW_RESULT),
            "TILELANG_LAYOUT_AB_POLICIES": "register-count,io-aware",
            "TILELANG_LAYOUT_AB_SCHEMA": "tilelang-layout-divergent-runtime-ab-v2",
            "TILELANG_LAYOUT_AB_CYCLES": "30",
            "TILELANG_LAYOUT_AB_MIN_BATCH_MS": "50",
            "TILELANG_LAYOUT_AB_WARM_SECONDS": "1",
            "TILELANG_CACHE_DIR": str(WORK_ROOT / "cache"),
            "TILELANG_DISABLE_CACHE": "0",
            "TILELANG_KERNEL_CACHE_USE_LIB_STAMP": "0",
            "TILELANG_LAYOUT_AB_EVIDENCE_BOUNDARY": (
                "One rented consumer Blackwell GPU measures 18 unchanged PrimFuncs selected by the "
                "cross-architecture compile-only divergence scan. It compares register-count and io-aware "
                "with correctness, ptxas resources, eager launches, and CUDA Graph replay. Other GPU "
                "architectures, model-level workloads, and the global 1.50x goal remain outside this run."
            ),
        }
    )
    completed = run(
        [sys.executable, str(WORK_ROOT / "benchmark_layout_divergent_cases_t4.py")],
        timeout=3600,
        env=env,
        cwd=WORK_ROOT,
    )
    if not RAW_RESULT.is_file():
        raise RuntimeError("benchmark completed without a raw result")
    raw = json.loads(RAW_RESULT.read_text(encoding="utf-8"))
    return completed, raw


def write_manifest() -> None:
    entries = []
    for path in sorted(EVIDENCE_ROOT.iterdir(), key=lambda item: item.name):
        if path.is_file() and path != MANIFEST:
            entries.append(f"{sha256(path)}  {path.name}")
    MANIFEST.write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": "tilelang-layout-divergent-vast-blackwell-run-v1",
        "repository": "nya-a-cat/tilelang",
        "runner_source_sha": os.environ.get("TILELANG_RUNNER_SOURCE_SHA"),
        "runner_sha256": os.environ.get("TILELANG_RUNNER_SHA256"),
        "benchmark_source_sha": BENCHMARK_SOURCE_SHA,
        "runtime_source_sha": RUNTIME_SOURCE_SHA,
        "native_base_sha": NATIVE_BASE_SHA,
        "expected_gpu_name": EXPECTED_GPU_NAME,
        "expected_gpu_capability": list(EXPECTED_CAPABILITY),
        "container_image": os.environ.get("TILELANG_CONTAINER_IMAGE"),
        "container_image_digest": os.environ.get("TILELANG_CONTAINER_IMAGE_DIGEST"),
        "vast_offer_id": os.environ.get("TILELANG_VAST_OFFER_ID"),
        "vast_instance_id": os.environ.get("TILELANG_VAST_INSTANCE_ID"),
        "started_unix": started,
        "evidence_boundary": (
            "One rented RTX 50-series Blackwell GPU validates the same-PrimFunc backend layout A/B. "
            "B-series data-center Blackwell, other GPU architectures, model-level workloads, and the global "
            "1.50x goal remain outside this evidence."
        ),
    }
    try:
        initial_torch = verify_blackwell()
        payload["system_before"] = {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": initial_torch,
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total,compute_cap,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc": command_output(["nvcc", "--version"]),
        }
        payload["provenance"] = install_exact_runtime(initial_torch)
        completed, benchmark = execute_benchmark()
        payload["benchmark"] = benchmark
        payload["benchmark_result_sha256"] = sha256(RAW_RESULT)
        payload["benchmark_stdout"] = completed.stdout
        payload["benchmark_stderr"] = completed.stderr
        payload["status"] = "complete" if benchmark.get("status") == "complete" else "partial"
    except Exception as error:  # noqa: BLE001 - preserve partial forensic evidence
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
        if RAW_RESULT.is_file():
            payload["benchmark"] = json.loads(RAW_RESULT.read_text(encoding="utf-8"))
            payload["benchmark_result_sha256"] = sha256(RAW_RESULT)
    finally:
        payload["system_after"] = {
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,memory.used",
                    "--format=csv,noheader",
                ]
            )
        }
        payload["duration_seconds"] = time.time() - started
        LOG.write_text("\n\n".join(RUN_LOGS), encoding="utf-8")
        payload["execution_log_sha256"] = sha256(LOG)
        RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_manifest()
        print(
            "TILELANG_LAYOUT_BLACKWELL_RUN="
            + json.dumps(
                {
                    "result": str(RESULT),
                    "result_sha256": sha256(RESULT),
                    "raw_result": str(RAW_RESULT) if RAW_RESULT.is_file() else None,
                    "log": str(LOG),
                    "log_sha256": sha256(LOG),
                    "manifest": str(MANIFEST),
                    "manifest_sha256": sha256(MANIFEST),
                    "status": payload.get("status"),
                    "duration_seconds": payload["duration_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if payload.get("status") != "complete":
            print(
                "TILELANG_LAYOUT_BLACKWELL_ERROR="
                + json.dumps(
                    {
                        "error": payload.get("error"),
                        "status": payload.get("status"),
                        "traceback_tail": str(payload.get("traceback", ""))[-6000:],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    if payload.get("status") != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
