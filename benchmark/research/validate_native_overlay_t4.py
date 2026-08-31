"""Validate a verified TileLang native overlay on a fresh free Colab T4."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile


REPOSITORY = "nya-a-cat/tilelang"
OVERLAY_SOURCE_SHA = "297cdb52b8063ee2dbae0a74da3db4c12af9db03"
OVERLAY_ARTIFACT_ID = 9770505476
OVERLAY_ARTIFACT_SHA256 = "1853279fbb04e86165c23342f5110332f2283bce5e339fc7d3d4426be5f441ad"
BASE_SOURCE_SHA = "4df968c3a85e723dc1870c62a0745284660bffd3"
BASE_RELEASE_TAG = f"colab-fast-{BASE_SOURCE_SHA}"
BASE_WHEEL_NAME = "tilelang-0.1.13+cu130.git4df968c3-cp39-abi3-linux_x86_64.whl"
BASE_WHEEL_URL = (
    f"https://github.com/{REPOSITORY}/releases/download/{BASE_RELEASE_TAG}/"
    "tilelang-0.1.13%2Bcu130.git4df968c3-cp39-abi3-linux_x86_64.whl"
)
BASE_WHEEL_SHA256 = "f3ddcaa79cb10a5b61f4c8ed2eb6941fb30a3f53ddb45aeff2008b6824835f82"
EXPECTED_VERSION = "0.1.13+cu130.git4df968c3"
ARTIFACT_URL_ENV = "TILELANG_NATIVE_OVERLAY_URL"
CACHE_DIR = f"/tmp/tilelang-native-overlay-cache-{OVERLAY_SOURCE_SHA[:12]}"
RESULT_MARKER = "TILELANG_NATIVE_OVERLAY_RESULT="
CHILD_MARKER = "TILELANG_NATIVE_OVERLAY_CHILD="
LENGTH = 4096
THREADS = 256
CHILD_SOURCE = r'''
import hashlib
import json
import os
import time

import torch
import tilelang
import tilelang.language as T

LENGTH = 4096
THREADS = 256

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the native-overlay validation child")


@T.prim_func
def add_one(A: T.Tensor((LENGTH,), T.float32), B: T.Tensor((LENGTH,), T.float32)):
    with T.Kernel(T.ceildiv(LENGTH, THREADS), threads=THREADS) as (bx,):
        for tx in T.Parallel(THREADS):
            index = bx * THREADS + tx
            if index < LENGTH:
                B[index] = A[index] + 1.0


program = add_one.with_attr("global_symbol", "validate_native_overlay_add_one")
started = time.perf_counter()
kernel = tilelang.compile(
    program,
    out_idx=None,
    target="cuda",
    execution_backend="tvm_ffi",
    verbose=True,
)
compile_seconds = time.perf_counter() - started

x = torch.arange(LENGTH, device="cuda", dtype=torch.float32)
output = torch.empty_like(x)
for _ in range(100):
    kernel(x, output)
torch.cuda.synchronize()
torch.testing.assert_close(output, x + 1.0, rtol=0.0, atol=0.0)

result = {
    "compile_seconds": compile_seconds,
    "cache_key": getattr(kernel, "_tilelang_cache_key", None),
    "cache_path": getattr(kernel, "_tilelang_cache_path", None),
    "kernel_source_sha256": hashlib.sha256(kernel.kernel_source.encode()).hexdigest(),
    "launches": 100,
    "correctness": "exact",
    "output_sum": float(output.sum().item()),
    "tilelang_version": tilelang.__version__,
    "torch_version": torch.__version__,
    "device_name": torch.cuda.get_device_name(0),
    "device_capability": list(torch.cuda.get_device_capability(0)),
}
print("TILELANG_NATIVE_OVERLAY_CHILD=" + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
'''


def run(command: list[str], *, capture: bool = False, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path, expected_hash: str) -> dict[str, object]:
    started = time.perf_counter()
    urllib.request.urlretrieve(url, output)
    actual_hash = sha256_file(output)
    if actual_hash != expected_hash:
        raise RuntimeError(f"download hash mismatch for {output.name}: {actual_hash}")
    return {
        "filename": output.name,
        "sha256": actual_hash,
        "bytes": output.stat().st_size,
        "download_seconds": time.perf_counter() - started,
    }


def extract_artifact(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or any(part in ("", ".", "..") for part in member.parts):
                raise RuntimeError(f"unsafe artifact path: {info.filename}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"artifact contains a symbolic link: {info.filename}")
            output = destination.joinpath(*member.parts)
            if destination not in output.resolve().parents and output.resolve() != destination:
                raise RuntimeError(f"artifact member escapes extraction root: {info.filename}")
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(bundle.read(info))
            output.chmod((mode & 0o777) or 0o644)


def unique_file(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {len(matches)}")
    return matches[0]


def verify_checksums(checksum_path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        filename = filename.lstrip("*")
        if filename.startswith("./"):
            filename = filename[2:]
        if filename in expected or "/" in filename or filename in ("", ".", ".."):
            raise RuntimeError(f"invalid checksum member: {filename!r}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"invalid checksum digest for {filename}")
        expected[filename] = digest

    actual_files = {
        path.name: path
        for path in checksum_path.parent.iterdir()
        if path.is_file() and path.name != checksum_path.name
    }
    if set(actual_files) != set(expected):
        raise RuntimeError("artifact checksum inventory mismatch")
    for name, path in actual_files.items():
        if sha256_file(path) != expected[name]:
            raise RuntimeError(f"artifact checksum mismatch: {name}")
    return expected


def prepare_overlay() -> dict[str, object]:
    signed_url = os.environ.pop(ARTIFACT_URL_ENV, "")
    if not signed_url:
        raise RuntimeError(f"missing ephemeral artifact URL in {ARTIFACT_URL_ENV}")
    archive = Path("/tmp/tilelang-native-overlay-actions-artifact.zip")
    record = download(signed_url, archive, OVERLAY_ARTIFACT_SHA256)
    del signed_url

    extraction_root = Path("/tmp/tilelang-native-overlay-actions-artifact")
    extract_artifact(archive, extraction_root)
    checksum_path = unique_file(extraction_root, "SHA256SUMS")
    checksums = verify_checksums(checksum_path)
    manifest_path = unique_file(extraction_root, "native-overlay-manifest.json")
    overlay_path = unique_file(extraction_root, "tilelang-native-overlay.tar.gz")
    installer_path = unique_file(extraction_root, "apply_colab_native_overlay.sh")
    manifest = json.loads(manifest_path.read_text())

    expected = {
        "schema": "tilelang-colab-native-overlay-v1",
        "repository": REPOSITORY,
        "source_sha": OVERLAY_SOURCE_SHA,
        "native_base_sha": BASE_SOURCE_SHA,
        "native_base_tag": BASE_RELEASE_TAG,
        "base_distribution_version": EXPECTED_VERSION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"unexpected overlay manifest {key}: {manifest.get(key)!r}")
    base_wheel = manifest.get("base_wheel")
    if not isinstance(base_wheel, dict):
        raise RuntimeError("overlay manifest has no base_wheel object")
    if base_wheel.get("name") != BASE_WHEEL_NAME or base_wheel.get("sha256") != BASE_WHEEL_SHA256:
        raise RuntimeError("overlay manifest names a different base wheel")
    if sha256_file(overlay_path) != manifest.get("overlay_sha256"):
        raise RuntimeError("overlay archive and manifest disagree")
    installer_path.chmod(0o755)
    return {
        "download": record,
        "artifact_id": OVERLAY_ARTIFACT_ID,
        "artifact_digest": f"sha256:{OVERLAY_ARTIFACT_SHA256}",
        "checksums": checksums,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "overlay_path": overlay_path,
        "installer_path": installer_path,
    }


def install_base_wheel() -> dict[str, object]:
    wheel_path = Path("/tmp") / BASE_WHEEL_NAME
    record = download(BASE_WHEEL_URL, wheel_path, BASE_WHEEL_SHA256)
    started = time.perf_counter()
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "apache-tvm-ffi==0.1.12",
            "torch-c-dlpack-ext==0.1.5",
            "z3-solver==4.15.4.0",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheel_path)])
    record["install_seconds"] = time.perf_counter() - started
    record["url"] = BASE_WHEEL_URL
    return record


def apply_overlay(prepared: dict[str, object]) -> str:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    return run(
        [
            str(prepared["installer_path"]),
            str(prepared["overlay_path"]),
            str(prepared["manifest_path"]),
        ],
        capture=True,
        env=env,
    )


def snapshot_tree(root: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            file_stat = path.stat()
            files[str(path.relative_to(root))] = {
                "bytes": file_stat.st_size,
                "mtime_ns": file_stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files.values()),
        "snapshot_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def native_libraries() -> dict[str, object]:
    from tilelang.libinfo import find_lib_path

    libraries: dict[str, object] = {}
    for name in ("tilelang", "tvm_runtime", "tvm_compiler"):
        path = Path(find_lib_path(name)).resolve()
        ldd = run(["ldd", str(path)], capture=True)
        if "not found" in ldd:
            raise RuntimeError(f"unresolved dependency in {path}:\n{ldd}")
        libraries[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "ldd": ldd,
        }
    return libraries


def nvidia_snapshot() -> str:
    return run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    )


def run_child(label: str) -> tuple[dict[str, object], str]:
    child_path = Path("/tmp/tilelang-native-overlay-validation-child.py")
    if child_path.exists() and child_path.read_text() != CHILD_SOURCE:
        raise RuntimeError(f"unexpected existing child source: {child_path}")
    child_path.write_text(CHILD_SOURCE)
    child_env = os.environ.copy()
    child_env["TILELANG_CACHE_DIR"] = CACHE_DIR
    output = run([sys.executable, str(child_path)], capture=True, env=child_env)
    marker_lines = [line for line in output.splitlines() if line.startswith(CHILD_MARKER)]
    if len(marker_lines) != 1:
        raise RuntimeError(f"{label} child did not emit one result marker:\n{output}")
    return json.loads(marker_lines[0][len(CHILD_MARKER) :]), output


def package_versions() -> dict[str, str]:
    names = ["tilelang", "apache-tvm-ffi", "torch-c-dlpack-ext", "z3-solver", "torch"]
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    os.environ["TILELANG_CACHE_DIR"] = CACHE_DIR
    cache_root = Path(CACHE_DIR)
    if cache_root.exists() and any(cache_root.rglob("*")):
        raise RuntimeError(f"fresh Colab VM has a populated validation cache: {CACHE_DIR}")

    started = time.perf_counter()
    prepared = prepare_overlay()
    base_wheel = install_base_wheel()
    installer_log = apply_overlay(prepared)

    import torch
    import tilelang

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    if tilelang.__version__ != EXPECTED_VERSION:
        raise RuntimeError(f"unexpected TileLang version: {tilelang.__version__}")

    distribution = importlib.metadata.distribution("tilelang")
    package = Path(distribution.locate_file("tilelang")).resolve()
    identity = json.loads((package / "_python_overlay_identity.json").read_text())
    if identity.get("source_sha") != OVERLAY_SOURCE_SHA:
        raise RuntimeError("installed runtime identity does not match overlay source")
    libraries = native_libraries()
    manifest = prepared["manifest"]
    if libraries["tilelang"]["sha256"] != manifest["patched_libtilelang_sha256"]:
        raise RuntimeError("installed libtilelang.so does not match overlay manifest")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cache_dir": CACHE_DIR,
        "nvidia_smi_start": nvidia_snapshot(),
        "native_libraries": libraries,
        "runtime_identity": identity,
    }

    empty_snapshot = snapshot_tree(cache_root)
    first, first_log = run_child("first")
    first_snapshot = snapshot_tree(cache_root)
    second, second_log = run_child("second")
    second_snapshot = snapshot_tree(cache_root)

    if empty_snapshot["file_count"] != 0:
        raise RuntimeError("validation cache was not empty before first compile")
    if first_snapshot["file_count"] == 0:
        raise RuntimeError("first compile did not persist a kernel cache entry")
    if first_snapshot != second_snapshot:
        raise RuntimeError("second child modified the persisted kernel cache")
    if "Found kernel in disk cache" in first_log:
        raise RuntimeError("first child unexpectedly reported a disk-cache hit")
    if "Found kernel in disk cache" not in second_log:
        raise RuntimeError("second child did not report a disk-cache hit")
    if first["cache_key"] != second["cache_key"]:
        raise RuntimeError("fresh compile and cache load used different cache keys")
    if first["kernel_source_sha256"] != second["kernel_source_sha256"]:
        raise RuntimeError("fresh compile and cache load produced different kernel sources")

    environment["nvidia_smi_end"] = nvidia_snapshot()
    payload = {
        "schema": "tilelang-native-overlay-t4-validation-v1",
        "status": "success",
        "created_unix": time.time(),
        "repository": REPOSITORY,
        "overlay_source_sha": OVERLAY_SOURCE_SHA,
        "native_base_sha": BASE_SOURCE_SHA,
        "artifact": {
            "id": prepared["artifact_id"],
            "digest": prepared["artifact_digest"],
            "download": prepared["download"],
            "checksums": prepared["checksums"],
            "manifest": manifest,
        },
        "base_wheel": base_wheel,
        "installer_log": installer_log,
        "environment": environment,
        "validation": {
            "program": "float32 add-one, length=4096, 256 threads",
            "execution_backend": "tvm_ffi",
            "target": "cuda (runtime resolved T4 sm_75)",
            "first_process": first,
            "second_process": second,
            "disk_cache_hit_log_evidence": {
                "first": "Found kernel in disk cache" in first_log,
                "second": "Found kernel in disk cache" in second_log,
            },
            "cache_before": empty_snapshot,
            "cache_after_first": first_snapshot,
            "cache_after_second": second_snapshot,
        },
        "total_seconds": time.perf_counter() - started,
        "evidence_boundary": (
            "Validates the exact Actions artifact, base-wheel installation, atomic overlay installer, "
            "native dynamic linking, sm_75 JIT compilation, exact output, and independent-process "
            "disk-cache reuse on one free Colab T4. This is no throughput or multi-architecture claim."
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    print(f"RESULT_JSON_SHA256={hashlib.sha256(serialized).hexdigest()}")
    print(f"RESULT_JSON_BYTES={len(serialized)}")
    print(RESULT_MARKER + serialized.decode(), flush=True)


if __name__ == "__main__":
    main()
