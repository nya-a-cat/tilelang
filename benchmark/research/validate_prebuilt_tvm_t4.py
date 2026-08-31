"""Validate the exact prebuilt-TVM development wheel on a free Colab T4."""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request


RELEASE_TAG = "colab-fast-4df968c3a85e723dc1870c62a0745284660bffd3"
WHEEL_NAME = "tilelang-0.1.13+cu130.git4df968c3-cp39-abi3-linux_x86_64.whl"
WHEEL_URL = (
    f"https://github.com/nya-a-cat/tilelang/releases/download/{RELEASE_TAG}/"
    "tilelang-0.1.13%2Bcu130.git4df968c3-cp39-abi3-linux_x86_64.whl"
)
WHEEL_SHA256 = "f3ddcaa79cb10a5b61f4c8ed2eb6941fb30a3f53ddb45aeff2008b6824835f82"
WHEEL_SOURCE_SHA = "4df968c3a85e723dc1870c62a0745284660bffd3"
TVM_SOURCE_SHA = "907a88c8791ccf33b9874821bc875e7abf624367"
TVM_SDK_FINGERPRINT = "97acb7c0c4c6dbcde76553d6a74d8ff03cf15d610a75e7aba30940ed029e00e5"
CACHE_DIR = f"/tmp/tilelang-prebuilt-tvm-cache-{WHEEL_SHA256[:12]}"
RESULT_PATH = Path("/content/tilelang-prebuilt-tvm-t4-validation.json")
LENGTH = 4096
THREADS = 256
CHILD_MARKER = "TILELANG_CHILD_RESULT="


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


def install_exact_wheel() -> dict[str, object]:
    wheel_path = Path("/tmp") / WHEEL_NAME
    started = time.perf_counter()
    urllib.request.urlretrieve(WHEEL_URL, wheel_path)
    actual_hash = sha256_file(wheel_path)
    if actual_hash != WHEEL_SHA256:
        raise RuntimeError(f"wheel hash mismatch: expected {WHEEL_SHA256}, got {actual_hash}")
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
    return {
        "url": WHEEL_URL,
        "filename": WHEEL_NAME,
        "sha256": actual_hash,
        "bytes": wheel_path.stat().st_size,
        "install_seconds": time.perf_counter() - started,
    }


def snapshot_tree(root: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            stat = path.stat()
            files[str(path.relative_to(root))] = {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
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
        libraries[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "ldd": run(["ldd", str(path)], capture=True),
        }
    return libraries


def nvidia_snapshot() -> str:
    try:
        return run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture=True,
        )
    except Exception as error:
        return f"unavailable: {type(error).__name__}: {error}"


def child_validation() -> None:
    os.environ["TILELANG_CACHE_DIR"] = CACHE_DIR
    import torch
    import tilelang
    import tilelang.language as T

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the T4 validation child")

    @T.prim_func
    def add_one(A: T.Tensor((LENGTH,), T.float32), B: T.Tensor((LENGTH,), T.float32)):
        with T.Kernel(T.ceildiv(LENGTH, THREADS), threads=THREADS) as (bx,):
            for tx in T.Parallel(THREADS):
                index = bx * THREADS + tx
                if index < LENGTH:
                    B[index] = A[index] + 1.0

    program = add_one.with_attr("global_symbol", "validate_prebuilt_tvm_add_one")
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
    print(CHILD_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


def run_child(label: str) -> tuple[dict[str, object], str]:
    child_env = os.environ.copy()
    child_env["TILELANG_CACHE_DIR"] = CACHE_DIR
    output = run([sys.executable, str(Path(__file__).resolve()), "--child"], capture=True, env=child_env)
    marker_lines = [line for line in output.splitlines() if line.startswith(CHILD_MARKER)]
    if len(marker_lines) != 1:
        raise RuntimeError(f"{label} child did not emit exactly one result marker:\n{output}")
    return json.loads(marker_lines[0][len(CHILD_MARKER) :]), output


def package_versions() -> dict[str, str]:
    names = ["tilelang", "apache-tvm-ffi", "torch-c-dlpack-ext", "z3-solver", "torch"]
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    os.environ["TILELANG_CACHE_DIR"] = CACHE_DIR
    cache_root = Path(CACHE_DIR)
    if cache_root.exists() and any(cache_root.rglob("*")):
        raise RuntimeError(f"fresh Colab VM unexpectedly has a populated validation cache: {CACHE_DIR}")

    started = time.perf_counter()
    wheel = install_exact_wheel()

    import torch
    import tilelang

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if tilelang.__version__ != "0.1.13+cu130.git4df968c3":
        raise RuntimeError(f"unexpected TileLang version: {tilelang.__version__}")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cache_dir": CACHE_DIR,
        "nvidia_smi_start": nvidia_snapshot(),
        "native_libraries": native_libraries(),
    }

    empty_snapshot = snapshot_tree(cache_root)
    first, first_log = run_child("first")
    first_snapshot = snapshot_tree(cache_root)
    second, second_log = run_child("second")
    second_snapshot = snapshot_tree(cache_root)

    if empty_snapshot["file_count"] != 0:
        raise RuntimeError("validation cache was not empty before the first compile")
    if first_snapshot["file_count"] == 0:
        raise RuntimeError("the first compile did not persist a kernel cache entry")
    if first_snapshot != second_snapshot:
        raise RuntimeError("the second child modified the persisted kernel cache")
    if "Found kernel in disk cache" in first_log:
        raise RuntimeError("the first independent child unexpectedly reported a disk-cache hit")
    if "Found kernel in disk cache" not in second_log:
        raise RuntimeError("the second independent child did not report a disk-cache hit")
    if first["cache_key"] != second["cache_key"] or first["kernel_source_sha256"] != second["kernel_source_sha256"]:
        raise RuntimeError("fresh compile and disk-cache load produced different kernel identities")

    environment["nvidia_smi_end"] = nvidia_snapshot()
    payload = {
        "schema": "tilelang-prebuilt-tvm-t4-validation-v1",
        "status": "success",
        "created_unix": time.time(),
        "repository": "nya-a-cat/tilelang",
        "wheel_source_sha": WHEEL_SOURCE_SHA,
        "tvm_source_sha": TVM_SOURCE_SHA,
        "tvm_sdk_fingerprint": TVM_SDK_FINGERPRINT,
        "release_tag": RELEASE_TAG,
        "wheel": wheel,
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
            "Validates exact-wheel import, native dynamic linking, sm_75 JIT compilation, exact output, "
            "and a cross-process disk-cache hit on one free Colab T4. It is not a throughput benchmark "
            "or multi-architecture acceptance result."
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload_sha256 = hashlib.sha256(serialized).hexdigest()
    compressed = gzip.compress(serialized, compresslevel=9)
    RESULT_PATH.write_bytes(serialized)
    RESULT_PATH.with_suffix(".json.gz").write_bytes(compressed)
    print(f"RESULT_JSON_SHA256={payload_sha256}")
    print(f"RESULT_JSON_BYTES={len(serialized)}")
    print(f"RESULT_GZIP_BASE64={base64.b64encode(compressed).decode()}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        child_validation()
    elif sys.argv[1:]:
        raise SystemExit(f"unexpected arguments: {sys.argv[1:]}")
    else:
        main()
