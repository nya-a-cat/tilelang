"""Validate and benchmark JITKernel.bind_outputs on a fresh free Colab T4.

The runner installs one exact native wheel, applies a checksummed pure-Python
Actions overlay, and compares the existing callee-allocated API with a bound
caller-owned output across several operator families.  The signed Actions
artifact URL is supplied through the environment and removed before any result
is serialized.
"""

from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import statistics
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.request
import zipfile


REPOSITORY = "nya-a-cat/tilelang"
BASE_SOURCE_SHA = "4df968c3a85e723dc1870c62a0745284660bffd3"
BASE_RELEASE_TAG = f"colab-fast-{BASE_SOURCE_SHA}"
BASE_WHEEL_NAME = "tilelang-0.1.13+cu130.git4df968c3-cp39-abi3-linux_x86_64.whl"
BASE_WHEEL_SHA256 = "f3ddcaa79cb10a5b61f4c8ed2eb6941fb30a3f53ddb45aeff2008b6824835f82"
EXPECTED_VERSION = "0.1.13+cu130.git4df968c3"
ARTIFACT_URL_ENV = "TILELANG_PYTHON_OVERLAY_URL"
ARTIFACT_SHA256_ENV = "TILELANG_PYTHON_OVERLAY_SHA256"
ARTIFACT_ID_ENV = "TILELANG_PYTHON_OVERLAY_ARTIFACT_ID"
SOURCE_SHA_ENV = "TILELANG_PYTHON_OVERLAY_SOURCE_SHA"
CASE_FILTER_ENV = "TILELANG_BOUND_OUTPUT_CASES"
RESULT_MARKER = "TILELANG_BOUND_OUTPUTS_RESULT="
RESULT_PATH = Path("/content/tilelang-bound-outputs-t4.json")
CYCLES = 25
MIN_BATCH_SECONDS = 0.025
MAX_BATCH_ITERS = 8192
MAX_TRANSIENT_OUTPUT_BYTES = 256 << 20
WARMUP_SECONDS = 1.0

TORCH: Any = None
TILELANG: Any = None
TL: Any = None


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: str
    size_class: str
    program: Any
    inputs: tuple[Any, ...]
    output: Any
    reference: Any
    atol: float
    rtol: float
    logical_work_items: int


def run(command: list[str], *, capture: bool = False, env: dict[str, str] | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            env=env,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )
    except subprocess.CalledProcessError as error:
        if capture and error.stdout:
            raise RuntimeError(f"command failed with exit code {error.returncode}:\n{error.stdout}") from error
        raise
    return completed.stdout.strip() if capture else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sha256(value: str, label: str) -> str:
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"invalid {label} SHA-256: {value!r}")
    return digest


def download(url: str, output: Path, expected_hash: str) -> dict[str, Any]:
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
            resolved = output.resolve()
            if destination not in resolved.parents and resolved != destination:
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
        if filename.startswith("overlay/"):
            filename = filename[len("overlay/") :]
        if filename in expected or "/" in filename or filename in ("", ".", ".."):
            raise RuntimeError(f"invalid checksum member: {filename!r}")
        expected[filename] = normalize_sha256(digest, filename)

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


def prepare_overlay() -> dict[str, Any]:
    signed_url = os.environ.pop(ARTIFACT_URL_ENV, "")
    artifact_hash = normalize_sha256(os.environ.pop(ARTIFACT_SHA256_ENV, ""), "artifact")
    source_sha = os.environ.get(SOURCE_SHA_ENV, "")
    artifact_id = os.environ.get(ARTIFACT_ID_ENV, "")
    if not signed_url or len(source_sha) != 40 or not artifact_id.isdigit():
        raise RuntimeError("missing exact Actions overlay provenance environment")

    archive = Path("/tmp/tilelang-python-overlay-actions-artifact.zip")
    record = download(signed_url, archive, artifact_hash)
    del signed_url
    extraction_root = Path("/tmp/tilelang-python-overlay-actions-artifact")
    extraction_root.mkdir(parents=True, exist_ok=True)
    extract_artifact(archive, extraction_root)

    checksum_path = unique_file(extraction_root, "SHA256SUMS")
    checksums = verify_checksums(checksum_path)
    manifest_path = unique_file(extraction_root, "overlay-manifest.json")
    overlay_path = unique_file(extraction_root, "tilelang-python-overlay.tar.gz")
    installer_path = unique_file(extraction_root, "apply_colab_python_overlay.sh")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema": "tilelang-colab-python-overlay-v1",
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "native_base_sha": BASE_SOURCE_SHA,
        "native_base_tag": BASE_RELEASE_TAG,
        "compatibility": "pure-python-source-overlay",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"unexpected overlay manifest {key}: {manifest.get(key)!r}")
    if manifest.get("overlay_sha256") != sha256_file(overlay_path):
        raise RuntimeError("overlay archive and manifest disagree")

    base_wheel = manifest.get("base_wheel")
    if not isinstance(base_wheel, dict):
        raise RuntimeError("overlay manifest has no base_wheel object")
    if base_wheel.get("name") != BASE_WHEEL_NAME:
        raise RuntimeError(f"unexpected base wheel name: {base_wheel.get('name')!r}")
    if normalize_sha256(str(base_wheel.get("digest", "")), "base wheel") != BASE_WHEEL_SHA256:
        raise RuntimeError("overlay manifest identifies a different native wheel")
    expected_url_prefix = f"https://github.com/{REPOSITORY}/releases/download/{BASE_RELEASE_TAG}/"
    if not str(base_wheel.get("url", "")).startswith(expected_url_prefix):
        raise RuntimeError("overlay manifest names an unexpected wheel URL")
    installer_path.chmod(0o755)
    return {
        "source_sha": source_sha,
        "artifact_id": int(artifact_id),
        "artifact_digest": f"sha256:{artifact_hash}",
        "download": record,
        "checksums": checksums,
        "manifest": manifest,
        "overlay_path": overlay_path,
        "installer_path": installer_path,
        "base_wheel": base_wheel,
    }


def install_base_wheel(base_wheel: dict[str, Any]) -> dict[str, Any]:
    wheel_path = Path("/tmp") / BASE_WHEEL_NAME
    record = download(str(base_wheel["url"]), wheel_path, BASE_WHEEL_SHA256)
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
    record["url"] = base_wheel["url"]
    return record


def apply_overlay(prepared: dict[str, Any]) -> str:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    return run(
        [str(prepared["installer_path"]), str(prepared["overlay_path"])],
        capture=True,
        env=env,
    )


def native_libraries() -> dict[str, dict[str, Any]]:
    from tilelang.libinfo import find_lib_path

    libraries: dict[str, dict[str, Any]] = {}
    for name in ("tilelang", "tvm_runtime", "tvm_compiler"):
        library = Path(find_lib_path(name)).resolve()
        ldd = run(["ldd", str(library)], capture=True)
        if "not found" in ldd:
            raise RuntimeError(f"unresolved native dependency in {library}")
        libraries[name] = {
            "path": str(library),
            "bytes": library.stat().st_size,
            "sha256": sha256_file(library),
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


def with_output_attr(program: Any, symbol: str) -> Any:
    return program.with_attr("global_symbol", symbol).with_attr("tilelang_out_idx", [-1])


def make_add_program(length: int, symbol: str) -> Any:
    threads = 256

    @TL.prim_func
    def main(A: TL.Tensor((length,), TL.float32), B: TL.Tensor((length,), TL.float32)):
        with TL.Kernel(TL.ceildiv(length, threads), threads=threads) as (bx,):
            for tx in TL.Parallel(threads):
                index = bx * threads + tx
                if index < length:
                    B[index] = A[index] + 1.0

    return with_output_attr(main, symbol)


def make_reduce_program(rows: int, width: int, symbol: str) -> Any:
    threads = 128

    @TL.prim_func
    def main(A: TL.Tensor((rows, width), TL.float32), B: TL.Tensor((rows,), TL.float32)):
        with TL.Kernel(rows, threads=threads) as bx:
            values = TL.alloc_fragment((width,), TL.float32)
            total = TL.alloc_fragment((1,), TL.float32)
            TL.copy(A[bx, 0], values)
            TL.reduce_sum(values, total, dim=0)
            B[bx] = total[0]

    return with_output_attr(main, symbol)


def make_rmsnorm_program(rows: int, width: int, symbol: str) -> Any:
    threads = 128

    @TL.prim_func
    def main(A: TL.Tensor((rows, width), TL.float32), B: TL.Tensor((rows, width), TL.float32)):
        with TL.Kernel(rows, threads=threads) as bx:
            values = TL.alloc_fragment((width,), TL.float32)
            squares = TL.alloc_fragment((width,), TL.float32)
            total = TL.alloc_fragment((1,), TL.float32)
            TL.copy(A[bx, 0], values)
            for index in TL.Parallel(width):
                squares[index] = values[index] * values[index]
            TL.reduce_sum(squares, total, dim=0)
            scale = TL.rsqrt(total[0] / width + 1e-6)
            for index in TL.Parallel(width):
                values[index] *= scale
            TL.copy(values, B[bx, 0])

    return with_output_attr(main, symbol)


def make_softmax_program(rows: int, width: int, symbol: str) -> Any:
    threads = 128

    @TL.prim_func
    def main(A: TL.Tensor((rows, width), TL.float32), B: TL.Tensor((rows, width), TL.float32)):
        with TL.Kernel(rows, threads=threads) as bx:
            values = TL.alloc_fragment((width,), TL.float32)
            maximum = TL.alloc_fragment((1,), TL.float32)
            total = TL.alloc_fragment((1,), TL.float32)
            TL.copy(A[bx, 0], values)
            TL.reduce_max(values, maximum, dim=0)
            for index in TL.Parallel(width):
                values[index] = TL.exp(values[index] - maximum[0])
            TL.reduce_sum(values, total, dim=0)
            for index in TL.Parallel(width):
                values[index] /= total[0]
            TL.copy(values, B[bx, 0])

    return with_output_attr(main, symbol)


def make_gemm_program(symbol: str) -> Any:
    rows, columns, reduction = 64, 64, 32

    @TL.prim_func
    def main(
        A: TL.Tensor((rows, reduction), TL.float32),
        B: TL.Tensor((reduction, columns), TL.float32),
        C: TL.Tensor((rows, columns), TL.float32),
    ):
        with TL.Kernel(1, threads=128):
            a_shared = TL.alloc_shared((rows, reduction), TL.float32)
            b_shared = TL.alloc_shared((reduction, columns), TL.float32)
            c_local = TL.alloc_fragment((rows, columns), TL.float32)
            TL.copy(A, a_shared)
            TL.copy(B, b_shared)
            TL.clear(c_local)
            TL.gemm(a_shared, b_shared, c_local, policy=TL.GemmWarpPolicy.FullRow)
            TL.copy(c_local, C)

    return with_output_attr(main, symbol)


def build_cases() -> list[BenchmarkCase]:
    device = TORCH.device("cuda")
    cases: list[BenchmarkCase] = []

    for length, size_class in ((4096, "small"), (1 << 20, "large_control")):
        source = TORCH.randn(length, device=device, dtype=TORCH.float32)
        output = TORCH.empty_like(source)
        cases.append(
            BenchmarkCase(
                name=f"add_f32_{length}",
                family="elementwise",
                size_class=size_class,
                program=make_add_program(length, f"bound_output_add_{length}"),
                inputs=(source,),
                output=output,
                reference=source + 1.0,
                atol=0.0,
                rtol=0.0,
                logical_work_items=length,
            )
        )

    rows, width = 8, 1024
    source = TORCH.randn(rows, width, device=device, dtype=TORCH.float32)
    cases.append(
        BenchmarkCase(
            name="reduce_sum_f32_8x1024",
            family="reduction",
            size_class="small",
            program=make_reduce_program(rows, width, "bound_output_reduce_sum_8x1024"),
            inputs=(source,),
            output=TORCH.empty(rows, device=device, dtype=TORCH.float32),
            reference=source.sum(dim=1),
            atol=2e-3,
            rtol=2e-3,
            logical_work_items=rows * width,
        )
    )
    cases.append(
        BenchmarkCase(
            name="rmsnorm_f32_8x1024",
            family="rmsnorm",
            size_class="small",
            program=make_rmsnorm_program(rows, width, "bound_output_rmsnorm_8x1024"),
            inputs=(source,),
            output=TORCH.empty_like(source),
            reference=source * TORCH.rsqrt(source.square().mean(dim=1, keepdim=True) + 1e-6),
            atol=2e-3,
            rtol=2e-3,
            logical_work_items=rows * width,
        )
    )
    cases.append(
        BenchmarkCase(
            name="softmax_f32_8x1024",
            family="softmax",
            size_class="small",
            program=make_softmax_program(rows, width, "bound_output_softmax_8x1024"),
            inputs=(source,),
            output=TORCH.empty_like(source),
            reference=TORCH.softmax(source, dim=1),
            atol=2e-4,
            rtol=2e-3,
            logical_work_items=rows * width,
        )
    )

    rows, columns, reduction = 64, 64, 32
    lhs = TORCH.randn(rows, reduction, device=device, dtype=TORCH.float32)
    rhs = TORCH.randn(reduction, columns, device=device, dtype=TORCH.float32)
    cases.append(
        BenchmarkCase(
            name="gemm_f32_64x64x32_sm75",
            family="gemm",
            size_class="small",
            program=make_gemm_program("bound_output_gemm_f32_64x64x32"),
            inputs=(lhs, rhs),
            output=TORCH.empty((rows, columns), device=device, dtype=TORCH.float32),
            reference=lhs @ rhs,
            atol=1e-4,
            rtol=1e-4,
            logical_work_items=rows * columns * reduction,
        )
    )
    return cases


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [float(sample["wall_us"]) for sample in samples]
    gpus = [float(sample["gpu_us"]) for sample in samples]
    return {
        "samples": len(samples),
        "wall_p50_us": statistics.median(walls),
        "wall_p90_us": percentile(walls, 0.90),
        "wall_p99_us": percentile(walls, 0.99),
        "gpu_p50_us": statistics.median(gpus),
        "gpu_p90_us": percentile(gpus, 0.90),
        "gpu_p99_us": percentile(gpus, 0.99),
    }


def warm_for(call: Callable[[], Any], seconds: float) -> int:
    calls = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(128):
            call()
            calls += 1
        TORCH.cuda.synchronize()
    return calls


def measure_batch(call: Callable[[], Any], iterations: int) -> tuple[float, float]:
    TORCH.cuda.synchronize()
    start_event = TORCH.cuda.Event(enable_timing=True)
    end_event = TORCH.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    for _ in range(iterations):
        call()
    end_event.record()
    TORCH.cuda.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1e6 / iterations
    gpu_us = start_event.elapsed_time(end_event) * 1e3 / iterations
    return wall_us, gpu_us


def calibrate(call: Callable[[], Any], iteration_cap: int) -> int:
    iterations = min(32, iteration_cap)
    while iterations < iteration_cap:
        wall_us, _ = measure_batch(call, iterations)
        if wall_us * iterations >= MIN_BATCH_SECONDS * 1e6:
            break
        iterations = min(iterations * 2, iteration_cap)
    return iterations


def allocation_probe(call: Callable[[], Any]) -> dict[str, int]:
    TORCH.cuda.synchronize()
    before = TORCH.cuda.memory_stats()
    pointers: set[int] = set()
    held: list[Any] = []
    for _ in range(100):
        result = call()
        pointers.add(int(result.data_ptr()))
        held.append(result)
        if len(held) > 8:
            held.pop(0)
    TORCH.cuda.synchronize()
    after = TORCH.cuda.memory_stats()
    return {
        "calls": 100,
        "unique_output_addresses": len(pointers),
        "allocation_requests": int(after["allocation.all.allocated"] - before["allocation.all.allocated"]),
        "allocated_bytes_requests": int(
            after["allocated_bytes.all.allocated"] - before["allocated_bytes.all.allocated"]
        ),
    }


def capture_graph(call: Callable[[], Any]) -> tuple[Any, Any]:
    for _ in range(5):
        captured_output = call()
    TORCH.cuda.synchronize()
    graph = TORCH.cuda.CUDAGraph()
    with TORCH.cuda.graph(graph):
        captured_output = call()
    TORCH.cuda.synchronize()
    return graph, captured_output


def measure_graph(graph: Any, iterations: int) -> dict[str, Any]:
    for _ in range(10):
        graph.replay()
    TORCH.cuda.synchronize()
    start_event = TORCH.cuda.Event(enable_timing=True)
    end_event = TORCH.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    end_event.record()
    TORCH.cuda.synchronize()
    return {
        "iterations": iterations,
        "wall_us": (time.perf_counter() - wall_start) * 1e6 / iterations,
        "gpu_us": start_event.elapsed_time(end_event) * 1e3 / iterations,
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError(f"geometric mean requires positive values: {values}")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_size_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_family[str(result["family"])].append(result)
        by_size_class[str(result["size_class"])].append(result)

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(group),
            "wall_p50_speedup_gmean": geometric_mean([item["speedup"]["wall_p50"] for item in group]),
            "gpu_p50_speedup_gmean": geometric_mean([item["speedup"]["gpu_p50"] for item in group]),
            "minimum_wall_p50_speedup": min(item["speedup"]["wall_p50"] for item in group),
        }

    family = {name: aggregate(group) for name, group in sorted(by_family.items())}
    size_class = {name: aggregate(group) for name, group in sorted(by_size_class.items())}
    return {
        "all_cases": aggregate(results),
        "by_family": family,
        "by_size_class": size_class,
        "family_balanced_wall_p50_speedup_gmean": geometric_mean(
            [value["wall_p50_speedup_gmean"] for value in family.values()]
        ),
        "family_balanced_gpu_p50_speedup_gmean": geometric_mean(
            [value["gpu_p50_speedup_gmean"] for value in family.values()]
        ),
    }


def benchmark_case(case: BenchmarkCase) -> dict[str, Any]:
    print(f"BOUND_OUTPUT case={case.name}: compiling callee ABI", flush=True)
    started = time.perf_counter()
    kernel = TILELANG.compile(
        case.program,
        target="cuda",
        execution_backend="tvm_ffi",
        verbose=False,
    )
    callee_compile_seconds = time.perf_counter() - started
    if kernel.out_idx != [len(kernel.params) - 1]:
        raise RuntimeError(f"unexpected normalized output indices for {case.name}: {kernel.out_idx}")

    last_callee: list[Any] = [None]

    def call_callee() -> Any:
        last_callee[0] = kernel(*case.inputs)
        return last_callee[0]

    callee_result = call_callee()
    TORCH.cuda.synchronize()
    TORCH.testing.assert_close(callee_result, case.reference, atol=case.atol, rtol=case.rtol)

    print(f"BOUND_OUTPUT case={case.name}: compiling caller ABI", flush=True)
    started = time.perf_counter()
    bound = kernel.bind_outputs(case.output)
    bind_compile_seconds = time.perf_counter() - started
    if kernel.bind_outputs(case.output) is not bound:
        raise RuntimeError("binding the same output did not reuse the callable")
    bound_result = bound(*case.inputs)
    if bound_result is not case.output:
        raise RuntimeError("bound callable did not return the caller-owned output")
    if kernel.call_into(*case.inputs, out=case.output) is not case.output:
        raise RuntimeError("call_into did not return the caller-owned output")
    TORCH.cuda.synchronize()
    TORCH.testing.assert_close(case.output, case.reference, atol=case.atol, rtol=case.rtol)

    companion = kernel._caller_allocated_kernel
    if companion is None or companion.out_idx:
        raise RuntimeError("caller-allocated companion was not prepared with a full-parameter ABI")
    source_attrs = kernel.prim_func.attrs
    companion_attrs = companion.prim_func.attrs
    if source_attrs is None or "tilelang_out_idx" not in source_attrs:
        raise RuntimeError("source PrimFunc lost its callee-allocated output attribute")
    if companion_attrs is not None and "tilelang_out_idx" in companion_attrs:
        raise RuntimeError("caller-allocated companion retained the callee output attribute")

    calls = {"callee": call_callee, "bound": lambda: bound(*case.inputs)}
    warmups = {label: warm_for(call, WARMUP_SECONDS) for label, call in calls.items()}
    output_bytes = int(case.output.numel() * case.output.element_size())
    iteration_cap = min(MAX_BATCH_ITERS, max(32, MAX_TRANSIENT_OUTPUT_BYTES // max(output_bytes, 1)))
    iterations = max(calibrate(call_callee, iteration_cap), calibrate(calls["bound"], iteration_cap))

    raw: dict[str, list[dict[str, Any]]] = {"callee": [], "bound": []}
    order = ("callee", "bound", "bound", "callee")
    for cycle in range(CYCLES):
        for order_index, label in enumerate(order):
            wall_us, gpu_us = measure_batch(calls[label], iterations)
            raw[label].append(
                {
                    "cycle": cycle,
                    "order_index": order_index,
                    "wall_us": wall_us,
                    "gpu_us": gpu_us,
                }
            )
    summaries = {label: summarize(samples) for label, samples in raw.items()}
    speedup = {
        "wall_p50": summaries["callee"]["wall_p50_us"] / summaries["bound"]["wall_p50_us"],
        "gpu_p50": summaries["callee"]["gpu_p50_us"] / summaries["bound"]["gpu_p50_us"],
    }
    allocation = {label: allocation_probe(call) for label, call in calls.items()}

    graphs: dict[str, Any] = {}
    graph_iterations = min(4096, iteration_cap)
    for label, call in calls.items():
        try:
            graph, captured_output = capture_graph(call)
            graphs[label] = {
                "status": "success",
                "output_address": int(captured_output.data_ptr()),
                "measurement": measure_graph(graph, graph_iterations),
            }
        except Exception as error:
            graphs[label] = {"status": "error", "error": f"{type(error).__name__}: {error}"}

    result = {
        "name": case.name,
        "family": case.family,
        "size_class": case.size_class,
        "logical_work_items": case.logical_work_items,
        "input_shapes": [list(value.shape) for value in case.inputs],
        "output_shape": list(case.output.shape),
        "output_bytes": output_bytes,
        "correctness": {"status": "success", "atol": case.atol, "rtol": case.rtol},
        "compile_seconds": {
            "callee_abi": callee_compile_seconds,
            "caller_abi_first_bind": bind_compile_seconds,
        },
        "cache": {
            "callee_key": getattr(kernel, "_tilelang_cache_key", None),
            "caller_key": getattr(companion, "_tilelang_cache_key", None),
            "distinct_keys": getattr(kernel, "_tilelang_cache_key", None)
            != getattr(companion, "_tilelang_cache_key", None),
        },
        "warmup_calls": warmups,
        "batch_iterations": iterations,
        "iteration_cap": iteration_cap,
        "cycles": CYCLES,
        "summaries": summaries,
        "speedup": speedup,
        "allocation_probe": allocation,
        "cuda_graph": graphs,
        "raw_samples": raw,
    }
    print(
        "BOUND_OUTPUT "
        f"case={case.name} iterations={iterations} "
        f"wall={summaries['callee']['wall_p50_us']:.3f}->{summaries['bound']['wall_p50_us']:.3f}us "
        f"speedup={speedup['wall_p50']:.3f}x "
        f"gpu={summaries['callee']['gpu_p50_us']:.3f}->{summaries['bound']['gpu_p50_us']:.3f}us",
        flush=True,
    )
    return result


def package_versions() -> dict[str, str]:
    names = ["tilelang", "apache-tvm-ffi", "torch-c-dlpack-ext", "z3-solver", "torch"]
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    global TORCH, TILELANG, TL

    overall_started = time.perf_counter()
    source_sha = os.environ.get(SOURCE_SHA_ENV, "")
    os.environ["TILELANG_CACHE_DIR"] = f"/tmp/tilelang-bound-outputs-cache-{source_sha[:12]}"
    prepared = prepare_overlay()
    base_wheel = install_base_wheel(prepared["base_wheel"])
    installer_log = apply_overlay(prepared)

    import torch
    import tilelang
    import tilelang.language as T

    TORCH = torch
    TILELANG = tilelang
    TL = T
    if not TORCH.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    if TILELANG.__version__ != EXPECTED_VERSION:
        raise RuntimeError(f"unexpected TileLang version: {TILELANG.__version__}")

    distribution = importlib.metadata.distribution("tilelang")
    package = Path(distribution.locate_file("tilelang")).resolve()
    identity = json.loads((package / "_python_overlay_identity.json").read_text())
    if identity.get("source_sha") != prepared["source_sha"]:
        raise RuntimeError("installed Python overlay identity does not match the Actions artifact")
    if identity.get("native_base_sha") != BASE_SOURCE_SHA:
        raise RuntimeError("installed Python overlay identifies a different native base")

    TORCH.manual_seed(20260901)
    TORCH.cuda.manual_seed_all(20260901)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_runtime": TORCH.version.cuda,
        "device_name": TORCH.cuda.get_device_name(0),
        "device_capability": list(TORCH.cuda.get_device_capability(0)),
        "cache_dir": os.environ["TILELANG_CACHE_DIR"],
        "runtime_identity": identity,
        "native_libraries": native_libraries(),
        "nvidia_smi_start": nvidia_snapshot(),
    }

    cases = build_cases()
    requested_cases = [name for name in os.environ.get(CASE_FILTER_ENV, "").split(",") if name]
    if requested_cases:
        available = {case.name: case for case in cases}
        missing = sorted(set(requested_cases) - set(available))
        if missing:
            raise RuntimeError(f"unknown benchmark cases: {missing}")
        cases = [available[name] for name in requested_cases]
    results = [benchmark_case(case) for case in cases]
    aggregates = aggregate_results(results)
    environment["nvidia_smi_end"] = nvidia_snapshot()
    payload = {
        "schema": "tilelang-bound-outputs-t4-v1",
        "status": "success",
        "created_unix": time.time(),
        "repository": REPOSITORY,
        "candidate_source_sha": prepared["source_sha"],
        "native_base_sha": BASE_SOURCE_SHA,
        "artifact": {
            "id": prepared["artifact_id"],
            "digest": prepared["artifact_digest"],
            "download": prepared["download"],
            "checksums": prepared["checksums"],
            "manifest": prepared["manifest"],
        },
        "base_wheel": base_wheel,
        "installer_log": installer_log,
        "environment": environment,
        "method": {
            "comparison": "existing callee allocation vs JITKernel.bind_outputs caller-owned reuse",
            "order": ["callee", "bound", "bound", "callee"],
            "cycles": CYCLES,
            "warmup_seconds_per_mode_per_case": WARMUP_SECONDS,
            "minimum_batch_seconds": MIN_BATCH_SECONDS,
            "maximum_transient_callee_output_bytes": MAX_TRANSIENT_OUTPUT_BYTES,
            "speedup_definition": "median(callee_wall_us) / median(bound_wall_us)",
            "aggregation": "geometric mean across cases, plus equal-weight geometric mean across operator families",
            "selected_cases": [case.name for case in cases],
        },
        "aggregates": aggregates,
        "results": results,
        "total_seconds": time.perf_counter() - overall_started,
        "evidence_boundary": (
            "One free Colab T4 screen of one exact Python overlay on one checksummed native wheel. "
            "It validates API semantics, output identity, numerical correctness, eager wall overhead, "
            f"GPU event time, allocation requests, and CUDA Graph controls for {len(cases)} fixed cases. "
            "It does not establish a multi-GPU, model-level, or 1.50x TileLang-wide claim."
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    compressed = gzip.compress(serialized, compresslevel=9)
    RESULT_PATH.write_bytes(serialized)
    RESULT_PATH.with_suffix(".json.gz").write_bytes(compressed)
    print(f"RESULT_JSON_SHA256={hashlib.sha256(serialized).hexdigest()}")
    print(f"RESULT_JSON_BYTES={len(serialized)}")
    print(f"RESULT_GZIP_BASE64={base64.b64encode(compressed).decode()}")
    print(RESULT_MARKER + json.dumps({"aggregates": aggregates, "status": "success"}, separators=(",", ":")))


if __name__ == "__main__":
    main()
