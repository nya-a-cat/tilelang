from __future__ import annotations

import cloudpickle
import os

import tilelang
import tilelang.cache.kernel_cache as kernel_cache_mod
from tilelang.backend import create_backend_context
from tilelang.cache.cuda_binary_cache import CUDABinaryCache
from tilelang.contrib.cuda_resource_info import filter_and_record, pop_recorded, reset_recorder
from tilelang.contrib.kernel_resource_info import KernelResourceUsage
from tilelang.cache.kernel_cache import KernelCache
from tilelang.env import env
from tvm.target import Target


def _set_cache_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    tilelang.enable_cache()
    KernelCache._get_cache_namespace.cache_clear()
    CUDABinaryCache._get_tilelang_lib_stamp.cache_clear()
    return cache_dir


def test_kernel_cache_namespace_includes_host_platform(monkeypatch):
    monkeypatch.setattr(kernel_cache_mod, "__version__", "1.2.3+cuda.gitabc")
    monkeypatch.setattr(kernel_cache_mod.sys, "platform", "linux")
    monkeypatch.setattr(kernel_cache_mod.platform, "machine", lambda: "aarch64")
    KernelCache._get_cache_namespace.cache_clear()

    assert KernelCache._get_cache_namespace() == os.path.join("1.2.3_cuda_gitabc", "linux-aarch64")


def test_cuda_binary_cache_hit_skips_nvcc_compile(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.cuda import backend as cuda_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    compile_calls = []

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False):
        compile_calls.append((code, target_format, tuple(arch), tuple(options or ())))
        return bytearray(b"fake-cubin")

    monkeypatch.setattr(cuda_backend.nvcc, "compile_cuda", fake_compile_cuda)

    target = Target({"kind": "cuda", "arch": "sm_90a"})
    source = 'extern "C" __global__ void kernel() {}'

    fast_math_pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["--extra-device-vectorization"],
    }

    first = cuda_backend.tilelang_callback_cuda_compile(source, target)
    second = cuda_backend.tilelang_callback_cuda_compile(source, target)
    # Different compiler options (e.g. --use_fast_math) change the generated
    # SASS without changing the source, so they must NOT share a cache entry.
    third = cuda_backend.tilelang_callback_cuda_compile(source, target, fast_math_pass_configs)
    fourth = cuda_backend.tilelang_callback_cuda_compile(source, target, fast_math_pass_configs)

    assert bytes(first) == b"fake-cubin"
    assert bytes(second) == b"fake-cubin"
    assert bytes(third) == b"fake-cubin"
    assert bytes(fourth) == b"fake-cubin"
    # first compiles, second hits; third compiles (new options), fourth hits
    assert len(compile_calls) == 2
    assert compile_calls[0][3] != compile_calls[1][3]
    assert all("--resource-usage" in call[3] for call in compile_calls)
    cache_files = list((tmp_path / "cache").glob("*/cuda-binaries/*.cubin"))
    assert len(cache_files) == 2


def _record_fake_cuda_usage(registers: int, spill_bytes: int = 0) -> None:
    filter_and_record(
        "ptxas info : Compiling entry function 'kernel' for 'sm_90a'\n"
        "ptxas info : Function properties for kernel\n"
        f"0 bytes stack frame, {spill_bytes} bytes spill stores, "
        f"{spill_bytes} bytes spill loads\n"
        f"ptxas info : Used {registers} registers, 4096 bytes smem\n"
    )


def test_cuda_auto_launch_bounds_selects_spill_free_register_reduction(
    monkeypatch, tmp_path
):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.cuda import backend as cuda_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")
    compile_sources = []

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False):
        compile_sources.append(code)
        if "__launch_bounds__(128, 2)" in code:
            _record_fake_cuda_usage(128)
            return bytearray(b"opt")
        _record_fake_cuda_usage(240)
        return bytearray(b"baseline")

    monkeypatch.setattr(cuda_backend.nvcc, "compile_cuda", fake_compile_cuda)
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    source = 'extern "C" __global__ void __launch_bounds__(128, 1) kernel() {}'

    reset_recorder()
    result = cuda_backend.tilelang_callback_cuda_compile(
        source,
        target,
        {tilelang.PassConfigKey.TL_ENABLE_AUTO_LAUNCH_BOUNDS: True},
    )
    usage = pop_recorded()

    assert bytes(result) == b"opt"
    assert len(compile_sources) == 2
    assert usage["kernel"].n_regs == 128
    assert usage["kernel"].n_spills == 0


def test_cuda_auto_launch_bounds_rejects_spilling_candidate(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.cuda import backend as cuda_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False):
        if "__launch_bounds__(128, 2)" in code:
            _record_fake_cuda_usage(128, spill_bytes=32)
            return bytearray(b"opt")
        _record_fake_cuda_usage(240)
        return bytearray(b"baseline")

    monkeypatch.setattr(cuda_backend.nvcc, "compile_cuda", fake_compile_cuda)
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    source = 'extern "C" __global__ void __launch_bounds__(128, 1) kernel() {}'

    reset_recorder()
    result = cuda_backend.tilelang_callback_cuda_compile(
        source,
        target,
        {tilelang.PassConfigKey.TL_ENABLE_AUTO_LAUNCH_BOUNDS: True},
    )
    usage = pop_recorded()

    assert bytes(result) == b"baseline"
    assert usage["kernel"].n_regs == 240
    assert usage["kernel"].n_spills == 0


def test_cuda_auto_launch_bounds_skips_low_register_kernel(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.cuda import backend as cuda_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")
    compile_sources = []

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False):
        compile_sources.append(code)
        _record_fake_cuda_usage(96)
        return bytearray(b"baseline")

    monkeypatch.setattr(cuda_backend.nvcc, "compile_cuda", fake_compile_cuda)
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    source = 'extern "C" __global__ void __launch_bounds__(128, 1) kernel() {}'

    result = cuda_backend.tilelang_callback_cuda_compile(
        source,
        target,
        {tilelang.PassConfigKey.TL_ENABLE_AUTO_LAUNCH_BOUNDS: True},
    )

    assert bytes(result) == b"baseline"
    assert compile_sources == [source]


def test_cuda_binary_cache_corrupted_entry_recompiles(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.cuda import backend as cuda_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    compile_calls = []

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False):
        compile_calls.append(code)
        return bytearray(b"fake-cubin")

    monkeypatch.setattr(cuda_backend.nvcc, "compile_cuda", fake_compile_cuda)

    target = Target({"kind": "cuda", "arch": "sm_90a"})
    source = 'extern "C" __global__ void kernel() {}'

    cuda_backend.tilelang_callback_cuda_compile(source, target)
    assert len(compile_calls) == 1

    [cache_file] = (tmp_path / "cache").glob("*/cuda-binaries/*.cubin")
    assert cache_file.with_name(cache_file.name + ".sha256").exists()
    # Same-size corruption, as left behind by a crashed writer/filesystem client.
    cache_file.write_bytes(b"\x00" * len(b"fake-cubin"))

    recompiled = cuda_backend.tilelang_callback_cuda_compile(source, target)
    assert bytes(recompiled) == b"fake-cubin"
    assert len(compile_calls) == 2

    # The corrupted entry was rewritten, so the next call hits the cache again.
    cuda_backend.tilelang_callback_cuda_compile(source, target)
    assert len(compile_calls) == 2


def test_cuda_binary_cache_accepts_legacy_entry_without_sidecar(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "legacy-key"
    path = CUDABinaryCache.get_path(key, "cubin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"legacy-cubin")

    assert CUDABinaryCache.load(key, "cubin") == b"legacy-cubin"


def test_cuda_binary_cache_round_trips_resource_usage(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    key = "resource-key"
    usage = {
        "kernel": KernelResourceUsage(
            n_regs=64,
            n_spills=2,
            shared_bytes=8192,
            barrier_count=4,
            arch="sm_100a",
        )
    }

    CUDABinaryCache.save_resource_usage(key, "cubin", usage)
    restored = CUDABinaryCache.load_resource_usage(key, "cubin")

    assert restored == usage


def test_cuda_binary_cache_hit_restores_resource_usage_to_jit_recorder(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.cuda import backend as cuda_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")
    compile_calls = []

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False):
        compile_calls.append(code)
        filter_and_record(
            "ptxas info : Compiling entry function 'kernel' for 'sm_90a'\n"
            "ptxas info : Function properties for kernel\n"
            "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "ptxas info : Used 72 registers, used 2 barriers, 4096 bytes smem\n"
        )
        return bytearray(b"fake-cubin")

    monkeypatch.setattr(cuda_backend.nvcc, "compile_cuda", fake_compile_cuda)
    target = Target({"kind": "cuda", "arch": "sm_90a"})
    source = 'extern "C" __global__ void kernel() {}'

    reset_recorder()
    cuda_backend.tilelang_callback_cuda_compile(source, target)
    first_usage = pop_recorded()
    reset_recorder()
    cuda_backend.tilelang_callback_cuda_compile(source, target)
    cached_usage = pop_recorded()

    assert len(compile_calls) == 1
    assert cached_usage == first_usage
    assert cached_usage["kernel"].n_regs == 72
    assert cached_usage["kernel"].barrier_count == 2


def test_disk_cache_load_failure_is_cache_miss(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    cache = KernelCache()
    key = "bad-host-executable"
    cache_path = tmp_path / "cache" / KernelCache._get_cache_namespace() / "kernels" / key
    cache_path.mkdir(parents=True)
    (cache_path / cache.device_kernel_path).write_text("// device")
    (cache_path / cache.host_kernel_path).write_text("// host")
    (cache_path / cache.kernel_lib_path).write_bytes(b"not-loadable")
    with (cache_path / cache.params_path).open("wb") as f:
        cloudpickle.dump(["param"], f)
    cache._write_manifest(str(cache_path))

    def fail_from_database(*args, **kwargs):
        raise RuntimeError("bad host executable")

    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(fail_from_database))

    loaded = cache._load_kernel_from_disk(
        key,
        backend_context=create_backend_context("cuda", execution_backend="tvm_ffi"),
        out_idx=[0],
        pass_configs=None,
        compile_flags=None,
        func=None,
    )

    assert loaded is None
    assert not cache_path.exists()
