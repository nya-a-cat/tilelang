"""CPU-only regression tests for exact NVRTC CUDA target propagation."""

from __future__ import annotations

import os

import pytest

from tilelang import tvm
from tilelang.contrib.nvrtc import get_nvrtc_arch_option
from tilelang.jit.adapter.nvrtc import is_nvrtc_available


@pytest.mark.parametrize(
    ("arch", "target_format", "expected"),
    [
        (80, "ptx", "--gpu-architecture=compute_80"),
        (90, "cubin", "--gpu-architecture=sm_90"),
        ("sm_90a", "cubin", "--gpu-architecture=sm_90a"),
        ("compute_100", "ptx", "--gpu-architecture=compute_100"),
        ("100f", "cubin", "--gpu-architecture=sm_100f"),
        ("sm_103a", "cubin", "--gpu-architecture=sm_103a"),
        ("sm_120a", "ptx", "--gpu-architecture=compute_120a"),
    ],
)
def test_nvrtc_arch_option_preserves_exact_feature_suffix(arch, target_format, expected):
    assert get_nvrtc_arch_option(target_format, arch) == expected


def test_nvrtc_arch_option_uses_current_target_exactly():
    with tvm.target.Target({"kind": "cuda", "arch": "sm_100f"}):
        assert get_nvrtc_arch_option("cubin") == "--gpu-architecture=sm_100f"


@pytest.mark.parametrize("arch", ["100x", "gfx942", "sm_", "100aa"])
def test_nvrtc_arch_option_rejects_invalid_tokens(arch):
    with pytest.raises(ValueError, match="exact CUDA architecture token"):
        get_nvrtc_arch_option("cubin", arch)


@pytest.mark.skipif(not is_nvrtc_available, reason="cuda-python is required to import the NVRTC adapter")
def test_nvrtc_library_generator_forwards_target_arch(monkeypatch):
    from tilelang.jit.adapter.nvrtc import libgen

    captured = {}

    def fake_compile_cuda(code, target_format, arch, options, verbose):
        captured.update(code=code, target_format=target_format, arch=arch, options=options, verbose=verbose)
        return b"cubin"

    monkeypatch.setattr(libgen, "compile_cuda", fake_compile_cuda)
    monkeypatch.setattr(libgen, "discover_cuda_include_paths", lambda _cuda_home: [])

    generator = libgen.NVRTCLibraryGenerator(tvm.target.Target({"kind": "cuda", "arch": "sm_103a"}))
    generator.assign_compile_flags([])
    generator.update_lib_code('extern "C" __global__ void main() {}')
    generator.update_host_func("")
    generator.compile_lib()

    try:
        assert captured["target_format"] == "cubin"
        assert captured["arch"] == "103a"
    finally:
        for path in (generator.srcpath, generator.libpath, generator.pypath):
            if path and os.path.exists(path):
                os.remove(path)
