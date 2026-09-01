from __future__ import annotations

from dataclasses import dataclass
import importlib

import pytest

from tilelang.jit.layout_autoselect import (
    AUTO_LAYOUT_POLICY,
    CUDADeviceProfile,
    IO_AWARE_POLICY,
    LAYOUT_COST_MODEL_KEY,
    REGISTER_COUNT_POLICY,
    compile_auto_layout,
    get_current_cuda_device_profile,
    select_compiled_layout,
)


@dataclass
class _Usage:
    n_regs: int
    arch: str = "sm_75"
    n_max_threads: int | None = None
    shared_bytes: int = 0
    spill_store_bytes: int = 0
    spill_load_bytes: int = 0


class _Kernel:
    def __init__(self, source: str, usage: _Usage, *, target: str = "cuda -arch=sm_75"):
        self._source = source
        self.resource_usage = {"kernel": usage}
        self.target = target

    def get_kernel_source(self) -> str:
        return self._source


@pytest.fixture
def t4_profile() -> CUDADeviceProfile:
    return CUDADeviceProfile(
        device_id=0,
        arch="sm_75",
        warp_size=32,
        max_threads_per_sm=1024,
        max_blocks_per_sm=16,
        registers_per_sm=65536,
        shared_bytes_per_sm=65536,
    )


def _source(threads: int, marker: str) -> str:
    return f'extern "C" __global__ void __launch_bounds__({threads}, 1) kernel(float* output) {{ /* {marker} */ output[0] = 0; }}'


def test_selects_io_aware_when_resident_warp_tier_is_preserved(t4_profile):
    register = _Kernel(_source(256, "register"), _Usage(n_regs=8))
    io_aware = _Kernel(_source(256, "io"), _Usage(n_regs=26))

    selected = select_compiled_layout(register, io_aware, device_profile=t4_profile)

    assert selected is io_aware
    assert selected.layout_selection["selected"] == IO_AWARE_POLICY
    assert selected.layout_selection["reason"] == "io-aware-without-resident-warp-loss"
    candidates = selected.layout_selection["candidates"]
    assert candidates[REGISTER_COUNT_POLICY]["min_resident_warps"] == 32
    assert candidates[IO_AWARE_POLICY]["min_resident_warps"] == 32


def test_preserves_register_layout_when_io_aware_crosses_occupancy_tier(t4_profile):
    register = _Kernel(_source(256, "register"), _Usage(n_regs=64))
    io_aware = _Kernel(_source(256, "io"), _Usage(n_regs=72))

    selected = select_compiled_layout(register, io_aware, device_profile=t4_profile)

    assert selected is register
    assert selected.layout_selection["reason"] == "io-aware-reduces-resident-warps"
    candidates = selected.layout_selection["candidates"]
    assert candidates[REGISTER_COUNT_POLICY]["min_resident_warps"] == 32
    assert candidates[IO_AWARE_POLICY]["min_resident_warps"] == 24


def test_avoids_spills_before_applying_occupancy_guard(t4_profile):
    register = _Kernel(_source(256, "register"), _Usage(n_regs=64, spill_store_bytes=8))
    io_aware = _Kernel(_source(256, "io"), _Usage(n_regs=72))

    selected = select_compiled_layout(register, io_aware, device_profile=t4_profile)

    assert selected is io_aware
    assert selected.layout_selection["reason"] == "avoid-compiler-reported-spills"


def test_identical_generated_source_keeps_stable_register_policy(t4_profile):
    source = _source(128, "same")
    register = _Kernel(source, _Usage(n_regs=24))
    io_aware = _Kernel(source, _Usage(n_regs=24))

    selected = select_compiled_layout(register, io_aware, device_profile=t4_profile)

    assert selected is register
    assert selected.layout_selection["reason"] == "identical-generated-source"


def test_missing_device_profile_and_cross_compile_fall_back_to_register(monkeypatch, t4_profile):
    register = _Kernel(_source(128, "register"), _Usage(n_regs=24))
    io_aware = _Kernel(_source(128, "io"), _Usage(n_regs=20))
    layout_module = importlib.import_module("tilelang.jit.layout_autoselect")
    monkeypatch.setattr(layout_module, "get_current_cuda_device_profile", lambda: None)

    selected_without_device = select_compiled_layout(register, io_aware)
    assert selected_without_device is register
    assert selected_without_device.layout_selection["reason"] == "cuda-device-profile-unavailable"

    cross_compiled = _Kernel(_source(128, "io"), _Usage(n_regs=20, arch="sm_90"), target="cuda -arch=sm_90")
    selected_cross_compile = select_compiled_layout(register, cross_compiled, device_profile=t4_profile)
    assert selected_cross_compile is register
    assert selected_cross_compile.layout_selection["reason"] == "compiled-architecture-does-not-match-current-device"


def test_auto_compile_replaces_pseudo_policy_without_mutating_input(t4_profile):
    original = {LAYOUT_COST_MODEL_KEY: AUTO_LAYOUT_POLICY, "tl.enable_fast_math": True}
    seen: list[dict[str, object]] = []
    register = _Kernel(_source(256, "register"), _Usage(n_regs=8))
    io_aware = _Kernel(_source(256, "io"), _Usage(n_regs=26))

    def compile_variant(pass_configs):
        seen.append(dict(pass_configs))
        return register if pass_configs[LAYOUT_COST_MODEL_KEY] == REGISTER_COUNT_POLICY else io_aware

    selected = compile_auto_layout(compile_variant, original, device_profile=t4_profile)

    assert selected is io_aware
    assert original[LAYOUT_COST_MODEL_KEY] == AUTO_LAYOUT_POLICY
    assert [config[LAYOUT_COST_MODEL_KEY] for config in seen] == [REGISTER_COUNT_POLICY, IO_AWARE_POLICY]
    assert all(config["tl.enable_fast_math"] is True for config in seen)


def test_auto_compile_returns_register_candidate_when_io_compile_fails(t4_profile):
    register = _Kernel(_source(128, "register"), _Usage(n_regs=24))

    def compile_variant(pass_configs):
        if pass_configs[LAYOUT_COST_MODEL_KEY] == IO_AWARE_POLICY:
            raise RuntimeError("io failure")
        return register

    selected = compile_auto_layout(
        compile_variant,
        {LAYOUT_COST_MODEL_KEY: AUTO_LAYOUT_POLICY},
        device_profile=t4_profile,
    )

    assert selected is register
    assert selected.layout_selection["reason"] == "io-aware-compile-failed"
    assert "io failure" in selected.layout_selection["compile_errors"][IO_AWARE_POLICY]


def test_cuda_device_profile_accepts_zero_compute_capability_minor(monkeypatch):
    from tilelang.carver.arch.driver import cuda_driver

    values = {
        cuda_driver.cudaDeviceAttrNames.cudaDevAttrWarpSize: 32,
        cuda_driver.cudaDeviceAttrNames.cudaDevAttrMaxThreadsPerMultiProcessor: 2048,
        cuda_driver.cudaDeviceAttrNames.cudaDevAttrMaxBlocksPerMultiprocessor: 32,
        cuda_driver.cudaDeviceAttrNames.cudaDevAttrMaxRegistersPerMultiprocessor: 65536,
        cuda_driver.cudaDeviceAttrNames.cudaDevAttrMaxSharedMemoryPerMultiprocessor: 233472,
        cuda_driver.cudaDeviceAttrNames.cudaDevAttrComputeCapabilityMajor: 9,
        cuda_driver.cudaDeviceAttrNames.cudaDevAttrComputeCapabilityMinor: 0,
    }
    monkeypatch.setattr(cuda_driver, "get_current_device", lambda: 3)
    monkeypatch.setattr(cuda_driver, "get_device_attribute", lambda attribute, device_id: values[attribute])

    profile = get_current_cuda_device_profile()

    assert profile is not None
    assert profile.device_id == 3
    assert profile.arch == "sm_90"


def test_public_compile_intercepts_auto_before_the_native_pass(monkeypatch):
    jit_module = importlib.import_module("tilelang.jit")

    class _PrimFunc:
        attrs = {}

    cached_calls = []
    sentinel = object()

    def fake_cached(**kwargs):
        cached_calls.append(kwargs)
        return sentinel

    def fake_auto_selector(compile_variant, pass_configs):
        assert pass_configs[LAYOUT_COST_MODEL_KEY] == AUTO_LAYOUT_POLICY
        candidate = dict(pass_configs)
        candidate[LAYOUT_COST_MODEL_KEY] = IO_AWARE_POLICY
        return compile_variant(candidate)

    monkeypatch.setattr(jit_module, "PrimFunc", _PrimFunc)
    monkeypatch.setattr(jit_module, "cached", fake_cached)
    monkeypatch.setattr(jit_module, "compile_auto_layout", fake_auto_selector)
    monkeypatch.setattr(jit_module, "normalize_pass_configs", lambda pass_configs: dict(pass_configs or {}))

    result = jit_module.compile(_PrimFunc(), pass_configs={LAYOUT_COST_MODEL_KEY: AUTO_LAYOUT_POLICY})

    assert result is sentinel
    assert len(cached_calls) == 1
    assert cached_calls[0]["pass_configs"][LAYOUT_COST_MODEL_KEY] == IO_AWARE_POLICY
