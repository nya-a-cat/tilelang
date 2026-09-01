"""Occupancy-guarded selection between compiled layout candidates.

The layout inference pass exposes two complementary policies.  The
``register-count`` policy minimizes a compiler-side register-slot proxy, while
``io-aware`` scores global-memory traffic and coalescing.  This module compiles
both existing policies for the exact same PrimFunc and selects the generated
kernel whose measured compiler resources keep at least the same resident-warp
tier.  Selection happens once at compile time, so the returned object is an
ordinary :class:`JITKernel` with no runtime dispatch wrapper.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Any, TypeVar

LAYOUT_COST_MODEL_KEY = "tl.layout_cost_model"
AUTO_LAYOUT_POLICY = "auto"
REGISTER_COUNT_POLICY = "register-count"
IO_AWARE_POLICY = "io-aware"
_SELECTOR_VERSION = "occupancy-guard-v1"

_SM_ARCH_PATTERN = re.compile(r"(?<![A-Za-z0-9])sm[_-]?(?P<digits>\d{2,3})a?(?![A-Za-z0-9])", re.IGNORECASE)
_CUDA_KERNEL_PATTERN = re.compile(
    r"__global__\s+void\s+"
    r"(?:__launch_bounds__\(\s*(?P<threads>\d+)(?:\s*,\s*\d+)?\s*\)\s+)?"
    r"(?P<name>[A-Za-z_]\w*)\s*\([^;{]*\)\s*\{",
    re.DOTALL,
)

_KernelT = TypeVar("_KernelT")


@dataclass(frozen=True)
class CUDADeviceProfile:
    """Runtime CUDA limits needed for a post-compile residency estimate."""

    device_id: int
    arch: str
    warp_size: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    registers_per_sm: int
    shared_bytes_per_sm: int
    # The current CUDA Best Practices Guide describes 256-register allocation
    # units per warp.  Keeping the unit explicit makes the estimate auditable
    # and lets architecture-specific profiles replace it later.
    register_allocation_unit: int = 256
    shared_allocation_unit: int = 256

    def __post_init__(self) -> None:
        positive_fields = (
            "warp_size",
            "max_threads_per_sm",
            "max_blocks_per_sm",
            "registers_per_sm",
            "shared_bytes_per_sm",
            "register_allocation_unit",
            "shared_allocation_unit",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if _normalize_sm_arch(self.arch) is None:
            raise ValueError(f"Invalid CUDA architecture {self.arch!r}")

    @property
    def max_warps_per_sm(self) -> int:
        return self.max_threads_per_sm // self.warp_size


@dataclass(frozen=True)
class _EntrySummary:
    name: str
    threads_per_block: int
    registers_per_thread: int
    shared_bytes_per_block: int
    spill_bytes: int
    resident_blocks: int | None
    resident_warps: int | None


@dataclass(frozen=True)
class _CandidateSummary:
    policy: str
    source_sha256: str
    arch: str | None
    entries: tuple[_EntrySummary, ...]
    complete_residency: bool
    min_resident_warps: int | None
    spill_bytes: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "source_sha256": self.source_sha256,
            "arch": self.arch,
            "entries": [asdict(entry) for entry in self.entries],
            "complete_residency": self.complete_residency,
            "min_resident_warps": self.min_resident_warps,
            "spill_bytes": self.spill_bytes,
        }


def get_current_cuda_device_profile() -> CUDADeviceProfile | None:
    """Query the current CUDA device without importing PyTorch."""
    from tilelang.carver.arch.driver.cuda_driver import (
        cudaDeviceAttrNames,
        get_current_device,
        get_device_attribute,
    )

    device_id = get_current_device()
    if device_id is None:
        return None
    attributes = {
        "warp_size": cudaDeviceAttrNames.cudaDevAttrWarpSize,
        "max_threads_per_sm": cudaDeviceAttrNames.cudaDevAttrMaxThreadsPerMultiProcessor,
        "max_blocks_per_sm": cudaDeviceAttrNames.cudaDevAttrMaxBlocksPerMultiprocessor,
        "registers_per_sm": cudaDeviceAttrNames.cudaDevAttrMaxRegistersPerMultiprocessor,
        "shared_bytes_per_sm": cudaDeviceAttrNames.cudaDevAttrMaxSharedMemoryPerMultiprocessor,
        "major": cudaDeviceAttrNames.cudaDevAttrComputeCapabilityMajor,
        "minor": cudaDeviceAttrNames.cudaDevAttrComputeCapabilityMinor,
    }
    values = {name: get_device_attribute(attribute, device_id) for name, attribute in attributes.items()}
    positive_names = ("warp_size", "max_threads_per_sm", "max_blocks_per_sm", "registers_per_sm", "shared_bytes_per_sm", "major")
    if any(values[name] is None or values[name] <= 0 for name in positive_names):
        return None
    if values["minor"] is None or values["minor"] < 0:
        return None
    return CUDADeviceProfile(
        device_id=device_id,
        arch=f"sm_{values['major']}{values['minor']}",
        warp_size=values["warp_size"],
        max_threads_per_sm=values["max_threads_per_sm"],
        max_blocks_per_sm=values["max_blocks_per_sm"],
        registers_per_sm=values["registers_per_sm"],
        shared_bytes_per_sm=values["shared_bytes_per_sm"],
    )


def _normalize_sm_arch(value: Any) -> str | None:
    if value is None:
        return None
    match = _SM_ARCH_PATTERN.search(str(value))
    return f"sm_{match.group('digits')}" if match is not None else None


def _kernel_source(kernel: Any) -> str:
    getter = getattr(kernel, "get_kernel_source", None)
    if callable(getter):
        source = getter()
    else:
        source = getattr(kernel, "kernel_source", "")
    return str(source or "")


def _kernel_resource_usage(kernel: Any) -> Mapping[str, Any]:
    usage = getattr(kernel, "resource_usage", {})
    return usage if isinstance(usage, Mapping) else {}


def _resource_field(entry: Any, name: str, default: Any = 0) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _target_arch(kernel: Any) -> str | None:
    target = getattr(kernel, "target", None)
    if target is None:
        return None
    export = getattr(target, "export", None)
    if callable(export):
        try:
            exported = export()
            if isinstance(exported, Mapping):
                arch = _normalize_sm_arch(exported.get("arch"))
                if arch is not None:
                    return arch
        except Exception:  # pragma: no cover - target string fallback is deterministic
            pass
    attrs = getattr(target, "attrs", None)
    getter = getattr(attrs, "get", None)
    if callable(getter):
        try:
            arch = _normalize_sm_arch(getter("arch"))
            if arch is not None:
                return arch
        except Exception:  # pragma: no cover - target string fallback is deterministic
            pass
    return _normalize_sm_arch(target)


def _compiled_arch(kernel: Any) -> str | None:
    target_arch = _target_arch(kernel)
    usage_arches = {
        normalized
        for entry in _kernel_resource_usage(kernel).values()
        if (normalized := _normalize_sm_arch(_resource_field(entry, "arch", None))) is not None
    }
    if target_arch is not None:
        usage_arches.add(target_arch)
    return next(iter(usage_arches)) if len(usage_arches) == 1 else None


def _launch_bounds(source: str) -> dict[str, int]:
    return {
        match.group("name"): int(match.group("threads"))
        for match in _CUDA_KERNEL_PATTERN.finditer(source)
        if match.group("threads") is not None
    }


def _round_up(value: int, unit: int) -> int:
    return math.ceil(value / unit) * unit


def _estimate_residency(
    *,
    threads_per_block: int,
    registers_per_thread: int,
    shared_bytes_per_block: int,
    profile: CUDADeviceProfile,
) -> tuple[int, int]:
    warps_per_block = math.ceil(threads_per_block / profile.warp_size)
    capacities = [
        profile.max_blocks_per_sm,
        profile.max_warps_per_sm // warps_per_block,
    ]
    if registers_per_thread > 0:
        registers_per_warp = _round_up(
            registers_per_thread * profile.warp_size,
            profile.register_allocation_unit,
        )
        registers_per_block = registers_per_warp * warps_per_block
        capacities.append(profile.registers_per_sm // registers_per_block)
    if shared_bytes_per_block > 0:
        allocated_shared_bytes = _round_up(shared_bytes_per_block, profile.shared_allocation_unit)
        capacities.append(profile.shared_bytes_per_sm // allocated_shared_bytes)
    resident_blocks = min(capacities)
    return resident_blocks, resident_blocks * warps_per_block


def _summarize_candidate(kernel: Any, policy: str, profile: CUDADeviceProfile | None) -> _CandidateSummary:
    source = _kernel_source(kernel)
    bounds = _launch_bounds(source)
    entries: list[_EntrySummary] = []
    usage = _kernel_resource_usage(kernel)
    for name, item in sorted(usage.items()):
        max_threads = _resource_field(item, "n_max_threads", None)
        threads_per_block = int(max_threads or bounds.get(name, 0))
        registers_per_thread = int(_resource_field(item, "n_regs", 0) or 0)
        shared_bytes_per_block = int(_resource_field(item, "shared_bytes", 0) or 0)
        spill_bytes = int(_resource_field(item, "spill_store_bytes", 0) or 0) + int(_resource_field(item, "spill_load_bytes", 0) or 0)
        resident_blocks = None
        resident_warps = None
        if profile is not None and threads_per_block > 0 and registers_per_thread > 0:
            resident_blocks, resident_warps = _estimate_residency(
                threads_per_block=threads_per_block,
                registers_per_thread=registers_per_thread,
                shared_bytes_per_block=shared_bytes_per_block,
                profile=profile,
            )
        entries.append(
            _EntrySummary(
                name=name,
                threads_per_block=threads_per_block,
                registers_per_thread=registers_per_thread,
                shared_bytes_per_block=shared_bytes_per_block,
                spill_bytes=spill_bytes,
                resident_blocks=resident_blocks,
                resident_warps=resident_warps,
            )
        )
    complete_residency = bool(entries) and all(entry.resident_warps is not None for entry in entries)
    min_resident_warps = min(entry.resident_warps for entry in entries) if complete_residency else None
    return _CandidateSummary(
        policy=policy,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        arch=_compiled_arch(kernel),
        entries=tuple(entries),
        complete_residency=complete_residency,
        min_resident_warps=min_resident_warps,
        spill_bytes=sum(entry.spill_bytes for entry in entries),
    )


def _selection_metadata(
    *,
    selected: str,
    reason: str,
    profile: CUDADeviceProfile | None,
    register: _CandidateSummary | None = None,
    io_aware: _CandidateSummary | None = None,
    compile_errors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "selector": _SELECTOR_VERSION,
        "requested": AUTO_LAYOUT_POLICY,
        "selected": selected,
        "reason": reason,
        "device": asdict(profile) if profile is not None else None,
        "candidates": {
            REGISTER_COUNT_POLICY: register.to_metadata() if register is not None else None,
            IO_AWARE_POLICY: io_aware.to_metadata() if io_aware is not None else None,
        },
        "compile_errors": dict(compile_errors or {}),
    }


def _attach_selection(kernel: _KernelT, metadata: dict[str, Any]) -> _KernelT:
    kernel.layout_selection = metadata
    return kernel


def select_compiled_layout(
    register_kernel: _KernelT,
    io_aware_kernel: _KernelT,
    *,
    device_profile: CUDADeviceProfile | None = None,
) -> _KernelT:
    """Select one already-compiled layout candidate conservatively."""
    register_source = _kernel_source(register_kernel)
    io_aware_source = _kernel_source(io_aware_kernel)
    if register_source == io_aware_source:
        register = _summarize_candidate(register_kernel, REGISTER_COUNT_POLICY, None)
        io_aware = _summarize_candidate(io_aware_kernel, IO_AWARE_POLICY, None)
        return _attach_selection(
            register_kernel,
            _selection_metadata(
                selected=REGISTER_COUNT_POLICY,
                reason="identical-generated-source",
                profile=None,
                register=register,
                io_aware=io_aware,
            ),
        )

    profile = device_profile if device_profile is not None else get_current_cuda_device_profile()
    register = _summarize_candidate(register_kernel, REGISTER_COUNT_POLICY, profile)
    io_aware = _summarize_candidate(io_aware_kernel, IO_AWARE_POLICY, profile)

    if register.spill_bytes != io_aware.spill_bytes and (register.spill_bytes == 0 or io_aware.spill_bytes == 0):
        if io_aware.spill_bytes == 0:
            selected_kernel = io_aware_kernel
            selected_policy = IO_AWARE_POLICY
        else:
            selected_kernel = register_kernel
            selected_policy = REGISTER_COUNT_POLICY
        return _attach_selection(
            selected_kernel,
            _selection_metadata(
                selected=selected_policy,
                reason="avoid-compiler-reported-spills",
                profile=profile,
                register=register,
                io_aware=io_aware,
            ),
        )

    if profile is None:
        reason = "cuda-device-profile-unavailable"
    elif register.arch != _normalize_sm_arch(profile.arch) or io_aware.arch != _normalize_sm_arch(profile.arch):
        reason = "compiled-architecture-does-not-match-current-device"
    elif not register.complete_residency or not io_aware.complete_residency:
        reason = "compiler-resource-usage-incomplete"
    elif io_aware.min_resident_warps >= register.min_resident_warps:
        return _attach_selection(
            io_aware_kernel,
            _selection_metadata(
                selected=IO_AWARE_POLICY,
                reason="io-aware-without-resident-warp-loss",
                profile=profile,
                register=register,
                io_aware=io_aware,
            ),
        )
    else:
        reason = "io-aware-reduces-resident-warps"

    return _attach_selection(
        register_kernel,
        _selection_metadata(
            selected=REGISTER_COUNT_POLICY,
            reason=reason,
            profile=profile,
            register=register,
            io_aware=io_aware,
        ),
    )


def compile_auto_layout(
    compile_variant: Callable[[dict[str, Any]], _KernelT],
    pass_configs: Mapping[str, Any],
    *,
    device_profile: CUDADeviceProfile | None = None,
) -> _KernelT:
    """Compile the two concrete layout policies and return the selected kernel."""
    base_configs = dict(pass_configs)
    if base_configs.get(LAYOUT_COST_MODEL_KEY) != AUTO_LAYOUT_POLICY:
        raise ValueError(f"{LAYOUT_COST_MODEL_KEY} must be {AUTO_LAYOUT_POLICY!r}")

    register_configs = dict(base_configs)
    register_configs[LAYOUT_COST_MODEL_KEY] = REGISTER_COUNT_POLICY
    try:
        register_kernel = compile_variant(register_configs)
    except Exception as register_error:
        io_configs = dict(base_configs)
        io_configs[LAYOUT_COST_MODEL_KEY] = IO_AWARE_POLICY
        try:
            io_aware_kernel = compile_variant(io_configs)
        except Exception as io_error:
            raise RuntimeError(
                "Both automatic layout candidates failed to compile: "
                f"{REGISTER_COUNT_POLICY}={register_error!r}; {IO_AWARE_POLICY}={io_error!r}"
            ) from register_error
        metadata = _selection_metadata(
            selected=IO_AWARE_POLICY,
            reason="register-count-compile-failed",
            profile=device_profile,
            compile_errors={REGISTER_COUNT_POLICY: repr(register_error)},
        )
        return _attach_selection(io_aware_kernel, metadata)

    io_configs = dict(base_configs)
    io_configs[LAYOUT_COST_MODEL_KEY] = IO_AWARE_POLICY
    try:
        io_aware_kernel = compile_variant(io_configs)
    except Exception as io_error:
        metadata = _selection_metadata(
            selected=REGISTER_COUNT_POLICY,
            reason="io-aware-compile-failed",
            profile=device_profile,
            compile_errors={IO_AWARE_POLICY: repr(io_error)},
        )
        return _attach_selection(register_kernel, metadata)

    return select_compiled_layout(register_kernel, io_aware_kernel, device_profile=device_profile)
