"""Backend-neutral compiler resource records and cache serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KernelResourceUsage:
    """Normalized per-entry resource usage reported by a device compiler."""

    n_regs: int = 0
    # Dword-equivalent spill pressure. Backends retain their raw byte/count
    # fields below so callers can apply a different traffic model when needed.
    n_spills: int = 0
    scratch_bytes: int = 0
    n_max_threads: int | None = None
    shared_bytes: int = 0
    stack_frame_bytes: int = 0
    spill_store_bytes: int = 0
    spill_load_bytes: int = 0
    constant_bytes: int = 0
    local_bytes: int = 0
    barrier_count: int = 0
    arch: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


def to_dict(usage: dict[str, KernelResourceUsage]) -> dict[str, dict[str, Any]]:
    """Convert resource records to a JSON-serializable mapping."""
    return {name: asdict(item) for name, item in usage.items()}


def from_dict(data: dict[str, dict[str, Any]]) -> dict[str, KernelResourceUsage]:
    """Build resource records while tolerating fields absent from older caches."""
    if not isinstance(data, dict):
        raise ValueError("Resource usage payload must be a mapping")
    out: dict[str, KernelResourceUsage] = {}
    for name, entry in data.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("Resource usage entries must map kernel names to objects")
        out[name] = KernelResourceUsage(
            n_regs=int(entry.get("n_regs", 0)),
            n_spills=int(entry.get("n_spills", 0)),
            scratch_bytes=int(entry.get("scratch_bytes", 0)),
            n_max_threads=entry.get("n_max_threads"),
            shared_bytes=int(entry.get("shared_bytes", 0)),
            stack_frame_bytes=int(entry.get("stack_frame_bytes", 0)),
            spill_store_bytes=int(entry.get("spill_store_bytes", 0)),
            spill_load_bytes=int(entry.get("spill_load_bytes", 0)),
            constant_bytes=int(entry.get("constant_bytes", 0)),
            local_bytes=int(entry.get("local_bytes", 0)),
            barrier_count=int(entry.get("barrier_count", 0)),
            arch=entry.get("arch"),
            extra=dict(entry.get("extra", {})),
        )
    return out


def dump_to_file(usage: dict[str, KernelResourceUsage], path: str) -> None:
    """Persist parsed resource usage so it survives kernel-cache hits."""
    with open(path, "w") as file:
        json.dump(to_dict(usage), file, indent=2, sort_keys=True)


def load_from_file(path: str) -> dict[str, KernelResourceUsage]:
    """Load resource records while tolerating fields absent from older caches."""
    with open(path) as file:
        data = json.load(file)
    return from_dict(data)
