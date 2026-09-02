"""Parse per-kernel CUDA resource usage emitted by ``nvcc --resource-usage``.

NVIDIA documents the output as one block per entry function::

    ptxas info    : Compiling entry function 'main_kernel' for 'sm_80'
    ptxas info    : Function properties for main_kernel
        0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    ptxas info    : Used 32 registers, 2048 bytes smem, 352 bytes cmem[0]

The parser records those statistics in thread-local storage during JIT
compilation and removes only recognized resource-report lines from compiler
diagnostics. Warnings, errors, and unfamiliar ptxas output remain visible.
"""

from __future__ import annotations

from contextlib import contextmanager
import re
import threading
from typing import Iterator

from .kernel_resource_info import KernelResourceUsage

_ENTRY_RE = re.compile(
    r"^\s*ptxas info\s*:\s*Compiling entry function\s+['\"](?P<name>[^'\"]+)['\"]\s+for\s+['\"](?P<arch>[^'\"]+)['\"]\s*$"
)
_FUNCTION_RE = re.compile(r"^\s*ptxas info\s*:\s*Function properties for\s+(?P<name>\S+)\s*$")
_STACK_RE = re.compile(
    r"^\s*(?P<stack>\d+)\s+bytes stack frame,\s*"
    r"(?P<stores>\d+)\s+bytes spill stores,\s*"
    r"(?P<loads>\d+)\s+bytes spill loads\s*$"
)
_USED_RE = re.compile(r"^\s*ptxas info\s*:\s*Used\s+(?P<regs>\d+)\s+registers(?P<resources>.*)$")
_BARRIER_RE = re.compile(r"\bused\s+(?P<count>\d+)\s+barriers\b")
_RESOURCE_RE = re.compile(r"(?P<bytes>\d+)\s+bytes\s+(?P<name>[A-Za-z]+(?:\[\d+\])?)")
_MODULE_RESOURCE_RE = re.compile(r"^\s*ptxas info\s*:\s*(?:\d+\s+bytes\s+(?:gmem|cmem\[\d+\])(?:,\s*)?)+\s*$")

_RECORDER = threading.local()


def reset_recorder() -> None:
    """Begin a fresh CUDA resource-recording window on this thread."""
    _RECORDER.items = {}
    _RECORDER.last = {}
    _RECORDER.auto_launch_bounds_selection = None


def pop_recorded() -> dict[str, KernelResourceUsage]:
    """Return all CUDA resource records since the last reset and clear them."""
    items = getattr(_RECORDER, "items", {})
    _RECORDER.items = {}
    _RECORDER.last = {}
    return dict(items)


def pop_last_recorded() -> dict[str, KernelResourceUsage]:
    """Return records parsed by the most recent compiler invocation."""
    items = getattr(_RECORDER, "last", {})
    _RECORDER.last = {}
    return dict(items)


def record_usage(usage: dict[str, KernelResourceUsage]) -> None:
    """Merge cached compiler records into the active recording window."""
    items = getattr(_RECORDER, "items", None)
    for name, item in usage.items():
        _record(items, name, item)


def record_auto_launch_bounds_selection(min_blocks_per_sm: int) -> None:
    """Record the launch-bound variant selected by the latest CUDA compile."""

    _RECORDER.auto_launch_bounds_selection = int(min_blocks_per_sm)


def pop_auto_launch_bounds_selection() -> int | None:
    """Return and clear the latest automatic launch-bound selection."""

    selection = getattr(_RECORDER, "auto_launch_bounds_selection", None)
    _RECORDER.auto_launch_bounds_selection = None
    return selection


@contextmanager
def isolated_recorder() -> Iterator[None]:
    """Temporarily isolate nested CUDA compiler resource records.

    The enclosing recorder is restored exactly on exit. Callers can use
    ``pop_recorded`` inside the context to inspect multiple speculative
    compilations, then merge only the selected result with ``record_usage``.
    """

    had_items = hasattr(_RECORDER, "items")
    had_last = hasattr(_RECORDER, "last")
    had_selection = hasattr(_RECORDER, "auto_launch_bounds_selection")
    previous_items = getattr(_RECORDER, "items", None)
    previous_last = getattr(_RECORDER, "last", None)
    previous_selection = getattr(_RECORDER, "auto_launch_bounds_selection", None)
    reset_recorder()
    try:
        yield
    finally:
        if had_items:
            _RECORDER.items = previous_items
        else:
            _RECORDER.__dict__.pop("items", None)
        if had_last:
            _RECORDER.last = previous_last
        else:
            _RECORDER.__dict__.pop("last", None)
        if had_selection:
            _RECORDER.auto_launch_bounds_selection = previous_selection
        else:
            _RECORDER.__dict__.pop("auto_launch_bounds_selection", None)


def _record(items: dict[str, KernelResourceUsage] | None, name: str, usage: KernelResourceUsage) -> None:
    if items is None:
        return
    existing = items.get(name)
    if existing is None:
        items[name] = usage
        return
    existing_arch = existing.arch or "unknown"
    items.pop(name)
    items[f"{name}@{existing_arch}"] = existing
    items[f"{name}@{usage.arch or 'unknown'}"] = usage


def filter_and_record(output: str) -> str:
    """Record recognized resource blocks and return remaining diagnostics."""
    if "ptxas info" not in output:
        _RECORDER.last = {}
        return output

    kept_lines: list[str] = []
    current_name: str | None = None
    current: KernelResourceUsage | None = None
    items = getattr(_RECORDER, "items", None)
    parsed: dict[str, KernelResourceUsage] = {}

    def finalize() -> None:
        nonlocal current_name, current
        if current_name is not None and current is not None:
            _record(items, current_name, current)
            _record(parsed, current_name, current)
        current_name = None
        current = None

    for line in output.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        entry_match = _ENTRY_RE.match(stripped)
        if entry_match is not None:
            finalize()
            current_name = entry_match.group("name")
            current = KernelResourceUsage(arch=entry_match.group("arch"))
            continue

        function_match = _FUNCTION_RE.match(stripped)
        if function_match is not None and current is not None:
            current_name = function_match.group("name")
            continue

        stack_match = _STACK_RE.match(stripped)
        if stack_match is not None and current is not None:
            current.stack_frame_bytes = int(stack_match.group("stack"))
            current.scratch_bytes = current.stack_frame_bytes
            current.spill_store_bytes = int(stack_match.group("stores"))
            current.spill_load_bytes = int(stack_match.group("loads"))
            current.n_spills = (current.spill_store_bytes + current.spill_load_bytes) // 4
            continue

        used_match = _USED_RE.match(stripped)
        if used_match is not None and current is not None:
            current.n_regs = int(used_match.group("regs"))
            resources = used_match.group("resources")
            barrier_match = _BARRIER_RE.search(resources)
            if barrier_match is not None:
                current.barrier_count = int(barrier_match.group("count"))
            for resource_match in _RESOURCE_RE.finditer(resources):
                byte_count = int(resource_match.group("bytes"))
                resource_name = resource_match.group("name")
                current.extra[resource_name] = str(byte_count)
                if resource_name == "smem":
                    current.shared_bytes = byte_count
                elif resource_name == "lmem":
                    current.local_bytes = byte_count
                elif resource_name.startswith("cmem["):
                    current.constant_bytes += byte_count
            finalize()
            continue

        if _MODULE_RESOURCE_RE.match(stripped) is not None:
            continue
        kept_lines.append(line)

    finalize()
    _RECORDER.last = parsed
    return "".join(kept_lines)
