"""Frozen, environment-specific measurements for graph layout selection.

Each key describes one isolated operator configuration or directed conversion.
Measurement producers must include their executable and timing protocol digests;
the graph solver never substitutes an unmeasured entry with a zero cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from types import MappingProxyType


ENVIRONMENT_FIELDS = (
    "gpu_name", "compute_capability", "driver_version", "cuda_version",
    "tilelang_revision", "tvm_revision", "compiler_flags", "timing_protocol",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def measurement_key(kind, *, operator_ir, dtype, shape, layouts, schedule):
    if kind not in ("operator", "conversion"):
        raise ValueError("measurement kind must be operator or conversion")
    if not isinstance(operator_ir, str) or not operator_ir:
        raise ValueError("an isolated operator IR identity is required")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("dtype is required")
    if not shape or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("shape must contain positive static integers")
    if not layouts or any(not isinstance(x, str) or not x for x in layouts):
        raise ValueError("ordered layout identities are required")
    if not isinstance(schedule, dict) or not schedule:
        raise ValueError("the complete kernel schedule identity is required")
    return dict(kind=kind, operator_ir=operator_ir, dtype=dtype, shape=list(shape),
                layouts=list(layouts), schedule=schedule)


class MissingMeasurement(KeyError):
    """The frozen calibration does not cover a requested configuration."""


class LatencyTable:
    def __init__(self, document, expected_environment):
        # Detach caller-owned containers before validating and freezing.
        document = json.loads(canonical(document))
        environment = document.get("environment")
        if document.get("schema") != 1 or not isinstance(environment, dict):
            raise ValueError("unsupported latency table schema")
        if any(k not in environment or environment[k] in (None, "") for k in ENVIRONMENT_FIELDS):
            raise ValueError("incomplete calibration environment")
        if canonical(environment) != canonical(expected_environment):
            raise ValueError("calibration environment differs from compilation environment")
        if document.get("frozen") is not True:
            raise ValueError("calibration must be frozen before layout selection")
        entries = {}
        for entry in document.get("entries", []):
            key = measurement_key(**entry["key"])
            identity = digest(key)
            if identity in entries:
                raise ValueError("duplicate measurement key")
            samples = entry["samples_ns"]
            if len(samples) < 3 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("at least three positive finite timing samples are required")
            if not isinstance(entry.get("executable_sha256"), str) or len(entry["executable_sha256"]) != 64:
                raise ValueError("measurement executable SHA-256 is required")
            if any(c not in "0123456789abcdef" for c in entry["executable_sha256"]):
                raise ValueError("invalid executable SHA-256")
            cost = max(1, math.ceil(statistics.median(samples)))
            if type(entry.get("cost_ns")) is not int or entry["cost_ns"] != cost:
                raise ValueError("integer cost differs from measured median")
            entries[identity] = cost
        self._entries = MappingProxyType(entries)
        self._serialized = canonical(document)
        self.sha256 = hashlib.sha256(self._serialized.encode()).hexdigest()

    @classmethod
    def read(cls, path, expected_environment):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), expected_environment)

    def cost(self, key):
        identity = digest(measurement_key(**key))
        if identity not in self._entries:
            raise MissingMeasurement(identity)
        return self._entries[identity]

    def coverage(self, keys):
        identities = {digest(measurement_key(**key)) for key in keys}
        missing = sorted(identities.difference(self._entries))
        return dict(required=len(identities), measured=len(identities) - len(missing), missing=missing,
                    table_sha256=self.sha256)

    def write(self, path):
        Path(path).write_text(self._serialized + "\n", encoding="utf-8")
