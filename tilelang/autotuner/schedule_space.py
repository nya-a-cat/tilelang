"""Deterministic configuration spaces for TileLang autotuning.

This module is deliberately independent from TVM so schedule spaces can be
constructed, inspected, and filtered before any compiler or device runtime is
loaded.  A :class:`ScheduleSpace` is a ``list`` subclass, which keeps it fully
compatible with the existing autotuner and its JSON cache keys.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

ScheduleConfig = dict[str, Any]
ConstraintPredicate = Callable[[Mapping[str, Any], "TargetProfile | None"], bool]
ValueTransform = Callable[[Any], Any]

_ARCH_PATTERN = re.compile(r"(?<![A-Za-z0-9])((?:sm|gfx)[_-]?[A-Za-z0-9]+)")


def _target_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class TargetProfile:
    """Compiler-independent target capabilities used while filtering a space.

    Features are explicit.  This keeps early schedule-space construction from
    silently guessing instruction support from an architecture name.  A later
    compiler integration can populate them from TileLang's target capability
    queries.
    """

    backend: str
    arch: str | None = None
    features: frozenset[str] = frozenset()
    limits: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = self.backend.strip().lower()
        if not backend:
            raise ValueError("Target backend must be a non-empty string")
        arch = _target_text(self.arch)
        features = frozenset(feature.strip() for feature in self.features if feature.strip())
        limits = {str(name): int(value) for name, value in self.limits.items()}
        if any(value < 0 for value in limits.values()):
            raise ValueError("Target limits must be non-negative")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "arch", arch)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "limits", MappingProxyType(limits))

    @classmethod
    def from_target(
        cls,
        target: str | Mapping[str, object] | Any,
        *,
        features: Iterable[str] = (),
        limits: Mapping[str, int] | None = None,
    ) -> TargetProfile:
        """Create a profile from a target string, mapping, or TVM-like object."""

        backend: str | None = None
        arch: str | None = None
        if isinstance(target, Mapping):
            kind = target.get("kind")
            backend = _target_text(getattr(kind, "name", kind))
            arch = _target_text(target.get("arch"))
        elif isinstance(target, str):
            text = target.strip()
            backend = text.split(maxsplit=1)[0] if text else None
            match = _ARCH_PATTERN.search(text)
            arch = match.group(1) if match else None
        else:
            kind = getattr(target, "kind", None)
            backend = _target_text(getattr(kind, "name", kind))
            attrs = getattr(target, "attrs", {})
            if isinstance(attrs, Mapping):
                arch = _target_text(attrs.get("arch"))
            if arch is None:
                arch = _target_text(getattr(target, "arch", None))

        if backend is None:
            raise ValueError(f"Cannot determine target backend from {target!r}")
        return cls(backend=backend, arch=arch, features=frozenset(features), limits=limits or {})

    def has(self, feature: str) -> bool:
        """Return whether an explicitly declared target feature is available."""

        return feature in self.features

    def limit(self, name: str) -> int | None:
        """Return a declared target resource limit."""

        return self.limits.get(name)


@dataclass(frozen=True)
class ScheduleConstraint:
    """Named predicate used to reject invalid schedule configurations."""

    name: str
    predicate: ConstraintPredicate

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Constraint name must be a non-empty string")
        if not callable(self.predicate):
            raise TypeError("Constraint predicate must be callable")

    def accepts(self, config: Mapping[str, Any], target: TargetProfile | None) -> bool:
        return bool(self.predicate(config, target))


def _identity(value: Any) -> Any:
    return value


@dataclass(frozen=True)
class PassConfigBinding:
    """Materialize one semantic schedule field as a compiler pass config."""

    parameter: str
    key: str
    transform: ValueTransform = _identity
    omit_values: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not self.parameter.isidentifier():
            raise ValueError(f"Bound schedule parameter {self.parameter!r} must be a valid Python identifier")
        if not self.key.strip():
            raise ValueError("Pass config key must be a non-empty string")
        if not callable(self.transform):
            raise TypeError("Pass config transform must be callable")

    def materialize(self, value: Any) -> tuple[bool, Any]:
        """Return ``(emit, value)`` for a semantic field value."""

        if any(value == omitted for omitted in self.omit_values):
            return False, None
        return True, self.transform(value)


def requires_feature(
    parameter: str,
    values: Iterable[Any],
    feature: str,
) -> ScheduleConstraint:
    """Require a target feature when ``parameter`` selects one of ``values``."""

    selected_values = tuple(values)
    if not selected_values:
        raise ValueError("Feature-gated values must not be empty")

    def predicate(config: Mapping[str, Any], target: TargetProfile | None) -> bool:
        selected = any(config.get(parameter) == value for value in selected_values)
        return not selected or (target is not None and target.has(feature))

    return ScheduleConstraint(f"{parameter}_requires_{feature}", predicate)


def within_target_limit(parameter: str, limit: str, *, scale: int = 1) -> ScheduleConstraint:
    """Constrain a numeric schedule parameter by an explicit target limit."""

    if scale <= 0:
        raise ValueError("Limit scale must be positive")

    def predicate(config: Mapping[str, Any], target: TargetProfile | None) -> bool:
        if target is None or target.limit(limit) is None:
            return True
        return int(config[parameter]) * scale <= target.limit(limit)

    return ScheduleConstraint(f"{parameter}_within_{limit}", predicate)


class ScheduleSpace(list[ScheduleConfig]):
    """A deterministic, constrained Cartesian schedule space.

    Parameter and value insertion order define enumeration order.  Duplicate
    values are rejected to prevent redundant compilation.  ``max_candidates``
    guards the pre-filter Cartesian product so an accidental space explosion is
    detected before allocation or compilation.
    """

    def __init__(
        self,
        parameters: Mapping[str, Iterable[Any]],
        *,
        fixed: Mapping[str, Any] | None = None,
        constraints: Iterable[ScheduleConstraint] = (),
        pass_config_bindings: Iterable[PassConfigBinding] = (),
        target: TargetProfile | None = None,
        max_candidates: int = 100_000,
    ) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        fixed_config = dict(fixed or {})
        for name in fixed_config:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"Fixed schedule parameter {name!r} must be a valid Python identifier")
        dimensions: list[tuple[str, tuple[Any, ...]]] = []
        for name, raw_values in parameters.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"Schedule parameter {name!r} must be a valid Python identifier")
            if name in fixed_config:
                raise ValueError(f"Schedule parameter {name!r} is also fixed")
            if isinstance(raw_values, (str, bytes)):
                raise TypeError(f"Values for {name!r} must be an iterable of choices")
            values = tuple(raw_values)
            if not values:
                raise ValueError(f"Schedule parameter {name!r} has no values")
            for index, value in enumerate(values):
                if any(value == previous for previous in values[:index]):
                    raise ValueError(f"Schedule parameter {name!r} contains duplicate value {value!r}")
            dimensions.append((name, values))

        raw_cardinality = math.prod(len(values) for _, values in dimensions)
        if raw_cardinality > max_candidates:
            raise ValueError(f"Schedule space has {raw_cardinality} raw candidates, exceeding max_candidates={max_candidates}")

        constraint_list = tuple(constraints)
        for constraint in constraint_list:
            if not isinstance(constraint, ScheduleConstraint):
                raise TypeError("constraints must contain ScheduleConstraint instances")

        binding_list = tuple(pass_config_bindings)
        binding_parameters: set[str] = set()
        for binding in binding_list:
            if not isinstance(binding, PassConfigBinding):
                raise TypeError("pass_config_bindings must contain PassConfigBinding instances")
            if binding.parameter not in dict(dimensions):
                raise ValueError(f"Pass config binding refers to unknown parameter {binding.parameter!r}")
            if binding.parameter in binding_parameters:
                raise ValueError(f"Schedule parameter {binding.parameter!r} has multiple pass config bindings")
            binding_parameters.add(binding.parameter)

        accepted: list[ScheduleConfig] = []
        rejected: Counter[str] = Counter()
        names = tuple(name for name, _ in dimensions)
        domains = tuple(values for _, values in dimensions)
        for values in itertools.product(*domains):
            config = dict(fixed_config)
            config.update(zip(names, values))
            for constraint in constraint_list:
                try:
                    valid = constraint.accepts(config, target)
                except Exception as err:
                    raise ValueError(f"Constraint {constraint.name!r} failed for config {config!r}") from err
                if not valid:
                    rejected[constraint.name] += 1
                    break
            else:
                accepted.append(self._materialize_config(config, binding_list))

        super().__init__(accepted)
        self.parameters = MappingProxyType({name: values for name, values in dimensions})
        self.fixed = MappingProxyType(fixed_config)
        self.constraints = constraint_list
        self.pass_config_bindings = binding_list
        self.target = target
        self.raw_cardinality = raw_cardinality
        self.rejection_counts = MappingProxyType(dict(rejected))

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable dry-run diagnostics for this space."""

        return {
            "parameters": {name: list(values) for name, values in self.parameters.items()},
            "fixed": dict(self.fixed),
            "target": None
            if self.target is None
            else {
                "backend": self.target.backend,
                "arch": self.target.arch,
                "features": sorted(self.target.features),
                "limits": dict(self.target.limits),
            },
            "raw_cardinality": self.raw_cardinality,
            "accepted_cardinality": len(self),
            "rejection_counts": dict(self.rejection_counts),
            "pass_config_bindings": [{"parameter": binding.parameter, "key": binding.key} for binding in self.pass_config_bindings],
        }

    @staticmethod
    def _materialize_config(config: ScheduleConfig, bindings: tuple[PassConfigBinding, ...]) -> ScheduleConfig:
        if not bindings:
            return config
        materialized = dict(config)
        pass_configs = dict(materialized.get("pass_configs", {}))
        for binding in bindings:
            value = materialized.pop(binding.parameter)
            emit, pass_config_value = binding.materialize(value)
            if not emit:
                continue
            if binding.key in pass_configs:
                raise ValueError(f"Pass config binding for {binding.parameter!r} conflicts with fixed key {binding.key!r}")
            pass_configs[binding.key] = pass_config_value
        if pass_configs:
            materialized["pass_configs"] = pass_configs
        return materialized
