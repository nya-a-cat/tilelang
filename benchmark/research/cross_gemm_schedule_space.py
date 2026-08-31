"""Deterministic resource-aware schedules for Cross-GEMM experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from tilelang.autotuner import (
    ScheduleConstraint,
    ScheduleSpace,
    TargetProfile,
    estimated_within_target_limit,
    within_target_limit,
)


@dataclass(frozen=True)
class CrossGemmWorkload:
    """Static dimensions and storage types for two-GEMM activation fusion."""

    M: int
    K: int
    N: int
    operand_bits: int = 16
    accumulator_bits: int = 32
    gemm_count: int = 2

    def __post_init__(self) -> None:
        values = (self.M, self.K, self.N, self.operand_bits, self.accumulator_bits, self.gemm_count)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("Cross-GEMM workload dimensions, bit widths, and gemm_count must be positive integers")


def _staged_depth(config: Mapping[str, Any]) -> int:
    num_stages = int(config["num_stages"])
    return 1 if num_stages == 0 else num_stages


def cross_gemm_shared_bytes(config: Mapping[str, Any], workload: CrossGemmWorkload) -> int:
    """Conservative shared-memory footprint including staged GEMM inputs."""

    block_m = int(config["block_M"])
    block_n = int(config["block_N"])
    block_k = int(config["block_K"])
    operand_bytes = math.ceil(workload.operand_bits / 8)
    staged_inputs = _staged_depth(config) * (block_m * block_k + workload.gemm_count * block_n * block_k)
    output_tile = block_m * block_n
    return operand_bytes * (staged_inputs + output_tile)


def cross_gemm_accumulator_registers(config: Mapping[str, Any], workload: CrossGemmWorkload) -> int:
    """Upper-bound accumulator registers held by one thread."""

    outputs = workload.gemm_count * int(config["block_M"]) * int(config["block_N"])
    registers = math.ceil(outputs * workload.accumulator_bits / 32 / int(config["threads"]))
    return registers


def cross_gemm_accumulator_registers_per_block(config: Mapping[str, Any], workload: CrossGemmWorkload) -> int:
    """Upper-bound accumulator registers allocated across one thread block."""

    return cross_gemm_accumulator_registers(config, workload) * int(config["threads"])


def _tiles_divide_workload(config: Mapping[str, Any], workload: CrossGemmWorkload) -> bool:
    return (
        workload.M % int(config["block_M"]) == 0 and workload.N % int(config["block_N"]) == 0 and workload.K % int(config["block_K"]) == 0
    )


def cross_gemm_schedule_space(workload: CrossGemmWorkload, target: TargetProfile) -> ScheduleSpace:
    """Build the executable Cross-GEMM space before compiling any kernel."""

    return ScheduleSpace(
        {
            "block_M": (32, 64, 128),
            "block_N": (32, 64, 128),
            "block_K": (16, 32, 64),
            "num_stages": (0, 2, 3),
            "threads": (128, 256),
        },
        target=target,
        constraints=(
            ScheduleConstraint("tiles_divide_workload", lambda config, _target: _tiles_divide_workload(config, workload)),
            within_target_limit("threads", "max_threads_per_block"),
            estimated_within_target_limit(
                "shared_bytes",
                "max_shared_bytes_per_block",
                lambda config, _target: cross_gemm_shared_bytes(config, workload),
            ),
            estimated_within_target_limit(
                "accumulator_registers",
                "max_registers_per_thread",
                lambda config, _target: cross_gemm_accumulator_registers(config, workload),
            ),
            estimated_within_target_limit(
                "accumulator_registers_per_block",
                "max_registers_per_block",
                lambda config, _target: cross_gemm_accumulator_registers_per_block(config, workload),
            ),
        ),
        max_candidates=1_000,
    )


def cross_gemm_schedule_estimate(
    config: Mapping[str, Any],
    workload: CrossGemmWorkload,
    target: TargetProfile,
) -> dict[str, int | float]:
    """Return explainable ranking features without predicting wall time."""

    block_m = int(config["block_M"])
    block_n = int(config["block_N"])
    grid_ctas = math.ceil(workload.M / block_m) * math.ceil(workload.N / block_n)
    multiprocessors = target.limit("multiprocessor_count")
    shared_bytes = cross_gemm_shared_bytes(config, workload)
    accumulator_registers = cross_gemm_accumulator_registers(config, workload)
    accumulator_registers_per_block = cross_gemm_accumulator_registers_per_block(config, workload)
    return {
        "grid_ctas": grid_ctas,
        "waves": grid_ctas / multiprocessors if multiprocessors else 0.0,
        "operand_bytes_per_k_output": math.ceil(workload.operand_bits / 8) * (1.0 / block_n + workload.gemm_count / block_m),
        "shared_bytes": shared_bytes,
        "shared_fraction": shared_bytes / target.limit("max_shared_bytes_per_block") if target.limit("max_shared_bytes_per_block") else 0.0,
        "accumulator_registers_per_thread": accumulator_registers,
        "accumulator_register_fraction": accumulator_registers / target.limit("max_registers_per_thread")
        if target.limit("max_registers_per_thread")
        else 0.0,
        "accumulator_registers_per_block": accumulator_registers_per_block,
        "accumulator_register_block_fraction": accumulator_registers_per_block / target.limit("max_registers_per_block")
        if target.limit("max_registers_per_block")
        else 0.0,
    }


def ranked_cross_gemm_schedules(
    workload: CrossGemmWorkload,
    target: TargetProfile,
) -> list[dict[str, Any]]:
    """Order valid schedules by reuse, pipeline support, and resource pressure."""

    space = cross_gemm_schedule_space(workload, target)

    def priority(config: Mapping[str, Any]) -> tuple[float | int, ...]:
        estimate = cross_gemm_schedule_estimate(config, workload, target)
        pipeline_preference = 0 if (int(config["num_stages"]) == 0) == (not target.has("async_copy")) else 1
        resource_pressure = max(
            float(estimate["shared_fraction"]),
            float(estimate["accumulator_register_fraction"]),
            float(estimate["accumulator_register_block_fraction"]),
        )
        return (
            float(estimate["operand_bytes_per_k_output"]),
            pipeline_preference,
            -int(config["block_K"]),
            resource_pressure,
            -int(config["threads"]),
        )

    return sorted((dict(config) for config in space), key=priority)


def cross_gemm_search_summary(
    workload: CrossGemmWorkload,
    target: TargetProfile,
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    """Produce a JSON-ready audit record for a bounded tuning launch."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    space = cross_gemm_schedule_space(workload, target)
    ranked = ranked_cross_gemm_schedules(workload, target)
    return {
        "workload": {
            "M": workload.M,
            "K": workload.K,
            "N": workload.N,
            "operand_bits": workload.operand_bits,
            "accumulator_bits": workload.accumulator_bits,
            "gemm_count": workload.gemm_count,
        },
        "space": space.summary(),
        "top_candidates": [
            {
                "config": config,
                "estimate": cross_gemm_schedule_estimate(config, workload, target),
            }
            for config in ranked[:top_k]
        ],
    }
