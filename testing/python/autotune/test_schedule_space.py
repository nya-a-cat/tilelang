"""CPU-only tests for deterministic autotuning schedule spaces."""

from __future__ import annotations

import json

import pytest

from tilelang.autotuner import (
    PassConfigBinding,
    ScheduleConstraint,
    ScheduleSpace,
    TargetProfile,
    estimated_within_target_limit,
    requires_feature,
    within_target_limit,
)
from benchmark.research.cross_gemm_schedule_space import (
    CrossGemmWorkload,
    cross_gemm_accumulator_registers,
    cross_gemm_schedule_space,
    cross_gemm_search_summary,
    cross_gemm_shared_bytes,
    ranked_cross_gemm_schedules,
)
from benchmark.mamba2.schedule_spaces import (
    LEGACY_PARAMETER_NAMES,
    executable_schedule_space,
    legacy_schedule_space,
    research_schedule_space,
)


def test_cartesian_space_preserves_deterministic_order_and_json_compatibility():
    space = ScheduleSpace({"block_M": [64, 128], "num_stages": [1, 2]}, fixed={"threads": 128})

    assert space == [
        {"threads": 128, "block_M": 64, "num_stages": 1},
        {"threads": 128, "block_M": 64, "num_stages": 2},
        {"threads": 128, "block_M": 128, "num_stages": 1},
        {"threads": 128, "block_M": 128, "num_stages": 2},
    ]
    assert json.loads(json.dumps(space)) == space
    assert space.summary()["raw_cardinality"] == 4
    assert space.summary()["accepted_cardinality"] == 4


def test_named_constraints_filter_and_report_first_rejection():
    space = ScheduleSpace(
        {"block_M": [64, 128], "num_stages": [1, 2, 3]},
        constraints=[
            ScheduleConstraint("small_tiles_use_two_stages", lambda config, _target: config["block_M"] != 64 or config["num_stages"] <= 2),
            ScheduleConstraint("large_tiles_use_one_stage", lambda config, _target: config["block_M"] != 128 or config["num_stages"] == 1),
        ],
    )

    assert space == [
        {"block_M": 64, "num_stages": 1},
        {"block_M": 64, "num_stages": 2},
        {"block_M": 128, "num_stages": 1},
    ]
    assert space.rejection_counts == {"small_tiles_use_two_stages": 1, "large_tiles_use_one_stage": 2}


def test_explicit_target_features_gate_backend_specific_choices():
    target = TargetProfile.from_target(
        {"kind": "cuda", "arch": "sm_90a"},
        features={"persistent_grid", "bulk_copy"},
    )
    space = ScheduleSpace(
        {
            "grid_mapping": ["spatial", "persistent"],
            "copy_policy": ["auto", "bulk"],
            "threads": [128],
        },
        target=target,
        constraints=[
            requires_feature("grid_mapping", ["persistent"], "persistent_grid"),
            requires_feature("copy_policy", ["bulk"], "bulk_copy"),
        ],
    )

    assert len(space) == 4
    assert target.backend == "cuda"
    assert target.arch == "sm_90a"


def test_missing_target_feature_rejects_only_gated_choice():
    target = TargetProfile.from_target("cuda -arch=sm_80")
    space = ScheduleSpace(
        {"copy_policy": ["auto", "bulk"]},
        target=target,
        constraints=[requires_feature("copy_policy", ["bulk"], "bulk_copy")],
    )

    assert space == [{"copy_policy": "auto"}]
    assert space.rejection_counts == {"copy_policy_requires_bulk_copy": 1}


def test_rocm_target_string_is_parsed_without_importing_a_device_runtime():
    target = TargetProfile.from_target("rocm -mcpu=gfx942", features={"mfma"})

    assert target.backend == "rocm"
    assert target.arch == "gfx942"
    assert target.has("mfma")


def test_target_resource_limit_filters_threads():
    target = TargetProfile("webgpu", limits={"max_threads_per_block": 256})
    space = ScheduleSpace(
        {"threads": [128, 256, 512]},
        target=target,
        constraints=[within_target_limit("threads", "max_threads_per_block")],
    )

    assert space == [{"threads": 128}, {"threads": 256}]


def test_derived_target_resource_limit_filters_and_reports_estimates():
    target = TargetProfile("cuda", "sm_75", limits={"max_shared_bytes_per_block": 64})
    space = ScheduleSpace(
        {"tile": [16, 32, 64]},
        target=target,
        constraints=[
            estimated_within_target_limit(
                "shared_bytes",
                "max_shared_bytes_per_block",
                lambda config, _target: int(config["tile"]) * 2,
            )
        ],
    )

    assert space == [{"tile": 16}, {"tile": 32}]
    assert space.rejection_counts == {"shared_bytes_within_max_shared_bytes_per_block": 1}


def test_semantic_fields_materialize_into_per_candidate_pass_configs():
    space = ScheduleSpace(
        {
            "threads": [128],
            "copy_policy": ["auto", "sync", "async"],
            "warp_specialization": [False, True],
        },
        fixed={"pass_configs": {"tl.enable_fast_math": True}},
        pass_config_bindings=[
            PassConfigBinding(
                "copy_policy",
                "tl.enable_async_copy",
                transform=lambda policy: policy == "async",
                omit_values=("auto",),
            ),
            PassConfigBinding(
                "warp_specialization",
                "tl.disable_warp_specialized",
                transform=lambda enabled: not enabled,
            ),
        ],
    )

    assert len(space) == 6
    assert space[0] == {
        "threads": 128,
        "pass_configs": {"tl.enable_fast_math": True, "tl.disable_warp_specialized": True},
    }
    assert space[-1] == {
        "threads": 128,
        "pass_configs": {
            "tl.enable_fast_math": True,
            "tl.enable_async_copy": True,
            "tl.disable_warp_specialized": False,
        },
    }


def test_mamba_legacy_space_is_preserved_exactly():
    space = legacy_schedule_space()

    assert len(space) == 90
    assert space[0] == {"block_M": 64, "block_N": 32, "block_K": 64, "block_Dstate": 128, "num_stages": 1}
    assert space[-1] == {"block_M": 256, "block_N": 64, "block_K": 256, "block_Dstate": 128, "num_stages": 5}


def test_mamba_research_space_contains_every_legacy_configuration():
    target = TargetProfile(
        "cuda",
        "sm_90a",
        features=frozenset({"persistent_grid", "async_copy", "warp_specialization"}),
        limits={"max_threads_per_block": 1024},
    )
    legacy = legacy_schedule_space()
    research = research_schedule_space(target)
    legacy_projection = {tuple(config[name] for name in LEGACY_PARAMETER_NAMES) for config in legacy}
    research_projection = {tuple(config[name] for name in LEGACY_PARAMETER_NAMES) for config in research}

    assert research.raw_cardinality == 2_880
    assert len(research) == 2_880
    assert legacy_projection <= research_projection


def test_mamba_research_space_capability_filter_is_conservative():
    target = TargetProfile("cuda", "sm_80", limits={"max_threads_per_block": 128})
    research = research_schedule_space(target)

    assert len(research) == 180
    assert {config["threads"] for config in research} == {128}
    assert {config["schedule_grid_mapping"] for config in research} == {"spatial"}
    assert {config["schedule_copy_policy"] for config in research} == {"auto"}
    assert {config["schedule_warp_specialization"] for config in research} == {False}


def test_mamba_executable_space_only_emits_kernel_args_and_pass_configs():
    target = TargetProfile(
        "cuda",
        "sm_90a",
        features=frozenset({"async_copy", "warp_specialization"}),
        limits={"max_threads_per_block": 1024},
    )
    space = executable_schedule_space(target)

    assert space.raw_cardinality == 720
    assert len(space) == 720
    assert all(not any(key.startswith("schedule_") for key in config) for config in space)
    assert {config["pass_configs"]["tl.enable_async_copy"] for config in space} == {False, True}
    assert {config["pass_configs"]["tl.disable_warp_specialized"] for config in space} == {False, True}


def test_cross_gemm_space_filters_resources_and_ranks_reuse_deterministically():
    workload = CrossGemmWorkload(M=256, K=896, N=4864)
    target = TargetProfile(
        "cuda",
        "sm_75",
        limits={
            "max_threads_per_block": 1024,
            "max_shared_bytes_per_block": 64 * 1024,
            "max_registers_per_thread": 255,
            "multiprocessor_count": 40,
        },
    )
    space = cross_gemm_schedule_space(workload, target)
    ranked = ranked_cross_gemm_schedules(workload, target)
    summary = cross_gemm_search_summary(workload, target, top_k=3)

    assert space.raw_cardinality == 162
    assert len(space) == 129
    assert all(cross_gemm_shared_bytes(config, workload) <= 64 * 1024 for config in space)
    assert all(cross_gemm_accumulator_registers(config, workload) <= 255 for config in space)
    assert ranked[0] == {
        "block_M": 128,
        "block_N": 128,
        "block_K": 32,
        "num_stages": 0,
        "threads": 256,
    }
    assert summary["top_candidates"][0]["config"] == ranked[0]
    assert json.loads(json.dumps(summary)) == summary


def test_space_explosion_is_rejected_before_materialization():
    with pytest.raises(ValueError, match="exceeding max_candidates=100"):
        ScheduleSpace({"x": range(11), "y": range(10)}, max_candidates=100)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"bad-name": [1]}, "valid Python identifier"),
        ({"threads": []}, "has no values"),
        ({"threads": [128, 128]}, "duplicate value"),
        ({"copy_policy": "auto"}, "iterable of choices"),
    ],
)
def test_invalid_spaces_fail_early(parameters, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ScheduleSpace(parameters)
