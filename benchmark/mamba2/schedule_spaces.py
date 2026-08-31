"""Schedule-space definitions for the Mamba-2 chunk-scan benchmark."""

from __future__ import annotations

from tilelang.autotuner import PassConfigBinding, ScheduleSpace, TargetProfile, requires_feature, within_target_limit

LEGACY_PARAMETER_NAMES = ("block_M", "block_N", "block_K", "block_Dstate", "num_stages")

_LEGACY_PARAMETERS = {
    "block_M": (64, 128, 256),
    "block_N": (32, 64),
    "block_K": (64, 128, 256),
    "block_Dstate": (128,),
    "num_stages": (1, 2, 3, 4, 5),
}


def legacy_schedule_space() -> ScheduleSpace:
    """Return the exact 90 configurations used by the published benchmark."""

    return ScheduleSpace(_LEGACY_PARAMETERS)


def research_schedule_space(target: TargetProfile) -> ScheduleSpace:
    """Return a compiler-facing dry-run space extending the legacy baseline.

    The ``schedule_*`` fields describe decisions that still need late IR
    materialization.  This function is intentionally kept separate from the
    executable benchmark until those transformations exist.
    """

    return ScheduleSpace(
        {
            **_LEGACY_PARAMETERS,
            "threads": (128, 256),
            "schedule_grid_mapping": ("spatial", "persistent"),
            "schedule_l2_swizzle_size": (1, 8),
            "schedule_copy_policy": ("auto", "async"),
            "schedule_warp_specialization": (False, True),
        },
        target=target,
        constraints=(
            requires_feature("schedule_grid_mapping", ("persistent",), "persistent_grid"),
            requires_feature("schedule_copy_policy", ("async",), "async_copy"),
            requires_feature("schedule_warp_specialization", (True,), "warp_specialization"),
            within_target_limit("threads", "max_threads_per_block"),
        ),
        max_candidates=10_000,
    )


def executable_schedule_space(target: TargetProfile) -> ScheduleSpace:
    """Return the currently materializable subset of the research space.

    Copy and warp-specialization choices lower into per-candidate pass configs;
    all remaining fields are accepted directly by ``chunk_scan_fwd``.
    """

    return ScheduleSpace(
        {
            **_LEGACY_PARAMETERS,
            "threads": (128, 256),
            "schedule_copy_policy": ("sync", "async"),
            "schedule_warp_specialization": (False, True),
        },
        target=target,
        constraints=(
            requires_feature("schedule_copy_policy", ("async",), "async_copy"),
            requires_feature("schedule_warp_specialization", (True,), "warp_specialization"),
            within_target_limit("threads", "max_threads_per_block"),
        ),
        pass_config_bindings=(
            PassConfigBinding(
                "schedule_copy_policy",
                "tl.enable_async_copy",
                transform=lambda policy: policy == "async",
            ),
            PassConfigBinding(
                "schedule_warp_specialization",
                "tl.disable_warp_specialized",
                transform=lambda enabled: not enabled,
            ),
        ),
        max_candidates=10_000,
    )
