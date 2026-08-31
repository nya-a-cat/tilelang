"""Tests for auto-tuner cache-key identity."""

import inspect
import threading
from dataclasses import replace

import pytest

from tilelang.autotuner import AutoTuner
from tilelang.autotuner.param import CompileArgs, ProfileArgs


def _kernel(block_size=128):
    return block_size


def _cache_key(compile_args=None, profile_args=None):
    tuner = AutoTuner(_kernel, configs=[{"block_size": 128}])
    tuner.compile_args = compile_args or CompileArgs()
    tuner.profile_args = profile_args or ProfileArgs()
    return tuner.generate_cache_key(inspect.signature(_kernel).parameters, {})


def test_cache_key_includes_output_indices():
    base_args = CompileArgs(out_idx=[0])

    assert _cache_key(compile_args=base_args) != _cache_key(compile_args=replace(base_args, out_idx=[1]))


def test_cache_key_includes_profile_validation_and_input_behavior():
    base_args = ProfileArgs(skip_check=False, cache_input_tensors=False)

    variants = (
        replace(base_args, skip_check=True),
        replace(base_args, cache_input_tensors=True),
    )

    base_key = _cache_key(profile_args=base_args)
    assert all(base_key != _cache_key(profile_args=variant) for variant in variants)


def test_cache_key_is_disabled_for_profile_callbacks():
    lock = threading.Lock()

    def callback(value):
        with lock:
            return value

    for callback_field in ("ref_prog", "supply_prog", "manual_check_prog"):
        assert _cache_key(profile_args=ProfileArgs(**{callback_field: callback})) is None


def test_compile_budget_selects_an_exact_prefix_without_mutating_configs():
    configs = [{"block_size": size} for size in (64, 128, 256)]

    selected = AutoTuner._select_configs(configs, max_trials=2)

    assert selected == configs[:2]
    assert configs == [{"block_size": 64}, {"block_size": 128}, {"block_size": 256}]


@pytest.mark.parametrize("max_trials", [0, -1, True, 1.5, "2"])
def test_compile_budget_rejects_invalid_limits(max_trials):
    with pytest.raises(ValueError, match="max_trials must be a positive integer or None"):
        AutoTuner._select_configs([{"block_size": 128}], max_trials=max_trials)


def test_compile_budget_uses_the_selected_candidates_in_cache_identity():
    configs = [{"block_size": size} for size in (64, 128, 256)]
    tuner = AutoTuner(_kernel, configs=configs)
    parameters = inspect.signature(_kernel).parameters

    two_trial_key = tuner.generate_cache_key(parameters, {}, configs=tuner._select_configs(configs, max_trials=2))
    full_key = tuner.generate_cache_key(parameters, {}, configs=tuner._select_configs(configs, max_trials=None))

    assert two_trial_key != full_key
