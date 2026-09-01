"""CPU-only tests for target-driven CUDA scheduling capabilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.carver.arch import cuda as carver_cuda
from tilelang.carver import utils as carver_utils
from tilelang.carver.roller.policy.tensorcore import TensorCorePolicy
from tilelang.contrib import nvcc
from tilelang.cuda.target import target_has_reg_reconfiguration

Target = tvm.target.Target


class _TaggedNode:
    def __init__(self, **tags):
        self.tags = tags

    def get_tag(self, name):
        return self.tags.get(name)


def _cuda_arch(sm_version: int, **attrs):
    arch = carver_cuda.CUDA.__new__(carver_cuda.CUDA)
    arch.sm_version = sm_version
    arch.compute_capability = f"sm_{sm_version}"
    for name, value in attrs.items():
        setattr(arch, name, value)
    return arch


def _policy(sm_version: int, **tags) -> TensorCorePolicy:
    policy = TensorCorePolicy.__new__(TensorCorePolicy)
    policy.arch = _cuda_arch(sm_version)
    policy.tags = {}
    policy.ordered_nodes = [_TaggedNode(**tags)]
    policy._legalize_info()
    return policy


@pytest.mark.parametrize("sm_version", [80, 86, 87, 89, 90, 100, 103, 120])
def test_ampere_and_newer_default_to_async_pipeline(sm_version):
    policy = _policy(sm_version)
    assert policy.pipeline_stage == 2
    assert policy.use_async_copy is True


@pytest.mark.parametrize("sm_version", [70, 75])
def test_pre_ampere_keeps_single_stage_synchronous_copy(sm_version):
    policy = _policy(sm_version)
    assert policy.pipeline_stage == 1
    assert policy.use_async_copy is False


def test_explicit_false_disables_async_copy():
    policy = _policy(90, pipeline_stage=2, use_async_copy=False)
    assert policy.pipeline_stage == 2
    assert policy.use_async_copy is False


def test_explicit_pipeline_stage_is_preserved():
    policy = _policy(100, pipeline_stage=4, use_async_copy=True)
    assert policy.pipeline_stage == 4
    assert policy.use_async_copy is True


def test_conflicting_multi_node_policy_tags_are_rejected():
    policy = TensorCorePolicy.__new__(TensorCorePolicy)
    policy.arch = _cuda_arch(90)
    policy.tags = {}
    policy.ordered_nodes = [_TaggedNode(pipeline_stage=2), _TaggedNode(pipeline_stage=3)]
    with pytest.raises(ValueError, match="conflicting pipeline_stage"):
        policy._legalize_info()


@pytest.mark.parametrize("sm_version", [80, 86, 89])
def test_pre_hopper_fused_graph_avoids_register_heavy_double_buffering(sm_version):
    policy = TensorCorePolicy.__new__(TensorCorePolicy)
    policy.arch = _cuda_arch(sm_version)
    policy.tags = {}
    policy.ordered_nodes = [
        _TaggedNode(pipeline_stage=2, use_async_copy=True),
        _TaggedNode(pipeline_stage=2, use_async_copy=True),
    ]
    policy._legalize_info()
    assert policy.pipeline_stage == 1
    assert policy.use_async_copy is False


@pytest.mark.parametrize("sm_version", [90, 100, 103, 120])
def test_hopper_and_newer_fused_graph_keeps_tma_double_buffering(sm_version):
    policy = TensorCorePolicy.__new__(TensorCorePolicy)
    policy.arch = _cuda_arch(sm_version)
    policy.tags = {}
    policy.ordered_nodes = [
        _TaggedNode(pipeline_stage=2, use_async_copy=True),
        _TaggedNode(pipeline_stage=2, use_async_copy=True),
    ]
    policy._legalize_info()
    assert policy.pipeline_stage == 2
    assert policy.use_async_copy is True


def test_explicit_fused_pipeline_override_is_authoritative():
    policy = TensorCorePolicy.__new__(TensorCorePolicy)
    policy.arch = _cuda_arch(89)
    policy.tags = {"pipeline_stage": 2, "use_async_copy": True}
    policy.ordered_nodes = [_TaggedNode(), _TaggedNode()]
    policy._legalize_info()
    assert policy.pipeline_stage == 2
    assert policy.use_async_copy is True


@pytest.mark.parametrize(
    ("sm_version", "target_arch", "expected_stage", "expected_async"),
    [(89, "sm_89", 1, False), (100, "sm_100a", 2, True)],
)
def test_flash_attention_graph_uses_arch_adaptive_pipeline(monkeypatch, sm_version, target_arch, expected_stage, expected_async):
    """Exercise the fused policy through the real two-MMA attention graph."""

    from tilelang.carver import matmul_analysis
    from tilelang.carver.template.flashattention import FlashAttentionTemplate

    arch = _cuda_arch(sm_version)
    arch.target = Target({"kind": "cuda", "arch": target_arch})
    monkeypatch.setattr(matmul_analysis, "get_arch", lambda _target: arch)

    template = object.__new__(FlashAttentionTemplate)
    template._arch = arch
    template.batch_size = 1
    template.num_heads = 1
    template.head_dim = 64
    template.seq_length = 128
    template.seq_kv_length = 128
    template.is_causal = False
    template.in_dtype = "float16"
    template.out_dtype = "float16"
    template.accum_dtype = "float32"
    template.initialize_function()

    policy = TensorCorePolicy.from_output_nodes(template.output_nodes, arch)

    assert len(policy.ordered_nodes) == 2
    assert policy.pipeline_stage == expected_stage
    assert policy.use_async_copy is expected_async


def test_target_arch_drives_cuda_capability(monkeypatch):
    fake_device = SimpleNamespace(exist=True, compute_version="8.0", multi_processor_count=108, warp_size=32)
    monkeypatch.setattr(carver_cuda.tvm.runtime, "cuda", lambda _device_id: fake_device)
    monkeypatch.setattr(carver_cuda.cuda_driver, "get_device_name", lambda: "fake-sm80-host")
    monkeypatch.setattr(carver_cuda.cuda_driver, "get_shared_memory_per_block", lambda: 163840)
    monkeypatch.setattr(carver_cuda.cuda_driver, "get_cuda_device_properties", lambda: None)
    monkeypatch.setattr(carver_cuda.cuda_driver, "get_persisting_l2_cache_max_size", lambda: 0)

    arch = carver_cuda.CUDA(Target({"kind": "cuda", "arch": "sm_100a"}))

    assert arch.sm_version == 100
    assert arch.compute_capability == "100"


def test_blackwell_legacy_tensorcore_precision_is_supported():
    arch = _cuda_arch(100)
    assert carver_cuda.is_blackwell_arch(arch)
    assert carver_cuda.is_tensorcore_supported_precision("float16", "float32", arch)
    assert carver_cuda.is_tensorcore_supported_precision("bfloat16", "float32", arch)
    assert carver_cuda.is_tensorcore_supported_precision("int8", "int32", arch)


@pytest.mark.parametrize("sm_version", [80, 86, 89, 90, 100, 103, 120])
@pytest.mark.parametrize("spelling", ["custom[tfloat32]", "tfloat32", "tf32"])
def test_ampere_and_newer_support_tf32_tensorcore_scheduling(sm_version, spelling):
    assert carver_cuda.is_tensorcore_supported_precision(spelling, "float32", _cuda_arch(sm_version))


@pytest.mark.parametrize("sm_version", [70, 75])
def test_pre_ampere_rejects_tf32_tensorcore_scheduling(sm_version):
    assert not carver_cuda.is_tensorcore_supported_precision("custom[tfloat32]", "float32", _cuda_arch(sm_version))


@pytest.mark.parametrize("arch", ["sm_100", "sm_100a", "sm_103a", "sm_120", "sm_120a"])
def test_have_tensorcore_parses_three_digit_targets(arch):
    assert nvcc.have_tensorcore(target=Target({"kind": "cuda", "arch": arch}))


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("sm_89", False),
        ("sm_90", False),
        ("sm_90a", True),
        ("sm_100", False),
        ("sm_100f", True),
        ("sm_100a", True),
        ("sm_120", False),
        ("sm_120f", True),
        ("sm_120a", True),
    ],
)
def test_reg_reconfiguration_requires_feature_specific_target(arch, expected):
    assert target_has_reg_reconfiguration(Target({"kind": "cuda", "arch": arch})) is expected


def test_tensorcore_only_hint_path_uses_primfunc_factory(monkeypatch):
    from tvm import te

    source = te.placeholder((16, 16), name="source", dtype="float16")
    result = te.compute((16, 16), lambda i, j: source[i, j] + 1, name="result")
    func = te.create_prim_func([source, result])
    fake_arch = SimpleNamespace(target=Target({"kind": "cuda", "arch": "sm_100a"}))
    fake_hint = object()

    class _FakePolicy:
        def emit_config(self, topk):
            assert topk == 1
            return [fake_hint]

    called = {}

    def fake_from_prim_func(*, func, arch, tags):
        called.update(func=func, arch=arch, tags=tags)
        return _FakePolicy()

    monkeypatch.setattr(carver_utils, "get_tensorized_func_and_tags", lambda *_args, **_kwargs: (func, {"tensorcore_config": [0, 1]}))
    monkeypatch.setattr(carver_utils.TensorCorePolicy, "from_prim_func", fake_from_prim_func)

    hints = carver_utils.get_roller_hints_from_func(func, fake_arch, topk=1, tensorcore_only=True)

    assert hints == [fake_hint]
    assert called == {"func": func, "arch": fake_arch, "tags": {"tensorcore_config": [0, 1]}}
