import threading
from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.jit.abi import prepare_tvm_ffi_caller_allocated_outputs
from tilelang.jit.kernel import JITKernel


tirx = tvm.tirx


def _stub_kernel(result_idx, total_params, caller_func):
    kernel = JITKernel.__new__(JITKernel)
    kernel.adapter = SimpleNamespace(result_idx=list(result_idx), params=[object() for _ in range(total_params)])
    kernel._caller_allocated_kernel = SimpleNamespace(torch_function=caller_func)
    kernel._caller_allocated_kernel_lock = threading.Lock()
    kernel._last_bound_outputs = None
    kernel._last_bound_output_call = None
    return kernel


def test_prepare_caller_allocated_outputs_removes_only_derived_attr():
    input_param = tirx.Var("input", "handle")
    output_param = tirx.Var("output", "handle")
    source = tirx.PrimFunc([input_param, output_param], tirx.Evaluate(0)).with_attr("tilelang_out_idx", [-1])

    prepared = prepare_tvm_ffi_caller_allocated_outputs(source)

    assert list(source.attrs["tilelang_out_idx"]) == [-1]
    assert prepared.attrs is None or "tilelang_out_idx" not in prepared.attrs
    assert prepare_tvm_ffi_caller_allocated_outputs(prepared).same_as(prepared)


def test_bind_outputs_reuses_single_trailing_output_and_callable():
    calls = []

    def caller_func(*args):
        calls.append(args)

    kernel = _stub_kernel([1], 2, caller_func)
    input_tensor = object()
    output_tensor = object()

    bound = kernel.bind_outputs(output_tensor)

    assert kernel.bind_outputs(output_tensor) is bound
    assert bound(input_tensor) is output_tensor
    assert kernel.call_into(input_tensor, out=output_tensor) is output_tensor
    assert calls == [(input_tensor, output_tensor), (input_tensor, output_tensor)]


def test_bind_outputs_reconstructs_interleaved_multi_output_arguments():
    calls = []

    def caller_func(*args):
        calls.append(args)

    kernel = _stub_kernel([3, 1], 5, caller_func)
    input_0, input_2, input_4 = object(), object(), object()
    output_3, output_1 = object(), object()

    bound = kernel.bind_outputs((output_3, output_1))
    result = bound(input_0, input_2, input_4)

    assert result == [output_3, output_1]
    assert calls == [(input_0, output_1, input_2, output_3, input_4)]


def test_bind_outputs_prefers_direct_caller_allocated_entry():
    wrapper_calls = []
    direct_calls = []
    entry_requests = []

    def wrapper(*args):
        wrapper_calls.append(args)

    def direct(*args):
        direct_calls.append(args)

    kernel = _stub_kernel([1], 2, wrapper)
    kernel._caller_allocated_kernel.adapter = SimpleNamespace(
        get_caller_allocated_call_entry=lambda: entry_requests.append(True) or direct
    )
    input_tensor = object()
    output_tensor = object()

    bound = kernel.bind_outputs(output_tensor)

    assert bound(input_tensor) is output_tensor
    assert kernel.bind_outputs(output_tensor) is bound
    assert entry_requests == [True]
    assert wrapper_calls == []
    assert direct_calls == [(input_tensor, output_tensor)]


def test_bind_outputs_validates_output_and_input_counts():
    no_output = _stub_kernel([], 1, lambda *_: None)
    with pytest.raises(ValueError, match="no callee-allocated outputs"):
        no_output.bind_outputs(object())

    multi_output = _stub_kernel([1, 2], 3, lambda *_: None)
    with pytest.raises(TypeError, match="list or tuple"):
        multi_output.bind_outputs(object())
    with pytest.raises(ValueError, match="expected 2 output buffers"):
        multi_output.bind_outputs([object()])
    with pytest.raises(TypeError, match="cannot be None"):
        multi_output.bind_outputs([object(), None])

    bound = multi_output.bind_outputs([object(), object()])
    with pytest.raises(ValueError, match="expected 1 inputs"):
        bound()


def test_caller_allocated_companion_is_compiled_once(monkeypatch):
    class FakePrimFunc:
        attrs = {"tilelang_out_idx": [-1]}

        def __init__(self, stripped=False):
            self.stripped = stripped

        def without_attr(self, attr_key):
            assert attr_key == "tilelang_out_idx"
            return FakePrimFunc(stripped=True)

    kernel = _stub_kernel([1], 2, lambda *_: None)
    kernel._caller_allocated_kernel = None
    kernel.prim_func = FakePrimFunc()
    kernel.target = "cuda"
    kernel.target_host = "c"
    kernel.execution_backend = "tvm_ffi"
    kernel.verbose = False
    kernel.pass_configs = {"example": True}
    kernel.compile_flags = ["--use_fast_math"]

    companion = SimpleNamespace(torch_function=lambda *_: None)
    cached_calls = []

    def fake_cached(**kwargs):
        cached_calls.append(kwargs)
        return companion

    import tilelang.cache as cache_module

    monkeypatch.setattr(cache_module, "cached", fake_cached)

    assert kernel._get_caller_allocated_kernel() is companion
    assert kernel._get_caller_allocated_kernel() is companion
    assert len(cached_calls) == 1
    assert cached_calls[0]["func"].stripped
    assert cached_calls[0]["out_idx"] is None
    assert cached_calls[0]["execution_backend"] == "tvm_ffi"
    assert cached_calls[0]["pass_configs"] == {"example": True}
    assert cached_calls[0]["compile_flags"] == ["--use_fast_math"]
