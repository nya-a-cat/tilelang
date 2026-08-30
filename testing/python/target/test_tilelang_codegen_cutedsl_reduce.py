import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.cuda.target import normalize_cutedsl_target


def _lower_cutedsl_partial_reduce():
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_90"})
    assert target is not None

    @T.prim_func
    def prog(A: T.Tensor((1, 512), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            x_frag = T.alloc_fragment((1, 512), "float32")
            sum_frag = T.alloc_fragment((1,), "float32")
            T.annotate_layout(
                {
                    x_frag: T.Fragment(x_frag.shape, forward_fn=lambda i, j: (j // 8, j % 8)),
                    sum_frag: T.Fragment(sum_frag.shape, forward_fn=lambda i, rep: (rep, 0), replicate=64),
                }
            )
            for i, j in T.Parallel(1, 512):
                x_frag[i, j] = A[i, j]
            T.reduce_sum(x_frag, sum_frag, dim=1)
            for i in T.Parallel(1):
                B[i] = sum_frag[i]

    with target:
        return lower(prog.with_attr("global_symbol", "main"), target=target)


def _lower_cutedsl_full_reduce(batch=1, rows=None, arch="sm_90", threads=128):
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": arch})
    assert target is not None
    rows = batch if rows is None else rows
    width = threads * 8

    @T.prim_func
    def prog(
        A: T.Tensor((rows, width), "float32"),
        B: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(1, threads=threads):
            x_frag = T.alloc_fragment((rows, width), "float32")
            sum_frag = T.alloc_fragment((rows,), "float32")
            T.copy(A, x_frag)
            T.reduce_sum(x_frag, sum_frag, dim=1, batch=batch)
            T.copy(sum_frag, B)

    with target:
        return lower(prog.with_attr("global_symbol", "main"), target=target)


def _dynamic_shared_bytes(artifact):
    values = [int(func.attrs.get("dyn_shared_memory_buf", 0)) for func in artifact.device_mod.functions.values()]
    assert len(values) == 1
    return values[0]


def test_cutedsl_codegen_partial_reduce_named_barrier():
    """The partial scalar AllReduce uses its exact participant count."""
    artifact = _lower_cutedsl_partial_reduce()
    assert "tl.NamedBarrier(64)" in artifact.kernel_source


@pytest.mark.parametrize("threads", [64, 128])
def test_cutedsl_codegen_full_reduce_uses_hierarchical_path(threads):
    artifact = _lower_cutedsl_full_reduce(threads=threads)
    warps = threads // 32
    expected = f"tl.AllReduce(tl.SumOp, {threads}, 1, 0, tl.NamedBarrier({threads}), 1, 0, True).run"
    assert expected in artifact.kernel_source
    assert _dynamic_shared_bytes(artifact) == warps * 4


def test_cutedsl_codegen_batched_full_reduce_uses_run_batch_alias():
    artifact = _lower_cutedsl_full_reduce(batch=4)
    assert "tl.NamedBarrier(128), 4, 4, True).run_batch" in artifact.kernel_source
    assert _dynamic_shared_bytes(artifact) == 64


def test_cutedsl_blackwell_batch_allreduce_uses_supported_scalar_interface():
    artifact = _lower_cutedsl_full_reduce(batch=4, arch="sm_100a")
    source = artifact.kernel_source
    assert "f32x2" not in source, source
    assert "tl.NamedBarrier(128), 4, 4, True).run_batch" in source, source
    assert _dynamic_shared_bytes(artifact) == 64


def test_cutedsl_consecutive_batch_chunks_sync_workspace_reuse():
    artifact = _lower_cutedsl_full_reduce(batch=4, rows=8)
    source = artifact.kernel_source
    assert source.count("tl.AllReduce") == 2, source
    assert source.count("tl.NamedBarrier(128), 4, 4, True).run_batch") == 2, source
    assert source.count("tl.sync_threads()") == 1, source
    assert _dynamic_shared_bytes(artifact) == 64


if __name__ == "__main__":
    tilelang.testing.main()
