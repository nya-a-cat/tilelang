"""GPU checks for the internal ownership-preserving conversion primitive."""

import pytest
import torch
import tilelang
from tilelang import language as T
from tilelang.layout import Fragment
import tvm_ffi


def layouts(n, kind):
    source = Fragment((n,), forward_fn=lambda i: (i % 128, i // 128))
    if kind == "register":
        target = Fragment((n,), forward_fn=lambda i: (i % 128, (n - 1 - i) // 128))
    elif kind == "shuffle":
        target = Fragment((n,), forward_fn=lambda i: ((i % 128 // 32) * 32 + (i + 1) % 32, i // 128))
    elif kind == "shared":
        target = Fragment((n,), forward_fn=lambda i: ((i + 32) % 128, i // 128))
    else:
        target = Fragment((n,), forward_fn=lambda i, rep: (rep, i), replicate=128)
    return source, target


@pytest.mark.parametrize("kind,expected", [("register", 0), ("shuffle", 1), ("shared", 2), ("replicated", 2)])
def test_conversion_ownership_classification(kind, expected):
    source, target = layouts(256, kind)
    assert tvm_ffi.get_global_func("tl.layout.conversion_kind")(source, target, 128) == expected


@pytest.mark.parametrize("dtype", ["float16", "float32", "int32", "float64"])
@pytest.mark.parametrize("kind", ["register", "shuffle", "shared", "replicated"])
@pytest.mark.parametrize("n", [160, 256])
def test_conversion_roundtrip_and_loop_reuse(dtype, kind, n):
    source_layout, target_layout = layouts(n, kind)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=128):
            source = T.alloc_fragment((n,), dtype)
            target = T.alloc_fragment((n,), dtype)
            restored = T.alloc_fragment((n,), dtype)
            T.annotate_layout({source: source_layout, target: target_layout, restored: source_layout})
            for _ in T.serial(3):
                T.copy(A, source)
                T.evaluate(T.call_intrin("handle", "tl.tileop.layout_convert",
                                       T.region(source[0], "r", n), T.region(target[0], "w", n)))
                T.evaluate(T.call_intrin("handle", "tl.tileop.layout_convert",
                                       T.region(target[0], "r", n), T.region(restored[0], "w", n)))
                T.copy(restored, B)

    kernel = tilelang.compile(main, target="cuda", execution_backend="nvrtc")
    a = torch.arange(n, device="cuda").to(getattr(torch, dtype))
    b = torch.full_like(a, -1)
    launch = kernel.adapter._forward_from_prebuild_lib
    launch(a, b, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(a, b, stream=torch.cuda.current_stream().cuda_stream)
    b.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
