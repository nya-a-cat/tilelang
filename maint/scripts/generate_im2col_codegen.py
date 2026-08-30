"""Lower a fixed TileLang Im2Col convolution for paired codegen experiments."""

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


def make_im2col_kernel():
    """Return the fixed workload used by the backend old/new comparison."""
    n, channels, height, width, filters, kernel = 1, 32, 8, 8, 32, 3
    stride, dilation, padding = 1, 1, 1
    block_m, block_n, block_k = 16, 32, 32
    output_h = (height + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    output_w = (width + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    @T.prim_func
    def conv(
        data: T.Tensor((n, height, width, channels), T.float16),
        weight: T.Tensor((kernel, kernel, channels, filters), T.float16),
        out: T.Tensor((n, output_h, output_w, filters), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(filters, block_n),
            T.ceildiv(n * output_h * output_w, block_m),
            threads=128,
        ) as (bx, by):
            data_shared = T.alloc_shared((block_m, block_k), T.float16)
            weight_shared = T.alloc_shared((block_k, block_n), T.float16)
            out_local = T.alloc_fragment((block_m, block_n), T.float32)
            out_shared = T.alloc_shared((block_m, block_n), T.float16)

            weight_flat = T.Tensor((kernel * kernel * channels, filters), T.float16, weight.data)
            out_flat = T.Tensor((n * output_h * output_w, filters), T.float16, out.data)

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(kernel * kernel * channels, block_k), num_stages=3):
                T.im2col(data, data_shared, by, k_iter, kernel, stride, dilation, padding)
                T.copy(weight_flat[k_iter * block_k, bx * block_n], weight_shared)
                T.gemm(data_shared, weight_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_m, bx * block_n])

    return conv


def lower_source(arch: str) -> str:
    target = {"kind": "cuda", "arch": arch}
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(make_im2col_kernel(), target=target, enable_device_compile=False)
    if artifact.kernel_source is None:
        raise RuntimeError("TileLang lowering did not return CUDA source")
    return artifact.kernel_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="Exact target token, for example sm_100a")
    parser.add_argument("--output", required=True, type=Path, help="Generated CUDA source path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(lower_source(args.arch))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
