"""Lower a fixed TileLang GEMM at a selected software-pipeline depth."""

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


def make_gemm_kernel(num_stages: int):
    m = n = k = 1024
    block_m = block_n = 128
    block_k = 32

    @T.prim_func
    def gemm(
        a: T.Tensor((m, k), T.float16),
        b: T.Tensor((n, k), T.float16),
        c: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (bx, by):
            a_shared = T.alloc_shared((block_m, block_k), T.float16)
            b_shared = T.alloc_shared((block_n, block_k), T.float16)
            c_local = T.alloc_fragment((block_m, block_n), T.float32)
            c_shared = T.alloc_shared((block_m, block_n), T.float16)

            T.clear(c_local)
            for k_iter in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                T.copy(a[by * block_m, k_iter * block_k], a_shared)
                T.copy(b[bx * block_n, k_iter * block_k], b_shared)
                T.gemm(a_shared, b_shared, c_local, transpose_B=True)

            T.copy(c_local, c_shared)
            T.copy(c_shared, c[by * block_m, bx * block_n])

    return gemm


def lower_source(arch: str, num_stages: int) -> str:
    target = {"kind": "cuda", "arch": arch}
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(
            make_gemm_kernel(num_stages),
            target=target,
            enable_device_compile=False,
        )
    if artifact.kernel_source is None:
        raise RuntimeError("TileLang lowering did not return CUDA source")
    return artifact.kernel_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="Exact target token, for example sm_89")
    parser.add_argument("--num-stages", required=True, type=int, choices=range(1, 5))
    parser.add_argument("--output", required=True, type=Path, help="Generated CUDA source path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(lower_source(args.arch, args.num_stages))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
