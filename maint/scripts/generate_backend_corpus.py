"""Lower a representative multi-operator TileLang backend corpus to CUDA.

The workloads are intentionally fixed so generated CUDA/PTX/CUBIN/SASS can be
compared across backend revisions and exact CUDA targets without a GPU.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm

from generate_gemm_pipeline_codegen import make_gemm_kernel
from generate_im2col_codegen import make_im2col_kernel


def make_elementwise_kernel():
    m, n, block = 1024, 1024, 256

    @T.prim_func
    def elementwise(
        source: T.Tensor((m, n), T.float32),
        output: T.Tensor((m, n), T.float32),
    ):
        with T.Kernel(T.ceildiv(m * n, block), threads=block) as block_idx:
            for thread_idx in T.Parallel(block):
                index = block_idx * block + thread_idx
                if index < m * n:
                    row = index // n
                    col = index % n
                    output[row, col] = T.tanh(source[row, col]) + source[row, col] * 0.5

    return elementwise


def make_reduction_kernel():
    m, n = 1024, 1024

    @T.prim_func
    def reduction(
        source: T.Tensor((m, n), T.float32),
        output: T.Tensor((m,), T.float32),
    ):
        with T.Kernel(m, threads=128) as row:
            source_local = T.alloc_fragment((1, n), T.float32)
            reduced = T.alloc_fragment((1,), T.float32)
            T.copy(source[row, :], source_local)
            T.reduce_sum(source_local, reduced, dim=1)
            output[row] = reduced[0]

    return reduction


def make_rmsnorm_kernel():
    m, n = 1024, 1024

    @T.prim_func
    def rmsnorm(
        source: T.Tensor((m, n), T.float32),
        output: T.Tensor((m, n), T.float32),
    ):
        with T.Kernel(m, threads=128) as row:
            source_local = T.alloc_fragment((1, n), T.float32)
            square_local = T.alloc_fragment((1, n), T.float32)
            square_sum = T.alloc_fragment((1,), T.float32)
            T.copy(source[row, :], source_local)
            for i, j in T.Parallel(1, n):
                square_local[i, j] = source_local[i, j] * source_local[i, j]
            T.reduce_sum(square_local, square_sum, dim=1)
            scale = T.rsqrt(square_sum[0] / n + 1e-6)
            for i, j in T.Parallel(1, n):
                output[row + i, j] = source_local[i, j] * scale

    return rmsnorm


def make_softmax_kernel():
    m, n = 1024, 1024

    @T.prim_func
    def softmax(
        source: T.Tensor((m, n), T.float16),
        output: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(m, threads=128) as row:
            source_local = T.alloc_fragment((1, n), T.float16)
            exp_local = T.alloc_fragment((1, n), T.float32)
            max_value = T.alloc_fragment((1,), T.float16)
            exp_sum = T.alloc_fragment((1,), T.float32)
            T.copy(source[row, :], source_local)
            T.reduce_max(source_local, max_value, dim=1)
            for i, j in T.Parallel(1, n):
                exp_local[i, j] = T.exp2((source_local[i, j] - max_value[i]) * 1.44269504)
            T.reduce_sum(exp_local, exp_sum, dim=1)
            for i, j in T.Parallel(1, n):
                output[row + i, j] = exp_local[i, j] / exp_sum[i]

    return softmax


def make_transpose_kernel():
    m, n, block = 1024, 1024, 32

    @T.prim_func
    def transpose(
        source: T.Tensor((m, n), T.float32),
        output: T.Tensor((n, m), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, block), T.ceildiv(m, block), threads=128) as (bx, by):
            tile = T.alloc_shared((block, block), T.float32)
            T.copy(source[by * block, bx * block], tile)
            for i, j in T.Parallel(block, block):
                output[bx * block + j, by * block + i] = tile[i, j]

    return transpose


def make_attention_kernel(num_stages: int = 2):
    batch, heads, seq, dim = 1, 1, 256, 64
    block_m = block_n = 64
    scale = (1.0 / dim) ** 0.5 * 1.44269504

    @T.prim_func
    def attention(
        query: T.Tensor((batch, heads, seq, dim), T.float16),
        key: T.Tensor((batch, heads, seq, dim), T.float16),
        value: T.Tensor((batch, heads, seq, dim), T.float16),
        output: T.Tensor((batch, heads, seq, dim), T.float16),
    ):
        with T.Kernel(T.ceildiv(seq, block_m), heads, batch, threads=128) as (bx, head, batch_idx):
            query_shared = T.alloc_shared((block_m, dim), T.float16)
            key_shared = T.alloc_shared((block_n, dim), T.float16)
            value_shared = T.alloc_shared((block_n, dim), T.float16)
            output_shared = T.alloc_shared((block_m, dim), T.float16)
            score = T.alloc_fragment((block_m, block_n), T.float32)
            score_cast = T.alloc_fragment((block_m, block_n), T.float16)
            output_accum = T.alloc_fragment((block_m, dim), T.float32)
            score_max = T.alloc_fragment((block_m,), T.float32)
            score_max_prev = T.alloc_fragment((block_m,), T.float32)
            score_scale = T.alloc_fragment((block_m,), T.float32)
            score_sum = T.alloc_fragment((block_m,), T.float32)
            normalizer = T.alloc_fragment((block_m,), T.float32)

            T.copy(query[batch_idx, head, bx * block_m : (bx + 1) * block_m, :], query_shared)
            T.clear(output_accum)
            T.clear(normalizer)
            T.fill(score_max, -T.infinity(T.float32))

            for key_block in T.Pipelined(T.ceildiv(seq, block_n), num_stages=num_stages):
                T.copy(key[batch_idx, head, key_block * block_n : (key_block + 1) * block_n, :], key_shared)
                T.clear(score)
                T.gemm(query_shared, key_shared, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(score_max, score_max_prev)
                T.fill(score_max, -T.infinity(T.float32))
                T.reduce_max(score, score_max, dim=1, clear=False)
                for i in T.Parallel(block_m):
                    score_max[i] = T.max(score_max[i], score_max_prev[i])
                    score_scale[i] = T.exp2(score_max_prev[i] * scale - score_max[i] * scale)
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.exp2(score[i, j] * scale - score_max[i] * scale)
                T.reduce_sum(score, score_sum, dim=1)
                for i in T.Parallel(block_m):
                    normalizer[i] = normalizer[i] * score_scale[i] + score_sum[i]
                T.copy(score, score_cast)
                for i, j in T.Parallel(block_m, dim):
                    output_accum[i, j] *= score_scale[i]
                T.copy(value[batch_idx, head, key_block * block_n : (key_block + 1) * block_n, :], value_shared)
                T.gemm(score_cast, value_shared, output_accum, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_m, dim):
                output_accum[i, j] /= normalizer[i]
            T.copy(output_accum, output_shared)
            T.copy(output_shared, output[batch_idx, head, bx * block_m : (bx + 1) * block_m, :])

    return attention


WORKLOADS: dict[str, Callable[[], object]] = {
    "elementwise": make_elementwise_kernel,
    "reduction": make_reduction_kernel,
    "rmsnorm": make_rmsnorm_kernel,
    "softmax": make_softmax_kernel,
    "transpose": make_transpose_kernel,
    "gemm": lambda: make_gemm_kernel(num_stages=2),
    "im2col": make_im2col_kernel,
    "attention": make_attention_kernel,
}


def lower_source(operation: str, arch: str, num_stages: int | None = None) -> str:
    target = {"kind": "cuda", "arch": arch}
    if operation == "gemm" and num_stages is not None:
        workload = make_gemm_kernel(num_stages)
    elif operation == "attention" and num_stages is not None:
        workload = make_attention_kernel(num_stages)
    else:
        workload = WORKLOADS[operation]()
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(
            workload,
            target=target,
            enable_device_compile=False,
        )
    if artifact.kernel_source is None:
        raise RuntimeError("TileLang lowering did not return CUDA source")
    return artifact.kernel_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="Exact target token, for example sm_100a")
    parser.add_argument("--operation", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--num-stages", type=int, choices=range(1, 5))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(lower_source(args.operation, args.arch, args.num_stages))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
