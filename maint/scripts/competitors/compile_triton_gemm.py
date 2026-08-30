"""Compile the fixed 1024^3 FP16 GEMM with the current Triton package."""

from __future__ import annotations

import argparse
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    program = tl.program_id(axis=0)
    grid_n = tl.cdiv(n, block_n)
    block_row = program // grid_n
    block_col = program % grid_n
    row_offsets = block_row * block_m + tl.arange(0, block_m)
    col_offsets = block_col * block_n + tl.arange(0, block_n)
    k_offsets = tl.arange(0, block_k)
    a_offsets = row_offsets[:, None] * k + k_offsets[None, :]
    b_offsets = k_offsets[:, None] * n + col_offsets[None, :]
    accum = tl.zeros((block_m, block_n), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k, block_k)):
        a = tl.load(a_ptr + a_offsets)
        b = tl.load(b_ptr + b_offsets)
        accum = tl.dot(a, b, accum)
        a_offsets += block_k
        b_offsets += block_k * n
    output_offsets = row_offsets[:, None] * n + col_offsets[None, :]
    tl.store(c_ptr + output_offsets, accum.to(tl.float16))


def compile_gemm(compute_capability: int, num_stages: int) -> tuple[bytes, str]:
    source = ASTSource(
        fn=gemm_kernel,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16"},
        constexprs={
            "m": 1024,
            "n": 1024,
            "k": 1024,
            "block_m": 128,
            "block_n": 128,
            "block_k": 32,
        },
    )
    result = triton.compile(
        source,
        target=GPUTarget("cuda", compute_capability, 32),
        options={"num_warps": 4, "num_stages": num_stages},
    )
    return result.asm["cubin"], result.asm["ptx"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-capability", type=int, required=True)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cubin, ptx = compile_gemm(args.compute_capability, args.num_stages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(cubin)
    args.output.with_suffix(".ptx").write_text(ptx)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
