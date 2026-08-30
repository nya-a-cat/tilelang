"""Compile the fixed 1024^3 FP16 GEMM with NVIDIA cuTile Python."""

from __future__ import annotations

import argparse
from pathlib import Path

import cuda.tile as ct
import numpy as np
from cuda.tile._cext import CallingConvention
from cuda.tile._compile import compile_tile
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile.compilation import ArrayConstraint, ConstantConstraint, KernelSignature


def gemm_kernel(
    a,
    b,
    c,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
    tile_k: ct.Constant[int],
):
    block_m = ct.bid(0)
    block_n = ct.bid(1)
    num_tiles = ct.num_tiles(a, axis=1, shape=(tile_m, tile_k))
    accum = ct.full((tile_m, tile_n), 0, dtype=np.float32)
    for k_iter in range(num_tiles):
        a_tile = ct.load(a, index=(block_m, k_iter), shape=(tile_m, tile_k), latency=8)
        b_tile = ct.load(b, index=(k_iter, block_n), shape=(tile_k, tile_n), latency=8)
        accum = ct.mma(a_tile, b_tile, accum)
    ct.store(c, index=(block_m, block_n), tile=ct.astype(accum, c.dtype), latency=2)


def _array_constraint() -> ArrayConstraint:
    return ArrayConstraint(
        dtype=ct.float16,
        ndim=2,
        index_dtype=ct.int32,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(1024, 1),
        shape_constant=(1024, 1024),
        stride_divisible_by=(1024, 1),
        shape_divisible_by=(128, 128),
        base_addr_divisible_by=16,
    )


def compile_gemm(arch: str) -> bytes:
    signature = KernelSignature(
        [
            _array_constraint(),
            _array_constraint(),
            _array_constraint(),
            ConstantConstraint(128),
            ConstantConstraint(128),
            ConstantConstraint(32),
        ],
        CallingConvention.cutile_python_v2(),
        symbol="cutile_gemm_1024",
    )
    result = compile_tile(
        gemm_kernel,
        [signature],
        sm_arch=arch,
        compiler_options=CompilerOptions(opt_level=3),
        return_cubin=True,
    )
    if result.cubin is None:
        raise RuntimeError("cuTile did not return a cubin")
    return result.cubin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compile_gemm(args.arch))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
