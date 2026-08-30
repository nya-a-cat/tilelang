"""Lower a configurable TileLang block reduction to CUDA source.

This source-only generator is intended for paired AllReduce CUDA/PTX/SASS
experiments on machines without a GPU.  Use ``analyze_cuda_codegen.py`` to
compile and disassemble its output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


DTYPES = {
    "float16": T.float16,
    "bfloat16": T.bfloat16,
    "float32": T.float32,
    "int32": T.int32,
}

REDUCERS = {
    "absmax": T.reduce_absmax,
    "abssum": T.reduce_abssum,
    "bitand": T.reduce_bitand,
    "bitor": T.reduce_bitor,
    "bitxor": T.reduce_bitxor,
    "sum": T.reduce_sum,
    "max": T.reduce_max,
    "min": T.reduce_min,
}


def make_reduction_kernel(
    *,
    threads: int,
    width: int,
    dtype: str,
    op: str,
    batch: int,
    rows: int | None = None,
):
    if threads <= 0 or threads > 1024 or threads & (threads - 1):
        raise ValueError("threads must be a power of two in [1, 1024]")
    if width <= 0 or width % threads:
        raise ValueError("width must be positive and divisible by threads")
    if batch <= 0:
        raise ValueError("batch must be positive")
    if rows is None:
        rows = batch
    if rows <= 0 or rows % batch:
        raise ValueError("rows must be positive and divisible by batch")

    tl_dtype = DTYPES[dtype]
    reduce_fn = REDUCERS[op]

    @T.prim_func
    def reduction(
        source: T.Tensor((rows, width), tl_dtype),
        output: T.Tensor((rows,), tl_dtype),
    ):
        with T.Kernel(1, threads=threads):
            source_local = T.alloc_fragment((rows, width), tl_dtype)
            reduced = T.alloc_fragment((rows,), tl_dtype)
            T.copy(source, source_local)
            reduce_fn(source_local, reduced, dim=1, batch=batch)
            T.copy(reduced, output)

    return reduction


def lower_source(
    *,
    arch: str,
    threads: int,
    width: int,
    dtype: str,
    op: str,
    batch: int,
    rows: int | None = None,
    force_baseline: bool = False,
) -> str:
    target = {"kind": "cuda", "arch": arch}
    workload = make_reduction_kernel(
        threads=threads,
        width=width,
        dtype=dtype,
        op=op,
        batch=batch,
        rows=rows,
    )
    pass_config = {"tl.reducer_force_baseline": True} if force_baseline else {}
    with tvm.transform.PassContext(config=pass_config), tvm.target.Target(target):
        artifact = tilelang.lower(workload, target=target, enable_device_compile=False)
    if artifact.kernel_source is None:
        raise RuntimeError("TileLang lowering did not return CUDA source")
    return artifact.kernel_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="Exact target token, for example sm_100a")
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--op", choices=sorted(REDUCERS), default="sum")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rows", type=int, help="Output rows; defaults to batch")
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="Force the legacy reducer plan; the workload shape still determines participation",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        lower_source(
            arch=args.arch,
            threads=args.threads,
            width=args.width,
            dtype=args.dtype,
            op=args.op,
            batch=args.batch,
            rows=args.rows,
            force_baseline=args.force_baseline,
        )
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
