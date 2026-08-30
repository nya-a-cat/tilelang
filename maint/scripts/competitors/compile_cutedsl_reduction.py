"""Compile a configurable TileLang reduction through the CuTeDSL backend."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import tilelang
from tilelang import env
from tilelang.cuda.target import normalize_cutedsl_target


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "maint/scripts"))
from generate_reduction_codegen import make_reduction_kernel  # noqa: E402


class _TensorArgument:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    def dim_order(self) -> tuple[int, ...]:
        return tuple(range(len(self.shape)))


def compile_reduction(
    *,
    arch: str,
    threads: int,
    width: int,
    dtype: str,
    op: str,
    batch: int,
    rows: int | None = None,
) -> Path:
    # Compiler-development probes must not reuse a kernel cached before the
    # current native lowering library was rebuilt.
    env.disable_cache()
    rows = batch if rows is None else rows
    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": arch})
    kernel = tilelang.compile(
        make_reduction_kernel(
            threads=threads,
            width=width,
            dtype=dtype,
            op=op,
            batch=batch,
            rows=rows,
        ),
        target=target,
        out_idx=[1],
        execution_backend="cutedsl",
    )
    module = kernel.adapter.pymodule
    source = _TensorArgument((rows, width))
    output = _TensorArgument((rows,))
    module._generate_cubin_if_needed(source, output)
    cubin = Path(module._cubin_path)
    if not cubin.is_file():
        raise RuntimeError("CuTeDSL backend did not emit a cubin")
    return cubin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="sm_100a")
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--op", default="sum")
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--rows", type=int, help="Output rows; defaults to batch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cubin = compile_reduction(
        arch=args.arch,
        threads=args.threads,
        width=args.width,
        dtype=args.dtype,
        op=args.op,
        batch=args.batch,
        rows=args.rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cubin, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
