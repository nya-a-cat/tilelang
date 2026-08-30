"""Compile the fixed 1024^3 FP16 GEMM with TileLang's CuTeDSL backend."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import tilelang
from tilelang.cuda.target import normalize_cutedsl_target


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "maint/scripts"))
from generate_gemm_pipeline_codegen import make_gemm_kernel  # noqa: E402


class _TensorArgument:
    shape = (1024, 1024)

    @staticmethod
    def dim_order() -> tuple[int, int]:
        return (0, 1)


def compile_gemm(arch: str) -> Path:
    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": arch})
    kernel = tilelang.compile(
        make_gemm_kernel(num_stages=2),
        target=target,
        out_idx=[2],
        execution_backend="cutedsl",
    )
    module = kernel.adapter.pymodule
    argument = _TensorArgument()
    # The backend intentionally supports fake tensor descriptors so the DSL
    # compiler can emit a CUBIN on build machines without a CUDA device.
    module._generate_cubin_if_needed(argument, argument, argument)
    cubin = Path(module._cubin_path)
    if not cubin.is_file():
        raise RuntimeError("CuTeDSL backend did not emit a cubin")
    return cubin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="sm_100a")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cubin = compile_gemm(args.arch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cubin, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
