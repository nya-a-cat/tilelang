"""Compile frozen real-workload vectorization variants without a GPU.

This diagnostic elaborates frozen configurations from real TileLang examples:
DeepSeek mHC prefill, W4A8 dequantization GEMM, FP16 GEMM, split-K GEMM,
RMSNorm, LayerNorm, FlashAttention, convolution, elementwise addition, and a
guarded varlen-attention backward staging reproduction.
Each byte-identical PrimFunc is lowered with the requested transformation modes.
Compilation-only candidates reuse the exact planner CUDA source bytes before
being compiled with TileLang's CUDA callback and disassembled for explicit
NVIDIA architectures.  The diagnostic never initializes or executes a GPU.
"""

from __future__ import annotations

import concurrent.futures
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Any
import zipfile

import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.contrib.cuda_resource_info import (
    pop_auto_launch_bounds_selection,
    pop_recorded,
    reset_recorder,
)
from tilelang.cuda.backend import tilelang_callback_cuda_compile


REPOSITORY = "nya-a-cat/tilelang"
CONFIG_KEY = "tl.vectorize_local_parallel"
SCALAR_EXP2_CONFIG_KEY = "tl.vectorize_local_parallel_scalar_exp2"
DISABLE_PREDICATED_ASYNC_COPY_CONFIG_KEY = "tl.disable_predicated_async_copy"
INPLACE_CONFIG_KEY = "tl.storage_rewrite_detect_inplace"
REGISTER_USAGE_CONFIG_KEY = "tl.ptxas_register_usage_level"
AUTO_LAUNCH_BOUNDS_CONFIG_KEY = "tl.enable_auto_launch_bounds"
DEVICE_COMPILE_FLAGS_CONFIG_KEY = "tl.device_compile_flags"
DISABLE_REINTERPRET_CONFIG_KEY = "tl.disable_reinterpret_vectorization"
DISABLE_INT4X2_UNPACK_CONFIG_KEY = "tl.disable_int4x2_unpack_peephole"
VARLEN_BATCH = T.symbolic("batch")
VARLEN_TOTAL_Q = T.symbolic("total_q")
VARLEN_TOTAL_K = T.symbolic("total_k")
MODES = tuple(
    mode.strip()
    for mode in os.environ.get(
        "TILELANG_REAL_VECTOR_MODES",
        "planner,legacy",
    ).split(",")
    if mode.strip()
)
WORKLOAD_NAMES = tuple(
    name.strip()
    for name in os.environ.get("TILELANG_REAL_VECTOR_WORKLOADS", "").split(",")
    if name.strip()
)
BASE_MODES = {
    "planner",
    "legacy",
    "planner_inplace",
    "planner_reinterpret_legacy",
    "planner_int4x2_unpack_legacy",
    "planner_auto_lb",
    "planner_nvcc_extra_vectorization",
    "planner_sync_predicated_copy",
    "planner_scalar_exp2_vectorized",
}
REGISTER_USAGE_MODE_RE = re.compile(r"planner_ru(?P<level>\d+)$")
LAUNCH_BOUNDS_MODE_RE = re.compile(r"planner_lb(?P<blocks>\d+)$")
DEFAULT_ARCHES = "sm_75,sm_80,sm_90a,sm_100a,sm_120a"
RESULT_PATH = Path(
    os.environ.get(
        "TILELANG_REAL_VECTOR_RESULT",
        "tilelang-real-vectorization-workloads.json",
    )
)
REPORT_PATH = Path(
    os.environ.get(
        "TILELANG_REAL_VECTOR_REPORT",
        "tilelang-real-vectorization-workloads.md",
    )
)
RAW_ARCHIVE_PATH = Path(
    os.environ.get(
        "TILELANG_REAL_VECTOR_RAW_ARCHIVE",
        "tilelang-real-vectorization-workloads-raw.zip",
    )
)
RAW_DIR = Path(
    os.environ.get(
        "TILELANG_REAL_VECTOR_RAW_DIR",
        "tilelang-real-vectorization-workloads-raw",
    )
)
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")
ARCHES = tuple(
    arch.strip()
    for arch in os.environ.get("TILELANG_REAL_VECTOR_ARCHES", DEFAULT_ARCHES).split(",")
    if arch.strip()
)
MAX_WORKERS = int(os.environ.get("TILELANG_REAL_VECTOR_COMPILE_WORKERS", "4"))
SASS_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)",
    re.MULTILINE,
)
PACKED_TYPE_RE = re.compile(
    r"\b(?:float|double|int|uint|short|ushort|char|uchar|half|bfloat)[234]\b"
)

if not ARCHES or len(set(ARCHES)) != len(ARCHES):
    raise ValueError("TILELANG_REAL_VECTOR_ARCHES must contain distinct architectures")
if len(set(WORKLOAD_NAMES)) != len(WORKLOAD_NAMES):
    raise ValueError("TILELANG_REAL_VECTOR_WORKLOADS must contain distinct workload names")
invalid_modes = [
    mode
    for mode in MODES
    if mode not in BASE_MODES
    and REGISTER_USAGE_MODE_RE.fullmatch(mode) is None
    and LAUNCH_BOUNDS_MODE_RE.fullmatch(mode) is None
]
invalid_register_levels = [
    mode
    for mode in MODES
    if (match := REGISTER_USAGE_MODE_RE.fullmatch(mode)) is not None
    and not 0 <= int(match.group("level")) <= 10
]
invalid_launch_bounds = [
    mode
    for mode in MODES
    if (match := LAUNCH_BOUNDS_MODE_RE.fullmatch(mode)) is not None
    and not 2 <= int(match.group("blocks")) <= 8
]
if (
    not MODES
    or len(set(MODES)) != len(MODES)
    or not {"planner", "legacy"} <= set(MODES)
    or invalid_modes
    or invalid_register_levels
    or invalid_launch_bounds
):
    raise ValueError(
        "TILELANG_REAL_VECTOR_MODES must contain planner and legacy; optional modes are "
        "planner_inplace, planner_reinterpret_legacy, planner_int4x2_unpack_legacy, "
        "planner_auto_lb, planner_scalar_exp2_vectorized, planner_ru0 through "
        "planner_ru10, planner_nvcc_extra_vectorization, "
        "planner_sync_predicated_copy, and planner_lb2 through planner_lb8"
    )
if MAX_WORKERS < 1:
    raise ValueError("TILELANG_REAL_VECTOR_COMPILE_WORKERS must be positive")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_workloads(repo_root: Path) -> list[dict[str, Any]]:
    mhc_path = repo_root / "examples/deepseek_mhc/example_mhc_pre.py"
    w4a8_path = repo_root / "examples/dequantize_gemm/example_dequant_gemm_w4a8.py"
    gemm_path = repo_root / "examples/gemm/example_gemm.py"
    rms_norm_path = repo_root / "examples/norm/rms_norm.py"
    layer_norm_path = repo_root / "examples/norm/layernorm.py"
    flash_attention_path = (
        repo_root / "examples/flash_attention/example_mha_fwd_bshd.py"
    )
    convolution_path = repo_root / "examples/convolution/example_convolution.py"
    elementwise_path = repo_root / "examples/elementwise/example_elementwise_add.py"
    splitk_path = repo_root / "examples/gemm_splitk/example_tilelang_gemm_splitk.py"
    for path in (
        mhc_path,
        w4a8_path,
        gemm_path,
        rms_norm_path,
        layer_norm_path,
        flash_attention_path,
        convolution_path,
        elementwise_path,
        splitk_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"frozen example entrypoint is missing: {path}")

    mhc = load_module("tilelang_trace_example_mhc_pre", mhc_path)
    w4a8 = load_module("tilelang_trace_example_dequant_gemm_w4a8", w4a8_path)
    gemm = load_module("tilelang_trace_example_gemm", gemm_path)
    rms_norm = load_module("tilelang_trace_example_rms_norm", rms_norm_path)
    layer_norm = load_module("tilelang_trace_example_layer_norm", layer_norm_path)
    flash_attention = load_module(
        "tilelang_trace_example_mha_fwd_bshd", flash_attention_path
    )
    convolution = load_module(
        "tilelang_trace_example_convolution", convolution_path
    )
    elementwise = load_module(
        "tilelang_trace_example_elementwise", elementwise_path
    )
    splitk = load_module("tilelang_trace_example_splitk", splitk_path)

    tokens = 2048
    hidden_size = 4096
    hc_mult = 4
    n_splits = 1
    hc_mult3 = hc_mult * (2 + hc_mult)
    mhc_jit = mhc.mhc_pre_big_fuse_tilelang
    mhc_prim = mhc_jit.get_tir(
        T.Tensor((n_splits, tokens, hc_mult3), T.float32),
        T.Tensor((n_splits, tokens), T.float32),
        T.Tensor((3,), T.float32),
        T.Tensor((hc_mult3,), T.float32),
        T.Tensor((tokens, hc_mult, hidden_size), T.bfloat16),
        T.Tensor((tokens, hc_mult), T.float32),
        T.Tensor((tokens, hc_mult * hc_mult), T.float32),
        T.Tensor((tokens, hidden_size), T.bfloat16),
        hidden_size=hidden_size,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=10,
        n_splits=n_splits,
        hc_mult=hc_mult,
    )

    w4a8_jit = w4a8.matmul_int8xint4.jit_impl
    w4a8_prim = w4a8_jit.get_tir(
        4096,
        4096,
        4096,
        "int8",
        "int32",
        "int32",
        num_bits=4,
        block_M=32,
        block_N=32,
        block_K=128,
        num_stages=1,
        threads=128,
    )

    gemm_jit = gemm.matmul
    gemm_prim = gemm_jit.get_tir(
        M=1024,
        N=1024,
        K=1024,
        block_M=128,
        block_N=128,
        block_K=32,
    )

    rms_norm_jit = rms_norm.rms_norm
    rms_norm_prim = rms_norm_jit.get_tir(M=8192, N=8192, blk_m=1)

    layer_norm_jit = layer_norm._layernorm_fwd
    layer_norm_prim = layer_norm_jit.get_tir(
        N=4096,
        D=8192,
        eps=1e-5,
        blk_m=1,
        threads=256,
        in_dtype="bfloat16",
        out_dtype="bfloat16",
    )

    flash_attention_jit = flash_attention.flashattn.jit_impl
    flash_attention_prim = flash_attention_jit.get_tir(
        8,
        32,
        4096,
        128,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )

    convolution_jit = convolution.convolution
    convolution_prim = convolution_jit.get_tir(
        N=128,
        C=128,
        H=64,
        W=64,
        F=128,
        K=3,
        S=1,
        D=1,
        P=1,
        block_M=64,
        block_N=128,
        block_K=32,
        num_stages=3,
        threads=256,
    )

    elementwise_jit = elementwise.elementwise_add
    elementwise_prim = elementwise_jit.get_tir(
        M=4096,
        N=4096,
        block_M=32,
        block_N=32,
        in_dtype="float32",
        out_dtype="float32",
        threads=128,
    )

    splitk_jit = splitk.matmul
    splitk_prim = splitk_jit.get_tir(
        M=4096,
        N=4096,
        K=4096,
        block_M=128,
        block_N=128,
        block_K=32,
        split_k=4,
    )

    heads = 64
    head_dimension = 512
    block_m = 64
    block_n = 64

    @T.prim_func
    def varlen_attention_guarded_stage(
        Q: T.Tensor((VARLEN_TOTAL_Q, heads, head_dimension), T.bfloat16),
        K: T.Tensor((VARLEN_TOTAL_K, 1, head_dimension), T.bfloat16),
        cu_q: T.Tensor((VARLEN_BATCH + 1,), T.int32),
        cu_k: T.Tensor((VARLEN_BATCH + 1,), T.int32),
        Out: T.Tensor((VARLEN_TOTAL_K, 1, head_dimension), T.float32),
        max_seqlen_k: T.int32,
    ):
        with T.Kernel(
            heads,
            T.ceildiv(max_seqlen_k, block_m),
            VARLEN_BATCH,
            threads=256,
        ) as (bx, by, bz):
            K_shared = T.alloc_shared(
                (block_m, head_dimension), T.bfloat16
            )
            Q_shared = T.alloc_shared(
                (block_n, head_dimension), T.bfloat16
            )
            scores_shared = T.alloc_shared((block_m, block_n), T.bfloat16)
            scores_local = T.alloc_fragment((block_m, block_n), T.float32)
            accum = T.alloc_fragment((block_m, head_dimension), T.float32)

            q_start = cu_q[bz]
            q_end = cu_q[bz + 1]
            k_start = cu_k[bz]
            k_end = cu_k[bz + 1]
            q_length = q_end - q_start
            k_length = k_end - k_start

            for i, d in T.Parallel(block_m, head_dimension):
                if by * block_m + i < k_length:
                    K_shared[i, d] = K[k_start + by * block_m + i, 0, d]
                else:
                    K_shared[i, d] = T.bfloat16(0)

            T.clear(accum)
            for kb in T.Pipelined(
                T.ceildiv(q_length, block_n), num_stages=1
            ):
                for i, d in T.Parallel(block_n, head_dimension):
                    if kb * block_n + i < q_length:
                        Q_shared[i, d] = Q[q_start + kb * block_n + i, bx, d]
                    else:
                        Q_shared[i, d] = T.bfloat16(0)
                T.clear(scores_local)
                T.gemm(
                    K_shared,
                    Q_shared,
                    scores_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(scores_local, scores_shared)
                T.gemm(
                    scores_shared,
                    Q_shared,
                    accum,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i, d in T.Parallel(block_m, head_dimension):
                if by * block_m + i < k_length:
                    Out[k_start + by * block_m + i, 0, d] = accum[i, d]

    return [
        {
            "name": "deepseek_mhc_pre_big_fuse",
            "example_path": mhc_path,
            "prim_func": mhc_prim,
            "pass_configs": dict(mhc_jit.pass_configs or {}),
            "configuration": {
                "tokens": tokens,
                "hidden_size": hidden_size,
                "hc_mult": hc_mult,
                "n_splits": n_splits,
                "sinkhorn_repeat": 10,
                "rms_eps": 1e-6,
                "hc_pre_eps": 1e-6,
                "hc_sinkhorn_eps": 1e-6,
                "hc_post_mult_value": 1.0,
            },
            "scope_note": "large fused kernel from the sequential mHC pre regression",
        },
        {
            "name": "dequant_gemm_w4a8_4096",
            "example_path": w4a8_path,
            "prim_func": w4a8_prim,
            "pass_configs": dict(w4a8_jit.pass_configs or {}),
            "configuration": {
                "M": 4096,
                "N": 4096,
                "K": 4096,
                "in_dtype": "int8",
                "out_dtype": "int32",
                "accum_dtype": "int32",
                "num_bits": 4,
                "block_M": 32,
                "block_N": 32,
                "block_K": 128,
                "num_stages": 1,
                "threads": 128,
            },
            "scope_note": "exact fixed configuration used by run_regression_perf",
        },
        {
            "name": "gemm_fp16_1024",
            "example_path": gemm_path,
            "prim_func": gemm_prim,
            "pass_configs": dict(gemm_jit.pass_configs or {}),
            "configuration": {
                "M": 1024,
                "N": 1024,
                "K": 1024,
                "block_M": 128,
                "block_N": 128,
                "block_K": 32,
                "dtype": "float16",
                "accum_dtype": "float32",
                "num_stages": 3,
                "threads": 128,
            },
            "scope_note": "exact fixed configuration used by run_regression_perf",
        },
        {
            "name": "rms_norm_fp32_8192",
            "example_path": rms_norm_path,
            "prim_func": rms_norm_prim,
            "pass_configs": dict(rms_norm_jit.pass_configs or {}),
            "configuration": {
                "M": 8192,
                "N": 8192,
                "block_M": 1,
                "dtype": "float32",
                "threads": 128,
            },
            "scope_note": "exact fixed configuration used by the example main program",
        },
        {
            "name": "layer_norm_bf16_4096x8192",
            "example_path": layer_norm_path,
            "prim_func": layer_norm_prim,
            "pass_configs": dict(layer_norm_jit.pass_configs or {}),
            "configuration": {
                "N": 4096,
                "D": 8192,
                "eps": 1e-5,
                "block_M": 1,
                "threads": 256,
                "in_dtype": "bfloat16",
                "out_dtype": "bfloat16",
            },
            "scope_note": "exact forward configuration used by the example main program",
        },
        {
            "name": "flash_attention_fp16_b8_h32_s4096_d128",
            "example_path": flash_attention_path,
            "prim_func": flash_attention_prim,
            "pass_configs": dict(flash_attention_jit.pass_configs or {}),
            "configuration": {
                "batch": 8,
                "heads": 32,
                "sequence_length": 4096,
                "head_dimension": 128,
                "is_causal": False,
                "block_M": 128,
                "block_N": 128,
                "num_stages": 1,
                "threads": 128,
            },
            "scope_note": "exact fixed configuration used by run_regression_perf",
        },
        {
            "name": "convolution_fp16_n128_c128_h64_w64_f128_k3",
            "example_path": convolution_path,
            "prim_func": convolution_prim,
            "pass_configs": dict(convolution_jit.pass_configs or {}),
            "configuration": {
                "N": 128,
                "C": 128,
                "H": 64,
                "W": 64,
                "F": 128,
                "K": 3,
                "stride": 1,
                "dilation": 1,
                "padding": 1,
                "block_M": 64,
                "block_N": 128,
                "block_K": 32,
                "num_stages": 3,
                "threads": 256,
            },
            "scope_note": "exact fixed configuration used by run_regression_perf",
        },
        {
            "name": "elementwise_add_fp32_4096x4096",
            "example_path": elementwise_path,
            "prim_func": elementwise_prim,
            "pass_configs": dict(elementwise_jit.pass_configs or {}),
            "configuration": {
                "M": 4096,
                "N": 4096,
                "block_M": 32,
                "block_N": 32,
                "threads": 128,
                "in_dtype": "float32",
                "out_dtype": "float32",
            },
            "scope_note": "exact fixed configuration used by run_regression_perf",
        },
        {
            "name": "splitk_gemm_fp16_4096_split4",
            "example_path": splitk_path,
            "prim_func": splitk_prim,
            "pass_configs": dict(splitk_jit.pass_configs or {}),
            "configuration": {
                "M": 4096,
                "N": 4096,
                "K": 4096,
                "block_M": 128,
                "block_N": 128,
                "block_K": 32,
                "split_k": 4,
                "threads": 128,
                "in_dtype": "float16",
                "accum_dtype": "float32",
            },
            "scope_note": "exact fixed configuration used by run_regression_perf",
        },
        {
            "name": "varlen_attention_guarded_stage_bwd_repro",
            "example_path": Path(__file__).resolve(),
            "prim_func": varlen_attention_guarded_stage,
            "pass_configs": {},
            "configuration": {
                "heads": heads,
                "head_dimension": head_dimension,
                "block_M": block_m,
                "block_N": block_n,
                "num_stages": 1,
                "threads": 256,
            },
            "scope_note": "public varlen-attention backward guarded-staging reproduction",
        },
    ]


def packed_types(source: str) -> dict[str, int]:
    return dict(sorted(Counter(PACKED_TYPE_RE.findall(source)).items()))


def validate_predicated_async_negative_controls() -> list[dict[str, Any]]:
    @T.prim_func
    def nonzero_fallback(
        A: T.Tensor((256,), T.float16),
        B: T.Tensor((2,), T.float16),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128,), T.float16)
            for tile in T.Pipelined(2, num_stages=1):
                for i in T.Parallel(128):
                    if tile * 128 + i < 130:
                        S[i] = A[tile * 128 + i]
                    else:
                        S[i] = T.float16(1)
                B[tile] = S[0]

    @T.prim_func
    def partial_copy(
        A: T.Tensor((256,), T.float16),
        B: T.Tensor((2,), T.float16),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128,), T.float16)
            for tile in T.Pipelined(2, num_stages=1):
                for i in T.Parallel(128):
                    if tile * 128 + i < 130:
                        S[i] = A[tile * 128 + i]
                B[tile] = S[0]

    @T.prim_func
    def state_dependent_guard(
        A: T.Tensor((256,), T.float16),
        Mask: T.Tensor((256,), T.int32),
        B: T.Tensor((2,), T.float16),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128,), T.float16)
            for tile in T.Pipelined(2, num_stages=1):
                for i in T.Parallel(128):
                    if Mask[tile * 128 + i] != 0:
                        S[i] = A[tile * 128 + i]
                    else:
                        S[i] = T.float16(0)
                B[tile] = S[0]

    @T.prim_func
    def state_dependent_index(
        A: T.Tensor((256,), T.float16),
        Indices: T.Tensor((256,), T.int32),
        B: T.Tensor((2,), T.float16),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128,), T.float16)
            for tile in T.Pipelined(2, num_stages=1):
                for i in T.Parallel(128):
                    if tile * 128 + i < 256:
                        S[i] = A[Indices[tile * 128 + i]]
                    else:
                        S[i] = T.float16(0)
                B[tile] = S[0]

    controls = {
        "nonzero_fallback": nonzero_fallback,
        "partial_copy": partial_copy,
        "state_dependent_guard": state_dependent_guard,
        "state_dependent_index": state_dependent_index,
    }
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    output_dir = RAW_DIR / "predicated_async_negative_controls"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, func in controls.items():
        with target:
            source = tl.lower(
                func, target=target, enable_device_compile=False
            ).kernel_source
        occurrences = len(
            re.findall(r"\bcp_async_gs(?:_conditional)?<", source)
        )
        if occurrences != 0:
            raise RuntimeError(
                f"predicated async negative control {name} emitted "
                f"{occurrences} cp.async call sites"
            )
        (output_dir / f"{name}.cu").write_text(source, encoding="utf-8")
        records.append(
            {
                "name": name,
                "arch": "sm_80",
                "cp_async_source_occurrences": occurrences,
                "source_sha256": sha256_text(source),
            }
        )
    return records


def register_usage_level(mode: str) -> int | None:
    match = REGISTER_USAGE_MODE_RE.fullmatch(mode)
    return int(match.group("level")) if match is not None else None


def launch_bounds_blocks(mode: str) -> int | None:
    match = LAUNCH_BOUNDS_MODE_RE.fullmatch(mode)
    return int(match.group("blocks")) if match is not None else None


def lowering_source_mode(mode: str) -> str:
    """Map compile-only experiments to their shared lowering source."""

    if (
        mode == "planner_auto_lb"
        or mode == "planner_nvcc_extra_vectorization"
        or register_usage_level(mode) is not None
        or launch_bounds_blocks(mode) is not None
    ):
        return "planner"
    return mode


def pass_configs_for_mode(base_configs: dict[str, Any], mode: str) -> dict[str, Any]:
    pass_configs = dict(base_configs)
    pass_configs[CONFIG_KEY] = mode != "legacy"
    pass_configs[SCALAR_EXP2_CONFIG_KEY] = mode == "planner_scalar_exp2_vectorized"
    pass_configs[INPLACE_CONFIG_KEY] = mode == "planner_inplace"
    pass_configs[DISABLE_REINTERPRET_CONFIG_KEY] = mode == "planner_reinterpret_legacy"
    pass_configs[DISABLE_INT4X2_UNPACK_CONFIG_KEY] = (
        mode == "planner_int4x2_unpack_legacy"
    )
    pass_configs[AUTO_LAUNCH_BOUNDS_CONFIG_KEY] = mode == "planner_auto_lb"
    if mode == "planner_sync_predicated_copy":
        pass_configs[DISABLE_PREDICATED_ASYNC_COPY_CONFIG_KEY] = True
    if mode == "planner_nvcc_extra_vectorization":
        compile_flags = pass_configs.get(DEVICE_COMPILE_FLAGS_CONFIG_KEY, [])
        if isinstance(compile_flags, str):
            compile_flags = [compile_flags]
        else:
            compile_flags = list(compile_flags)
        compile_flags.append("--extra-device-vectorization")
        pass_configs[DEVICE_COMPILE_FLAGS_CONFIG_KEY] = compile_flags
    usage_level = register_usage_level(mode)
    if usage_level is not None:
        pass_configs[REGISTER_USAGE_CONFIG_KEY] = usage_level
    return pass_configs


def rewrite_launch_bounds(source: str, blocks: int) -> tuple[str, int]:
    """Change only CUDA's min-blocks launch-bound argument.

    The lowering pipeline is kept byte-identical across the scan.  Rewriting
    generated source isolates ptxas register allocation from every TileLang IR
    and schedule decision.
    """

    rewritten, count = re.subn(
        r"(__launch_bounds__\(\s*\d+\s*,\s*)1(\s*\))",
        rf"\g<1>{blocks}\g<2>",
        source,
    )
    if count == 0:
        raise RuntimeError("launch-bound scan found no CUDA min-blocks argument")
    return rewritten, count


def extract_kernel_launch_metadata(device_mod: tvm.IRModule) -> dict[str, dict[str, Any]]:
    """Record occupancy-relevant launch metadata retained on device PrimFuncs."""

    launches: dict[str, dict[str, Any]] = {}
    for global_var, func in device_mod.functions.items():
        attrs = func.attrs
        symbol = str(attrs["global_symbol"]) if "global_symbol" in attrs else global_var.name_hint
        thread_extents: dict[str, int | str] = {}
        threads_per_block = 1
        if "thread_extent" in attrs:
            for tag, extent in attrs["thread_extent"].items():
                tag_name = str(tag)
                try:
                    extent_value: int | str = int(extent)
                except (TypeError, ValueError):
                    extent_value = str(extent)
                thread_extents[tag_name] = extent_value
                if tag_name.startswith("threadIdx.") and isinstance(extent_value, int):
                    threads_per_block *= extent_value

        dynamic_shared_memory_bytes: int | str = 0
        if "dyn_shared_memory_buf" in attrs:
            dynamic_shared_memory = attrs["dyn_shared_memory_buf"]
            try:
                dynamic_shared_memory_bytes = int(dynamic_shared_memory)
            except (TypeError, ValueError):
                dynamic_shared_memory_bytes = str(dynamic_shared_memory)

        launches[symbol] = {
            "threads_per_block": threads_per_block,
            "thread_extents": dict(sorted(thread_extents.items())),
            "dynamic_shared_memory_bytes": dynamic_shared_memory_bytes,
        }
    return dict(sorted(launches.items()))


def lower_sources(workloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workload_records: list[dict[str, Any]] = []
    compile_inputs: list[dict[str, Any]] = []
    for workload in workloads:
        prim_func = workload["prim_func"]
        primfunc_text = str(prim_func)
        record = {
            "name": workload["name"],
            "example_path": workload["example_path"].relative_to(Path.cwd()).as_posix(),
            "example_sha256": sha256_file(workload["example_path"]),
            "primfunc_sha256": sha256_text(primfunc_text),
            "configuration": workload["configuration"],
            "scope_note": workload["scope_note"],
            "function_pass_configs": {
                str(key): value for key, value in workload["pass_configs"].items()
            },
            "cases": [],
        }
        for arch in ARCHES:
            lowered_sources: dict[str, dict[str, Any]] = {}
            for mode in MODES:
                target = tvm.target.Target({"kind": "cuda", "arch": arch})
                source_mode = lowering_source_mode(mode)
                if source_mode not in lowered_sources:
                    lower_pass_configs = pass_configs_for_mode(
                        workload["pass_configs"], source_mode
                    )
                    started = time.perf_counter()
                    with tvm.transform.PassContext(
                        opt_level=3, config=lower_pass_configs
                    ), target:
                        artifact = tl.lower(
                            prim_func,
                            target=target,
                            enable_device_compile=False,
                        )
                    source = str(artifact.kernel_source or "")
                    if not source.strip():
                        raise RuntimeError(
                            f"empty CUDA source for "
                            f"{workload['name']}/{arch}/{source_mode}"
                        )
                    lowered_sources[source_mode] = {
                        "source": source,
                        "device_mod": artifact.device_mod,
                        "kernel_launches": extract_kernel_launch_metadata(
                            artifact.device_mod
                        ),
                        "lower_seconds": time.perf_counter() - started,
                    }
                lowered = lowered_sources[source_mode]
                source = lowered["source"]
                kernel_launches = lowered["kernel_launches"]
                lowered_source_sha256 = sha256_text(source)
                launch_blocks = launch_bounds_blocks(mode)
                launch_bounds_rewrites = 0
                if launch_blocks is not None:
                    source, launch_bounds_rewrites = rewrite_launch_bounds(
                        source, launch_blocks
                    )
                case_dir = RAW_DIR / workload["name"] / arch / mode
                case_dir.mkdir(parents=True, exist_ok=True)
                source_path = case_dir / "kernel.cu"
                source_path.write_text(source, encoding="utf-8")
                case = {
                    "arch": arch,
                    "mode": mode,
                    "lowering_source_mode": source_mode,
                    "lowering_reused": mode != source_mode,
                    "lower_seconds": lowered["lower_seconds"],
                    "source_sha256": sha256_text(source),
                    "lowered_source_sha256": lowered_source_sha256,
                    "source_bytes": len(source.encode()),
                    "packed_types": packed_types(source),
                    "packed_type_occurrences": len(PACKED_TYPE_RE.findall(source)),
                    "int4x2_unpack_occurrences": source.count("tl_unpack_int4x2("),
                    "cp_async_source_occurrences": len(
                        re.findall(r"\bcp_async_gs(?:_conditional)?<", source)
                    ),
                    "launch_bounds_min_blocks": launch_blocks or 1,
                    "launch_bounds_rewrites": launch_bounds_rewrites,
                    "kernel_launches": kernel_launches,
                    "source_path": source_path.relative_to(RAW_DIR).as_posix(),
                }
                record["cases"].append(case)
                compile_inputs.append(
                    {
                        "workload": workload["name"],
                        "arch": arch,
                        "mode": mode,
                        "source": source,
                        "source_path": source_path,
                        "device_mod": lowered["device_mod"],
                        "pass_configs": pass_configs_for_mode(
                            workload["pass_configs"], mode
                        ),
                        "case": case,
                    }
                )
        workload_records.append(record)
    return workload_records, compile_inputs


def parse_sass(sass: str) -> dict[str, Any]:
    opcodes = Counter(match.group("opcode").upper() for match in SASS_INSTRUCTION_RE.finditer(sass))
    if not opcodes:
        raise RuntimeError("nvdisasm output contains no recognized instructions")

    def count_prefixes(*prefixes: str) -> int:
        return sum(count for opcode, count in opcodes.items() if opcode.startswith(prefixes))

    return {
        "sass_sha256": sha256_text(sass),
        "sass_chars": len(sass),
        "instruction_count": sum(opcodes.values()),
        "groups": {
            "barrier": count_prefixes("BAR", "MBAR"),
            "warp_sync": count_prefixes("WARPSYNC"),
            "shuffle": count_prefixes("SHFL"),
            "redux": count_prefixes("REDUX"),
            "shared_load": count_prefixes("LDS", "LDSM"),
            "shared_store": count_prefixes("STS"),
            "global_load": count_prefixes("LDG"),
            "global_store": count_prefixes("STG"),
            "local_load": count_prefixes("LDL"),
            "local_store": count_prefixes("STL"),
            "async_copy": count_prefixes("LDGSTS"),
        },
        "opcodes": dict(opcodes.most_common()),
    }


def compile_case(item: dict[str, Any], nvdisasm: str) -> dict[str, Any]:
    target = tvm.target.Target({"kind": "cuda", "arch": item["arch"]})
    reset_recorder()
    started = time.perf_counter()
    cubin = bytes(
        tilelang_callback_cuda_compile(
            item["source"],
            target,
            item["pass_configs"],
            item["device_mod"],
        )
    )
    auto_launch_bounds_selection = pop_auto_launch_bounds_selection()
    compile_seconds = time.perf_counter() - started
    resources = {
        name: asdict(usage)
        for name, usage in sorted(pop_recorded().items())
    }
    case_dir = item["source_path"].parent
    cubin_path = case_dir / "kernel.cubin"
    sass_path = case_dir / "kernel.sass"
    cubin_path.write_bytes(cubin)
    disassembly = subprocess.run(
        [nvdisasm, str(cubin_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    ).stdout
    sass_path.write_text(disassembly, encoding="utf-8")
    result = {
        "compile_seconds": compile_seconds,
        "cubin_sha256": sha256_bytes(cubin),
        "cubin_bytes": len(cubin),
        "cubin_path": cubin_path.relative_to(RAW_DIR).as_posix(),
        "sass_path": sass_path.relative_to(RAW_DIR).as_posix(),
        "resources": resources,
        **parse_sass(disassembly),
    }
    if auto_launch_bounds_selection is not None:
        result["auto_launch_bounds_selection"] = auto_launch_bounds_selection
    return result


def compile_all(compile_inputs: list[dict[str, Any]]) -> None:
    nvdisasm = os.environ.get("NVDISASM") or shutil.which("nvdisasm")
    if nvdisasm is not None and not Path(nvdisasm).is_file():
        raise RuntimeError(f"configured NVDISASM does not exist: {nvdisasm}")
    if nvdisasm is None:
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            candidate = Path(cuda_home) / "bin/nvdisasm"
            if candidate.is_file():
                nvdisasm = str(candidate)
    if nvdisasm is None:
        raise RuntimeError("nvdisasm is required for the real-workload trace")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(compile_case, item, nvdisasm): item
            for item in compile_inputs
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                compiled = future.result()
            except RuntimeError as error:
                if launch_bounds_blocks(item["mode"]) is None:
                    raise
                error_text = str(error).strip() or type(error).__name__
                diagnostics = [
                    line.strip()
                    for line in error_text.splitlines()
                    if "ptxas " in line.lower()
                ]
                item["case"]["compile_error"] = (
                    "\n".join(diagnostics[-8:]) or error_text.splitlines()[-1]
                )
                print(
                    f"rejected {item['workload']}/{item['arch']}/{item['mode']}: "
                    f"{item['case']['compile_error']}"
                )
                continue
            item["case"].update(compiled)
            print(
                f"compiled {item['workload']}/{item['arch']}/{item['mode']}: "
                f"{compiled['instruction_count']} instructions, "
                f"{compiled['cubin_bytes']} cubin bytes"
            )


def enrich_comparisons(workload_records: list[dict[str, Any]]) -> dict[str, Any]:
    source_changed = 0
    sass_changed = 0
    planner_packed_gain = 0
    comparisons = 0
    inplace_source_changed = 0
    inplace_sass_changed = 0
    inplace_instruction_reduced = 0
    inplace_register_reduced = 0
    usage_modes = [mode for mode in MODES if register_usage_level(mode) is not None]
    usage_register_reduced = 0
    usage_spill_free = 0
    usage_instruction_reduced = 0
    reinterpret_source_changed = 0
    reinterpret_sass_changed = 0
    reinterpret_instruction_reduced = 0
    reinterpret_register_reduced = 0
    int4x2_source_changed = 0
    int4x2_sass_changed = 0
    int4x2_instruction_reduced = 0
    int4x2_register_reduced = 0
    scalar_exp2_source_changed = 0
    scalar_exp2_instruction_nonregressed = 0
    scalar_exp2_cubin_nonregressed = 0
    scalar_exp2_spill_nonregressed = 0
    scalar_exp2_strict_improvements = 0
    nvcc_vectorization_sass_changed = 0
    nvcc_vectorization_instruction_nonregressed = 0
    nvcc_vectorization_spill_nonregressed = 0
    nvcc_vectorization_strict_improvements = 0
    predicated_async_source_restored = 0
    predicated_async_sass_restored = 0
    predicated_async_expected_controls = 0
    for workload in workload_records:
        by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
        workload["comparisons"] = []
        for arch in ARCHES:
            planner = by_key[(arch, "planner")]
            legacy = by_key[(arch, "legacy")]
            source_delta = planner["source_sha256"] != legacy["source_sha256"]
            sass_delta = planner["sass_sha256"] != legacy["sass_sha256"]
            packed_delta = planner["packed_type_occurrences"] - legacy["packed_type_occurrences"]
            group_delta = {
                group: planner["groups"].get(group, 0) - legacy["groups"].get(group, 0)
                for group in sorted(set(planner["groups"]) | set(legacy["groups"]))
            }
            workload["comparisons"].append(
                {
                    "arch": arch,
                    "source_changed": source_delta,
                    "sass_changed": sass_delta,
                    "planner_minus_legacy": {
                        "source_bytes": planner["source_bytes"] - legacy["source_bytes"],
                        "packed_type_occurrences": packed_delta,
                        "cubin_bytes": planner["cubin_bytes"] - legacy["cubin_bytes"],
                        "instruction_count": planner["instruction_count"] - legacy["instruction_count"],
                        "groups": group_delta,
                    },
                }
            )
            source_changed += int(source_delta)
            sass_changed += int(sass_delta)
            planner_packed_gain += int(packed_delta > 0)
            comparisons += 1
        if "planner_inplace" in MODES:
            workload["inplace_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                inplace = by_key[(arch, "planner_inplace")]
                source_delta = inplace["source_sha256"] != planner["source_sha256"]
                sass_delta = inplace["sass_sha256"] != planner["sass_sha256"]
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner["resources"].values()),
                    default=0,
                )
                inplace_regs = max(
                    (int(item["n_regs"]) for item in inplace["resources"].values()),
                    default=0,
                )
                workload["inplace_comparisons"].append(
                    {
                        "arch": arch,
                        "source_changed": source_delta,
                        "sass_changed": sass_delta,
                        "inplace_minus_planner": {
                            "source_bytes": inplace["source_bytes"] - planner["source_bytes"],
                            "cubin_bytes": inplace["cubin_bytes"] - planner["cubin_bytes"],
                            "instruction_count": inplace["instruction_count"] - planner["instruction_count"],
                            "max_registers": inplace_regs - planner_regs,
                        },
                    }
                )
                inplace_source_changed += int(source_delta)
                inplace_sass_changed += int(sass_delta)
                inplace_instruction_reduced += int(
                    inplace["instruction_count"] < planner["instruction_count"]
                )
                inplace_register_reduced += int(inplace_regs < planner_regs)
        if usage_modes:
            workload["register_usage_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                planner_resources = list(planner["resources"].values())
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner_resources),
                    default=0,
                )
                for mode in usage_modes:
                    candidate = by_key[(arch, mode)]
                    resources = list(candidate["resources"].values())
                    candidate_regs = max(
                        (int(item["n_regs"]) for item in resources),
                        default=0,
                    )
                    spill_stores = sum(int(item["spill_store_bytes"]) for item in resources)
                    spill_loads = sum(int(item["spill_load_bytes"]) for item in resources)
                    comparison = {
                        "arch": arch,
                        "mode": mode,
                        "register_usage_level": register_usage_level(mode),
                        "source_changed": candidate["source_sha256"] != planner["source_sha256"],
                        "candidate_minus_planner": {
                            "cubin_bytes": candidate["cubin_bytes"] - planner["cubin_bytes"],
                            "instruction_count": candidate["instruction_count"]
                            - planner["instruction_count"],
                            "max_registers": candidate_regs - planner_regs,
                            "spill_store_bytes": spill_stores,
                            "spill_load_bytes": spill_loads,
                        },
                    }
                    workload["register_usage_comparisons"].append(comparison)
                    usage_register_reduced += int(candidate_regs < planner_regs)
                    usage_spill_free += int(spill_stores == 0 and spill_loads == 0)
                    usage_instruction_reduced += int(
                        candidate["instruction_count"] < planner["instruction_count"]
                    )
        if "planner_reinterpret_legacy" in MODES:
            workload["reinterpret_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                legacy_reinterpret = by_key[(arch, "planner_reinterpret_legacy")]
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner["resources"].values()),
                    default=0,
                )
                legacy_regs = max(
                    (
                        int(item["n_regs"])
                        for item in legacy_reinterpret["resources"].values()
                    ),
                    default=0,
                )
                comparison = {
                    "arch": arch,
                    "source_changed": planner["source_sha256"]
                    != legacy_reinterpret["source_sha256"],
                    "sass_changed": planner["sass_sha256"]
                    != legacy_reinterpret["sass_sha256"],
                    "planner_minus_legacy_reinterpret": {
                        "source_bytes": planner["source_bytes"]
                        - legacy_reinterpret["source_bytes"],
                        "cubin_bytes": planner["cubin_bytes"]
                        - legacy_reinterpret["cubin_bytes"],
                        "instruction_count": planner["instruction_count"]
                        - legacy_reinterpret["instruction_count"],
                        "max_registers": planner_regs - legacy_regs,
                    },
                }
                workload["reinterpret_comparisons"].append(comparison)
                reinterpret_source_changed += int(comparison["source_changed"])
                reinterpret_sass_changed += int(comparison["sass_changed"])
                reinterpret_instruction_reduced += int(
                    planner["instruction_count"] < legacy_reinterpret["instruction_count"]
                )
                reinterpret_register_reduced += int(planner_regs < legacy_regs)
        if "planner_int4x2_unpack_legacy" in MODES:
            workload["int4x2_unpack_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                legacy_unpack = by_key[(arch, "planner_int4x2_unpack_legacy")]
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner["resources"].values()),
                    default=0,
                )
                legacy_regs = max(
                    (int(item["n_regs"]) for item in legacy_unpack["resources"].values()),
                    default=0,
                )
                comparison = {
                    "arch": arch,
                    "source_changed": planner["source_sha256"]
                    != legacy_unpack["source_sha256"],
                    "sass_changed": planner["sass_sha256"]
                    != legacy_unpack["sass_sha256"],
                    "planner_minus_legacy_int4x2_unpack": {
                        "source_bytes": planner["source_bytes"]
                        - legacy_unpack["source_bytes"],
                        "cubin_bytes": planner["cubin_bytes"]
                        - legacy_unpack["cubin_bytes"],
                        "instruction_count": planner["instruction_count"]
                        - legacy_unpack["instruction_count"],
                        "max_registers": planner_regs - legacy_regs,
                        "helper_occurrences": planner["int4x2_unpack_occurrences"]
                        - legacy_unpack["int4x2_unpack_occurrences"],
                    },
                }
                workload["int4x2_unpack_comparisons"].append(comparison)
                int4x2_source_changed += int(comparison["source_changed"])
                int4x2_sass_changed += int(comparison["sass_changed"])
                int4x2_instruction_reduced += int(
                    planner["instruction_count"] < legacy_unpack["instruction_count"]
                )
                int4x2_register_reduced += int(planner_regs < legacy_regs)
        if "planner_scalar_exp2_vectorized" in MODES:
            workload["scalar_exp2_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                vectorized = by_key[(arch, "planner_scalar_exp2_vectorized")]
                planner_resources = list(planner["resources"].values())
                vectorized_resources = list(vectorized["resources"].values())
                planner_spills = sum(
                    int(item["spill_store_bytes"]) + int(item["spill_load_bytes"])
                    for item in planner_resources
                )
                vectorized_spills = sum(
                    int(item["spill_store_bytes"]) + int(item["spill_load_bytes"])
                    for item in vectorized_resources
                )
                source_delta = planner["source_sha256"] != vectorized["source_sha256"]
                instruction_safe = (
                    planner["instruction_count"] <= vectorized["instruction_count"]
                )
                cubin_safe = planner["cubin_bytes"] <= vectorized["cubin_bytes"]
                spill_safe = planner_spills <= vectorized_spills
                strict = (
                    planner["instruction_count"] < vectorized["instruction_count"]
                    or planner["cubin_bytes"] < vectorized["cubin_bytes"]
                    or planner_spills < vectorized_spills
                )
                workload["scalar_exp2_comparisons"].append(
                    {
                        "arch": arch,
                        "source_changed": source_delta,
                        "planner_minus_vectorized": {
                            "source_bytes": planner["source_bytes"]
                            - vectorized["source_bytes"],
                            "cubin_bytes": planner["cubin_bytes"]
                            - vectorized["cubin_bytes"],
                            "instruction_count": planner["instruction_count"]
                            - vectorized["instruction_count"],
                            "spill_bytes": planner_spills - vectorized_spills,
                        },
                        "instruction_nonregressed": instruction_safe,
                        "cubin_nonregressed": cubin_safe,
                        "spill_nonregressed": spill_safe,
                        "strict_improvement": strict,
                    }
                )
                scalar_exp2_source_changed += int(source_delta)
                scalar_exp2_instruction_nonregressed += int(instruction_safe)
                scalar_exp2_cubin_nonregressed += int(cubin_safe)
                scalar_exp2_spill_nonregressed += int(spill_safe)
                scalar_exp2_strict_improvements += int(strict)
        if "planner_nvcc_extra_vectorization" in MODES:
            workload["nvcc_extra_vectorization_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                candidate = by_key[(arch, "planner_nvcc_extra_vectorization")]
                planner_spills = sum(
                    int(item["spill_store_bytes"]) + int(item["spill_load_bytes"])
                    for item in planner["resources"].values()
                )
                candidate_spills = sum(
                    int(item["spill_store_bytes"]) + int(item["spill_load_bytes"])
                    for item in candidate["resources"].values()
                )
                if candidate["source_sha256"] != planner["source_sha256"]:
                    raise RuntimeError(
                        f"NVCC vectorization mode changed CUDA source for "
                        f"{workload['name']}/{arch}"
                    )
                if candidate["kernel_launches"] != planner["kernel_launches"]:
                    raise RuntimeError(
                        f"NVCC vectorization mode changed launch metadata for "
                        f"{workload['name']}/{arch}"
                    )
                instruction_safe = (
                    candidate["instruction_count"] <= planner["instruction_count"]
                )
                spill_safe = candidate_spills <= planner_spills
                strict = (
                    candidate["instruction_count"] < planner["instruction_count"]
                    or candidate["cubin_bytes"] < planner["cubin_bytes"]
                    or candidate_spills < planner_spills
                )
                comparison = {
                    "arch": arch,
                    "source_isolated": True,
                    "launch_metadata_isolated": True,
                    "sass_changed": candidate["sass_sha256"]
                    != planner["sass_sha256"],
                    "candidate_minus_planner": {
                        "cubin_bytes": candidate["cubin_bytes"]
                        - planner["cubin_bytes"],
                        "instruction_count": candidate["instruction_count"]
                        - planner["instruction_count"],
                        "spill_bytes": candidate_spills - planner_spills,
                    },
                    "instruction_nonregressed": instruction_safe,
                    "spill_nonregressed": spill_safe,
                    "strict_improvement": strict,
                }
                workload["nvcc_extra_vectorization_comparisons"].append(comparison)
                nvcc_vectorization_sass_changed += int(comparison["sass_changed"])
                nvcc_vectorization_instruction_nonregressed += int(instruction_safe)
                nvcc_vectorization_spill_nonregressed += int(spill_safe)
                nvcc_vectorization_strict_improvements += int(strict)
        if (
            "planner_sync_predicated_copy" in MODES
            and workload["name"]
            == "varlen_attention_guarded_stage_bwd_repro"
        ):
            workload["predicated_async_copy_comparisons"] = []
            for arch in ARCHES:
                planner = by_key[(arch, "planner")]
                synchronous = by_key[(arch, "planner_sync_predicated_copy")]
                sm_version = int(re.match(r"sm_(\d+)", arch).group(1))
                supports_cp_async = sm_version >= 80
                source_restored = (
                    planner["cp_async_source_occurrences"]
                    > synchronous["cp_async_source_occurrences"]
                )
                sass_restored = (
                    planner["groups"]["async_copy"]
                    > synchronous["groups"]["async_copy"]
                )
                expected_control = (
                    planner["cp_async_source_occurrences"] == 0
                    and synchronous["cp_async_source_occurrences"] == 0
                    and planner["groups"]["async_copy"] == 0
                    and synchronous["groups"]["async_copy"] == 0
                    and planner["source_sha256"] == synchronous["source_sha256"]
                    and planner["kernel_launches"]
                    == synchronous["kernel_launches"]
                    and planner["instruction_count"]
                    == synchronous["instruction_count"]
                    and planner["opcodes"] == synchronous["opcodes"]
                    and planner["groups"] == synchronous["groups"]
                    and planner["resources"] == synchronous["resources"]
                    and planner["cubin_bytes"] == synchronous["cubin_bytes"]
                )
                workload["predicated_async_copy_comparisons"].append(
                    {
                        "arch": arch,
                        "supports_cp_async": supports_cp_async,
                        "source_restored": source_restored,
                        "sass_restored": sass_restored,
                        "expected_control": expected_control,
                        "planner_minus_synchronous": {
                            "cp_async_source_occurrences": planner[
                                "cp_async_source_occurrences"
                            ]
                            - synchronous["cp_async_source_occurrences"],
                            "async_copy_instructions": planner["groups"][
                                "async_copy"
                            ]
                            - synchronous["groups"]["async_copy"],
                            "global_load_instructions": planner["groups"][
                                "global_load"
                            ]
                            - synchronous["groups"]["global_load"],
                            "shared_store_instructions": planner["groups"][
                                "shared_store"
                            ]
                            - synchronous["groups"]["shared_store"],
                            "instruction_count": planner["instruction_count"]
                            - synchronous["instruction_count"],
                        },
                    }
                )
                predicated_async_source_restored += int(source_restored)
                predicated_async_sass_restored += int(sass_restored)
                predicated_async_expected_controls += int(expected_control)
    targeted_predicated_scan = (
        set(MODES)
        == {"planner", "legacy", "planner_sync_predicated_copy"}
        and len(workload_records) == 1
        and workload_records[0]["name"]
        == "varlen_attention_guarded_stage_bwd_repro"
    )
    if (
        source_changed == 0 or sass_changed == 0 or planner_packed_gain == 0
    ) and not targeted_predicated_scan:
        raise RuntimeError(
            "planner/legacy trace produced no material source, SASS, or packed-type difference"
        )
    if "planner_int4x2_unpack_legacy" in MODES:
        w4a8 = next(
            workload
            for workload in workload_records
            if workload["name"] == "dequant_gemm_w4a8_4096"
        )
        by_key = {(case["arch"], case["mode"]): case for case in w4a8["cases"]}
        for arch in ARCHES:
            planner = by_key[(arch, "planner")]
            legacy_unpack = by_key[(arch, "planner_int4x2_unpack_legacy")]
            sm_version = int(re.match(r"sm_(\d+)", arch).group(1))
            if sm_version >= 100 and planner["int4x2_unpack_occurrences"] <= 0:
                raise RuntimeError(f"int4x2 unpack peephole did not trigger for {arch}")
            if sm_version < 100 and planner["int4x2_unpack_occurrences"] != 0:
                raise RuntimeError(f"int4x2 unpack peephole unexpectedly triggered for {arch}")
            if legacy_unpack["int4x2_unpack_occurrences"] != 0:
                raise RuntimeError(f"int4x2 unpack rollback still emitted helper for {arch}")
    if "planner_scalar_exp2_vectorized" in MODES:
        scalar_exp2_comparisons = len(ARCHES) * len(workload_records)
        if (
            scalar_exp2_source_changed == 0
            or scalar_exp2_instruction_nonregressed != scalar_exp2_comparisons
            or scalar_exp2_cubin_nonregressed != scalar_exp2_comparisons
            or scalar_exp2_spill_nonregressed != scalar_exp2_comparisons
            or scalar_exp2_strict_improvements == 0
        ):
            raise RuntimeError(
                "scalar-exp2 fallback failed source-change, machine-code "
                "nonregression, or strict-improvement acceptance"
            )
    if "planner_sync_predicated_copy" in MODES:
        candidate = next(
            (
                workload
                for workload in workload_records
                if workload["name"]
                == "varlen_attention_guarded_stage_bwd_repro"
            ),
            None,
        )
        if candidate is None:
            raise RuntimeError(
                "predicated cp.async scan requires the guarded-stage workload"
            )
        expected_supported = sum(
            int(int(re.match(r"sm_(\d+)", arch).group(1)) >= 80)
            for arch in ARCHES
        )
        expected_controls = len(ARCHES) - expected_supported
        if (
            predicated_async_source_restored != expected_supported
            or predicated_async_sass_restored != expected_supported
            or predicated_async_expected_controls != expected_controls
        ):
            raise RuntimeError(
                "predicated cp.async lowering failed source, SASS, or "
                "unsupported-architecture control acceptance"
            )
    return {
        "comparisons": comparisons,
        "source_changed": source_changed,
        "sass_changed": sass_changed,
        "planner_packed_gain": planner_packed_gain,
        "inplace_comparisons": len(ARCHES) * len(workload_records)
        if "planner_inplace" in MODES
        else 0,
        "inplace_source_changed": inplace_source_changed,
        "inplace_sass_changed": inplace_sass_changed,
        "inplace_instruction_reduced": inplace_instruction_reduced,
        "inplace_register_reduced": inplace_register_reduced,
        "register_usage_comparisons": len(ARCHES) * len(workload_records) * len(usage_modes),
        "register_usage_register_reduced": usage_register_reduced,
        "register_usage_spill_free": usage_spill_free,
        "register_usage_instruction_reduced": usage_instruction_reduced,
        "reinterpret_comparisons": len(ARCHES) * len(workload_records)
        if "planner_reinterpret_legacy" in MODES
        else 0,
        "reinterpret_source_changed": reinterpret_source_changed,
        "reinterpret_sass_changed": reinterpret_sass_changed,
        "reinterpret_instruction_reduced": reinterpret_instruction_reduced,
        "reinterpret_register_reduced": reinterpret_register_reduced,
        "int4x2_unpack_comparisons": len(ARCHES) * len(workload_records)
        if "planner_int4x2_unpack_legacy" in MODES
        else 0,
        "int4x2_unpack_source_changed": int4x2_source_changed,
        "int4x2_unpack_sass_changed": int4x2_sass_changed,
        "int4x2_unpack_instruction_reduced": int4x2_instruction_reduced,
        "int4x2_unpack_register_reduced": int4x2_register_reduced,
        "scalar_exp2_comparisons": len(ARCHES) * len(workload_records)
        if "planner_scalar_exp2_vectorized" in MODES
        else 0,
        "scalar_exp2_source_changed": scalar_exp2_source_changed,
        "scalar_exp2_instruction_nonregressed": scalar_exp2_instruction_nonregressed,
        "scalar_exp2_cubin_nonregressed": scalar_exp2_cubin_nonregressed,
        "scalar_exp2_spill_nonregressed": scalar_exp2_spill_nonregressed,
        "scalar_exp2_strict_improvements": scalar_exp2_strict_improvements,
        "nvcc_extra_vectorization_comparisons": len(ARCHES) * len(workload_records)
        if "planner_nvcc_extra_vectorization" in MODES
        else 0,
        "nvcc_extra_vectorization_sass_changed": nvcc_vectorization_sass_changed,
        "nvcc_extra_vectorization_instruction_nonregressed": (
            nvcc_vectorization_instruction_nonregressed
        ),
        "nvcc_extra_vectorization_spill_nonregressed": (
            nvcc_vectorization_spill_nonregressed
        ),
        "nvcc_extra_vectorization_strict_improvements": (
            nvcc_vectorization_strict_improvements
        ),
        "predicated_async_copy_comparisons": len(ARCHES)
        if "planner_sync_predicated_copy" in MODES
        else 0,
        "predicated_async_source_restored": predicated_async_source_restored,
        "predicated_async_sass_restored": predicated_async_sass_restored,
        "predicated_async_expected_controls": predicated_async_expected_controls,
    }


def enrich_launch_bounds_scan(
    workload_records: list[dict[str, Any]], aggregate: dict[str, Any]
) -> None:
    launch_modes = [mode for mode in MODES if launch_bounds_blocks(mode) is not None]
    if not launch_modes:
        aggregate.update(
            {
                "launch_bounds_comparisons": 0,
                "launch_bounds_register_reduced": 0,
                "launch_bounds_instruction_nonregressed": 0,
                "launch_bounds_spill_free": 0,
                "launch_bounds_static_candidates": 0,
                "launch_bounds_compile_failures": 0,
            }
        )
        return

    register_reduced = 0
    instruction_nonregressed = 0
    spill_free = 0
    static_candidates = 0
    compile_failures = 0
    comparisons = 0

    def max_resource(case: dict[str, Any], field: str) -> int:
        return max(
            (int(resource[field]) for resource in case["resources"].values()),
            default=0,
        )

    def sum_resource(case: dict[str, Any], field: str) -> int:
        return sum(int(resource[field]) for resource in case["resources"].values())

    for workload in workload_records:
        by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
        workload["launch_bounds_comparisons"] = []
        for arch in ARCHES:
            planner = by_key[(arch, "planner")]
            planner_regs = max_resource(planner, "n_regs")
            for mode in launch_modes:
                candidate = by_key[(arch, mode)]
                source_isolated = (
                    candidate["lowered_source_sha256"]
                    == planner["lowered_source_sha256"]
                )
                launch_metadata_isolated = (
                    candidate["kernel_launches"] == planner["kernel_launches"]
                )
                if not source_isolated:
                    raise RuntimeError(
                        f"launch-bound mode changed TileLang lowering for "
                        f"{workload['name']}/{arch}/{mode}"
                    )
                if not launch_metadata_isolated:
                    raise RuntimeError(
                        f"launch-bound mode changed kernel launch metadata for "
                        f"{workload['name']}/{arch}/{mode}"
                    )
                if compile_error := candidate.get("compile_error"):
                    workload["launch_bounds_comparisons"].append(
                        {
                            "arch": arch,
                            "mode": mode,
                            "min_blocks_per_sm": launch_bounds_blocks(mode),
                            "source_isolated": source_isolated,
                            "launch_metadata_isolated": launch_metadata_isolated,
                            "static_candidate": False,
                            "compile_error": compile_error,
                        }
                    )
                    compile_failures += 1
                    comparisons += 1
                    continue
                candidate_regs = max_resource(candidate, "n_regs")
                spill_stores = sum_resource(candidate, "spill_store_bytes")
                spill_loads = sum_resource(candidate, "spill_load_bytes")
                same_entries = candidate["resources"].keys() == planner["resources"].keys()
                reduced = same_entries and any(
                    int(candidate["resources"][name]["n_regs"])
                    < int(planner["resources"][name]["n_regs"])
                    for name in planner["resources"]
                )
                registers_nonregressed = same_entries and all(
                    int(candidate["resources"][name]["n_regs"])
                    <= int(planner["resources"][name]["n_regs"])
                    for name in planner["resources"]
                )
                local_memory_nonregressed = same_entries and all(
                    int(candidate["resources"][name][field])
                    <= int(planner["resources"][name][field])
                    for name in planner["resources"]
                    for field in ("local_bytes", "stack_frame_bytes")
                )
                nonregressed = (
                    candidate["instruction_count"] <= planner["instruction_count"]
                )
                zero_spill = spill_stores == 0 and spill_loads == 0
                cubin_nonregressed = candidate["cubin_bytes"] <= planner["cubin_bytes"]
                viable = (
                    reduced
                    and registers_nonregressed
                    and local_memory_nonregressed
                    and nonregressed
                    and zero_spill
                    and cubin_nonregressed
                )
                comparison = {
                    "arch": arch,
                    "mode": mode,
                    "min_blocks_per_sm": launch_bounds_blocks(mode),
                    "source_isolated": source_isolated,
                    "launch_metadata_isolated": launch_metadata_isolated,
                    "static_candidate": viable,
                    "same_resource_entries": same_entries,
                    "registers_nonregressed": registers_nonregressed,
                    "local_memory_nonregressed": local_memory_nonregressed,
                    "cubin_nonregressed": cubin_nonregressed,
                    "candidate_minus_planner": {
                        "cubin_bytes": candidate["cubin_bytes"]
                        - planner["cubin_bytes"],
                        "instruction_count": candidate["instruction_count"]
                        - planner["instruction_count"],
                        "max_registers": candidate_regs - planner_regs,
                        "spill_store_bytes": spill_stores,
                        "spill_load_bytes": spill_loads,
                    },
                }
                workload["launch_bounds_comparisons"].append(comparison)
                register_reduced += int(reduced)
                instruction_nonregressed += int(nonregressed)
                spill_free += int(zero_spill)
                static_candidates += int(viable)
                comparisons += 1

    aggregate.update(
        {
            "launch_bounds_comparisons": comparisons,
            "launch_bounds_register_reduced": register_reduced,
            "launch_bounds_instruction_nonregressed": instruction_nonregressed,
            "launch_bounds_spill_free": spill_free,
            "launch_bounds_static_candidates": static_candidates,
            "launch_bounds_compile_failures": compile_failures,
        }
    )


def enrich_auto_launch_bounds(
    workload_records: list[dict[str, Any]], aggregate: dict[str, Any]
) -> None:
    if "planner_auto_lb" not in MODES:
        aggregate.update(
            {
                "auto_launch_bounds_comparisons": 0,
                "auto_launch_bounds_candidate_selections": 0,
                "auto_launch_bounds_baseline_selections": 0,
                "auto_launch_bounds_instruction_nonregressed": 0,
                "auto_launch_bounds_spill_nonregressed": 0,
            }
        )
        return
    if "planner_lb2" not in MODES:
        raise RuntimeError("planner_auto_lb validation requires planner_lb2")

    candidate_selections = 0
    baseline_selections = 0
    instruction_nonregressed = 0
    spill_nonregressed = 0
    comparisons = 0

    def sum_resource(case: dict[str, Any], field: str) -> int:
        return sum(int(resource[field]) for resource in case["resources"].values())

    def machine_signature(case: dict[str, Any]) -> dict[str, Any]:
        return {
            "cubin_bytes": case["cubin_bytes"],
            "instruction_count": case["instruction_count"],
            "groups": case["groups"],
            "opcodes": case["opcodes"],
            "resources": case["resources"],
        }

    for workload in workload_records:
        by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
        workload["auto_launch_bounds_comparisons"] = []
        for arch in ARCHES:
            planner = by_key[(arch, "planner")]
            candidate = by_key[(arch, "planner_lb2")]
            automatic = by_key[(arch, "planner_auto_lb")]
            source_isolated = (
                automatic["lowered_source_sha256"] == planner["lowered_source_sha256"]
            )
            launch_metadata_isolated = (
                automatic["kernel_launches"] == planner["kernel_launches"]
            )
            if not source_isolated:
                raise RuntimeError(
                    f"automatic launch bounds changed TileLang lowering for "
                    f"{workload['name']}/{arch}"
                )
            if not launch_metadata_isolated:
                raise RuntimeError(
                    f"automatic launch bounds changed kernel launch metadata for "
                    f"{workload['name']}/{arch}"
                )

            selected_min_blocks = automatic.get("auto_launch_bounds_selection")
            if selected_min_blocks == 1:
                selection = "planner"
                reference = planner
                baseline_selections += 1
            elif selected_min_blocks == 2:
                selection = "planner_lb2"
                reference = candidate
                candidate_selections += 1
            else:
                raise RuntimeError(
                    f"automatic launch-bound selection metadata is invalid for "
                    f"{workload['name']}/{arch}: {selected_min_blocks}"
                )
            if machine_signature(automatic) != machine_signature(reference):
                raise RuntimeError(
                    f"automatic launch-bound machine code differs from its selected "
                    f"reference for {workload['name']}/{arch}"
                )

            spill_stores = sum_resource(automatic, "spill_store_bytes")
            spill_loads = sum_resource(automatic, "spill_load_bytes")
            planner_spill_stores = sum_resource(planner, "spill_store_bytes")
            planner_spill_loads = sum_resource(planner, "spill_load_bytes")
            nonregressed = (
                automatic["instruction_count"] <= planner["instruction_count"]
            )
            spill_safe = (
                spill_stores <= planner_spill_stores
                and spill_loads <= planner_spill_loads
                and (
                    selection == "planner"
                    or (spill_stores == 0 and spill_loads == 0)
                )
            )
            workload["auto_launch_bounds_comparisons"].append(
                {
                    "arch": arch,
                    "selection": selection,
                    "source_isolated": source_isolated,
                    "launch_metadata_isolated": launch_metadata_isolated,
                    "instruction_delta": automatic["instruction_count"]
                    - planner["instruction_count"],
                    "spill_store_bytes": spill_stores,
                    "spill_load_bytes": spill_loads,
                    "spill_store_delta": spill_stores - planner_spill_stores,
                    "spill_load_delta": spill_loads - planner_spill_loads,
                }
            )
            instruction_nonregressed += int(nonregressed)
            spill_nonregressed += int(spill_safe)
            comparisons += 1

    if instruction_nonregressed != comparisons or spill_nonregressed != comparisons:
        raise RuntimeError(
            "automatic launch-bound selector produced an instruction regression or spill"
        )
    aggregate.update(
        {
            "auto_launch_bounds_comparisons": comparisons,
            "auto_launch_bounds_candidate_selections": candidate_selections,
            "auto_launch_bounds_baseline_selections": baseline_selections,
            "auto_launch_bounds_instruction_nonregressed": instruction_nonregressed,
            "auto_launch_bounds_spill_nonregressed": spill_nonregressed,
        }
    )


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Frozen real-workload vectorization trace",
        "",
        f"- Source commit: `{payload['source_sha']}`",
        f"- Architectures: `{', '.join(payload['architectures'])}`",
        f"- Modes: `{', '.join(payload['modes'])}` from one installed TileLang build",
        "- Device compilation: yes, through TileLang's CUDA callback",
        "- GPU execution: no",
        "",
        "| Workload | Arch | packed types legacy→planner | instructions legacy→planner | registers legacy→planner | local load/store legacy→planner |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for workload in payload["workloads"]:
        by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
        for arch in payload["architectures"]:
            planner = by_key[(arch, "planner")]
            legacy = by_key[(arch, "legacy")]

            def register_count(case: dict[str, Any]) -> str:
                values = sorted({int(item["n_regs"]) for item in case["resources"].values()})
                return "/".join(map(str, values)) if values else "n/a"

            lines.append(
                f"| `{workload['name']}` | `{arch}` | "
                f"{legacy['packed_type_occurrences']}→{planner['packed_type_occurrences']} | "
                f"{legacy['instruction_count']}→{planner['instruction_count']} | "
                f"{register_count(legacy)}→{register_count(planner)} | "
                f"{legacy['groups']['local_load']}/{legacy['groups']['local_store']}→"
                f"{planner['groups']['local_load']}/{planner['groups']['local_store']} |"
            )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "This report proves exact frozen-program elaboration, CUDA source changes, device compilation, CUBIN/SASS changes, and compiler resource usage. Runtime latency and end-to-end speedup require real GPU execution.",
            "",
        ]
    )
    if "planner_inplace" in payload["modes"]:
        lines.extend(
            [
                "## Experimental inplace detector",
                "",
                "| Workload | Arch | instructions planner→inplace | registers planner→inplace | source/SASS changed |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                inplace = by_key[(arch, "planner_inplace")]

                def max_registers(case: dict[str, Any]) -> str:
                    values = [int(item["n_regs"]) for item in case["resources"].values()]
                    return str(max(values)) if values else "n/a"

                lines.append(
                    f"| `{workload['name']}` | `{arch}` | "
                    f"{planner['instruction_count']}→{inplace['instruction_count']} | "
                    f"{max_registers(planner)}→{max_registers(inplace)} | "
                    f"{planner['source_sha256'] != inplace['source_sha256']}/"
                    f"{planner['sass_sha256'] != inplace['sass_sha256']} |"
                )
        lines.append("")
    usage_modes = [mode for mode in payload["modes"] if register_usage_level(mode) is not None]
    if usage_modes:
        lines.extend(
            [
                "## PTXAS register-usage scan",
                "",
                "| Workload | Arch | level | instructions planner→candidate | registers planner→candidate | spill stores/loads (bytes) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                planner_regs = max(
                    (int(item["n_regs"]) for item in planner["resources"].values()),
                    default=0,
                )
                for mode in usage_modes:
                    candidate = by_key[(arch, mode)]
                    resources = list(candidate["resources"].values())
                    candidate_regs = max(
                        (int(item["n_regs"]) for item in resources),
                        default=0,
                    )
                    spill_stores = sum(int(item["spill_store_bytes"]) for item in resources)
                    spill_loads = sum(int(item["spill_load_bytes"]) for item in resources)
                    lines.append(
                        f"| `{workload['name']}` | `{arch}` | {register_usage_level(mode)} | "
                        f"{planner['instruction_count']}→{candidate['instruction_count']} | "
                        f"{planner_regs}→{candidate_regs} | {spill_stores}/{spill_loads} |"
                    )
        lines.append("")
    launch_modes = [
        mode for mode in payload["modes"] if launch_bounds_blocks(mode) is not None
    ]
    if launch_modes:
        lines.extend(
            [
                "## Kernel launch resource context",
                "",
                "| Workload | Arch | kernel | threads/block | dynamic shared bytes |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {
                (case["arch"], case["mode"]): case for case in workload["cases"]
            }
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                for kernel_name, launch in planner["kernel_launches"].items():
                    lines.append(
                        f"| `{workload['name']}` | `{arch}` | `{kernel_name}` | "
                        f"{launch['threads_per_block']} | "
                        f"{launch['dynamic_shared_memory_bytes']} |"
                    )
        lines.append("")
        lines.extend(
            [
                "## CUDA launch-bound scan",
                "",
                "| Workload | Arch | min blocks/SM | instructions planner→candidate | registers planner→candidate | spill stores/loads (bytes) | static candidate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {
                (case["arch"], case["mode"]): case for case in workload["cases"]
            }
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                planner_regs = max(
                    (
                        int(resource["n_regs"])
                        for resource in planner["resources"].values()
                    ),
                    default=0,
                )
                comparisons = {
                    comparison["mode"]: comparison
                    for comparison in workload["launch_bounds_comparisons"]
                    if comparison["arch"] == arch
                }
                for mode in launch_modes:
                    candidate = by_key[(arch, mode)]
                    comparison = comparisons[mode]
                    if compile_error := comparison.get("compile_error"):
                        lines.append(
                            f"| `{workload['name']}` | `{arch}` | "
                            f"{launch_bounds_blocks(mode)} | compile failed | "
                            f"compile failed | n/a | False |"
                        )
                        continue
                    candidate_regs = max(
                        (
                            int(resource["n_regs"])
                            for resource in candidate["resources"].values()
                        ),
                        default=0,
                    )
                    delta = comparison["candidate_minus_planner"]
                    lines.append(
                        f"| `{workload['name']}` | `{arch}` | "
                        f"{launch_bounds_blocks(mode)} | "
                        f"{planner['instruction_count']}→{candidate['instruction_count']} | "
                        f"{planner_regs}→{candidate_regs} | "
                        f"{delta['spill_store_bytes']}/{delta['spill_load_bytes']} | "
                        f"{comparison['static_candidate']} |"
                    )
        lines.append("")
    if "planner_auto_lb" in payload["modes"]:
        lines.extend(
            [
                "## Automatic CUDA launch-bound selection",
                "",
                "| Workload | Arch | selected binary | instruction delta | spill stores/loads (bytes) | spill delta |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            for comparison in workload["auto_launch_bounds_comparisons"]:
                lines.append(
                    f"| `{workload['name']}` | `{comparison['arch']}` | "
                    f"`{comparison['selection']}` | "
                    f"{comparison['instruction_delta']} | "
                    f"{comparison['spill_store_bytes']}/"
                    f"{comparison['spill_load_bytes']} | "
                    f"{comparison['spill_store_delta']}/"
                    f"{comparison['spill_load_delta']} |"
                )
        lines.append("")
    if "planner_reinterpret_legacy" in payload["modes"]:
        lines.extend(
            [
                "## Transparent reinterpret planning",
                "",
                "| Workload | Arch | packed types legacy→transparent | instructions legacy→transparent | registers legacy→transparent |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                legacy_reinterpret = by_key[(arch, "planner_reinterpret_legacy")]

                def register_count(case: dict[str, Any]) -> str:
                    values = sorted(
                        {int(item["n_regs"]) for item in case["resources"].values()}
                    )
                    return "/".join(map(str, values)) if values else "n/a"

                lines.append(
                    f"| `{workload['name']}` | `{arch}` | "
                    f"{legacy_reinterpret['packed_type_occurrences']}→"
                    f"{planner['packed_type_occurrences']} | "
                    f"{legacy_reinterpret['instruction_count']}→"
                    f"{planner['instruction_count']} | "
                    f"{register_count(legacy_reinterpret)}→{register_count(planner)} |"
                )
        lines.append("")
    if "planner_scalar_exp2_vectorized" in payload["modes"]:
        lines.extend(
            [
                "## Scalar FP32 exp2 local-loop fallback",
                "",
                "| Workload | Arch | instructions vectorized→fallback | CUBIN bytes vectorized→fallback | spill bytes vectorized→fallback |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {
                (case["arch"], case["mode"]): case for case in workload["cases"]
            }
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                vectorized = by_key[(arch, "planner_scalar_exp2_vectorized")]

                def spill_bytes(case: dict[str, Any]) -> int:
                    return sum(
                        int(item["spill_store_bytes"])
                        + int(item["spill_load_bytes"])
                        for item in case["resources"].values()
                    )

                lines.append(
                    f"| `{workload['name']}` | `{arch}` | "
                    f"{vectorized['instruction_count']}→{planner['instruction_count']} | "
                    f"{vectorized['cubin_bytes']}→{planner['cubin_bytes']} | "
                    f"{spill_bytes(vectorized)}→{spill_bytes(planner)} |"
                )
        lines.append("")
    if "planner_nvcc_extra_vectorization" in payload["modes"]:
        lines.extend(
            [
                "## NVCC extra device vectorization scan",
                "",
                "| Workload | Arch | instructions planner→candidate | CUBIN bytes planner→candidate | spill bytes planner→candidate | SASS changed |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {
                (case["arch"], case["mode"]): case for case in workload["cases"]
            }
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                candidate = by_key[(arch, "planner_nvcc_extra_vectorization")]

                def spill_bytes(case: dict[str, Any]) -> int:
                    return sum(
                        int(item["spill_store_bytes"])
                        + int(item["spill_load_bytes"])
                        for item in case["resources"].values()
                    )

                lines.append(
                    f"| `{workload['name']}` | `{arch}` | "
                    f"{planner['instruction_count']}→{candidate['instruction_count']} | "
                    f"{planner['cubin_bytes']}→{candidate['cubin_bytes']} | "
                    f"{spill_bytes(planner)}→{spill_bytes(candidate)} | "
                    f"{planner['sass_sha256'] != candidate['sass_sha256']} |"
                )
        lines.append("")
    if "planner_sync_predicated_copy" in payload["modes"]:
        lines.extend(
            [
                "## Predicated global-to-shared cp.async restoration",
                "",
                "| Workload | Arch | source cp.async sync→auto | SASS LDGSTS sync→auto | LDG sync→auto | STS sync→auto | instructions sync→auto |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            if "predicated_async_copy_comparisons" not in workload:
                continue
            by_key = {
                (case["arch"], case["mode"]): case for case in workload["cases"]
            }
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                synchronous = by_key[(arch, "planner_sync_predicated_copy")]
                lines.append(
                    f"| `{workload['name']}` | `{arch}` | "
                    f"{synchronous['cp_async_source_occurrences']}→"
                    f"{planner['cp_async_source_occurrences']} | "
                    f"{synchronous['groups']['async_copy']}→"
                    f"{planner['groups']['async_copy']} | "
                    f"{synchronous['groups']['global_load']}→"
                    f"{planner['groups']['global_load']} | "
                    f"{synchronous['groups']['shared_store']}→"
                    f"{planner['groups']['shared_store']} | "
                    f"{synchronous['instruction_count']}→"
                    f"{planner['instruction_count']} |"
                )
        lines.append("")
    if "planner_int4x2_unpack_legacy" in payload["modes"]:
        lines.extend(
            [
                "## Packed signed-int4x2 unpack",
                "",
                "| Workload | Arch | helper legacy→peephole | instructions legacy→peephole | registers legacy→peephole |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for workload in payload["workloads"]:
            by_key = {(case["arch"], case["mode"]): case for case in workload["cases"]}
            for arch in payload["architectures"]:
                planner = by_key[(arch, "planner")]
                legacy_unpack = by_key[(arch, "planner_int4x2_unpack_legacy")]

                def register_count(case: dict[str, Any]) -> str:
                    values = sorted(
                        {int(item["n_regs"]) for item in case["resources"].values()}
                    )
                    return "/".join(map(str, values)) if values else "n/a"

                lines.append(
                    f"| `{workload['name']}` | `{arch}` | "
                    f"{legacy_unpack['int4x2_unpack_occurrences']}→"
                    f"{planner['int4x2_unpack_occurrences']} | "
                    f"{legacy_unpack['instruction_count']}→"
                    f"{planner['instruction_count']} | "
                    f"{register_count(legacy_unpack)}→{register_count(planner)} |"
                )
        lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def archive_raw() -> None:
    RAW_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(RAW_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(RAW_DIR).as_posix())


def main() -> int:
    started = time.perf_counter()
    repo_root = Path.cwd().resolve()
    RAW_DIR.mkdir(parents=True, exist_ok=False)
    predicated_async_negative_controls = (
        validate_predicated_async_negative_controls()
        if "planner_sync_predicated_copy" in MODES
        else []
    )
    workloads = build_workloads(repo_root)
    if WORKLOAD_NAMES:
        by_name = {workload["name"]: workload for workload in workloads}
        missing = [name for name in WORKLOAD_NAMES if name not in by_name]
        if missing:
            raise ValueError(f"unknown TILELANG_REAL_VECTOR_WORKLOADS: {', '.join(missing)}")
        workloads = [by_name[name] for name in WORKLOAD_NAMES]
    workload_records, compile_inputs = lower_sources(workloads)
    compile_all(compile_inputs)
    aggregate = enrich_comparisons(workload_records)
    enrich_launch_bounds_scan(workload_records, aggregate)
    enrich_auto_launch_bounds(workload_records, aggregate)
    payload = {
        "schema": "tilelang-real-vectorization-workloads-v8",
        "repository": REPOSITORY,
        "source_sha": SOURCE_SHA,
        "python": platform.python_version(),
        "architectures": list(ARCHES),
        "modes": list(MODES),
        "selected_workloads": [workload["name"] for workload in workloads],
        "config_key": CONFIG_KEY,
        "device_compile": True,
        "gpu_execution": False,
        "compile_workers": MAX_WORKERS,
        "predicated_async_negative_controls": predicated_async_negative_controls,
        "aggregate": aggregate,
        "workloads": workload_records,
        "duration_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    archive_raw()
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    print(f"result={RESULT_PATH}")
    print(f"report={REPORT_PATH}")
    print(f"raw_archive={RAW_ARCHIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
