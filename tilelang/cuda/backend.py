from __future__ import annotations

import logging
import re

from tvm import tirx

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.contrib import nvcc
from tilelang.contrib.cuda_resource_info import (
    isolated_recorder,
    pop_last_recorded,
    pop_recorded,
    record_auto_launch_bounds_selection,
    record_usage,
    reset_recorder,
)
from tilelang.contrib.kernel_resource_info import KernelResourceUsage
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH, env
from tilelang.transform import PassConfigKey

from . import codegen, execution_backend, pipeline

_CUDA_GLOBAL_KERNEL_PATTERN = re.compile(r'(?:extern\s+"C"\s+)?__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?(\w+)')
_CUDA_DEFAULT_LAUNCH_BOUNDS_PATTERN = re.compile(
    r"(__launch_bounds__\(\s*\d+\s*,\s*)1(\s*\))"
)
_AUTO_LAUNCH_BOUNDS_MIN_REGISTERS = 224
# CUDA Programming Guide tables 30-31: resident threads and shared memory per
# SM. The selector only needs targets with an explicit architectural contract;
# unknown targets retain the established high-register fallback below.
_CUDA_OCCUPANCY_LIMITS = {
    75: (1024, 64 * 1024),
    80: (2048, 164 * 1024),
    86: (1536, 100 * 1024),
    87: (2048, 164 * 1024),
    89: (1536, 100 * 1024),
    90: (2048, 228 * 1024),
    100: (2048, 228 * 1024),
    120: (1536, 128 * 1024),
}
_CUDA_REGISTERS_PER_SM = 64 * 1024
_CUDA_SHARED_MEMORY_RESERVATION_PER_BLOCK = 1024

logger = logging.getLogger(__name__)


def _collect_external_cuda_kernel_names(source: str) -> list[str]:
    kernel_names: list[str] = []
    seen_names: set[str] = set()
    for match in _CUDA_GLOBAL_KERNEL_PATTERN.finditer(source):
        kernel_name = match.group(1)
        if kernel_name not in seen_names:
            kernel_names.append(kernel_name)
            seen_names.add(kernel_name)
    return kernel_names


def tilelang_callback_cuda_validate(device_mod):
    for _, base_func in device_mod.functions.items():
        if not isinstance(base_func, tirx.PrimFunc) or not base_func.attrs:
            continue

        code_block_source = base_func.attrs.get("code_block_source")
        if code_block_source is None:
            continue

        global_symbol = base_func.attrs.get("global_symbol")
        if global_symbol is None:
            raise ValueError("CodeGenTileLangCUDA expects source-kernel PrimFunc to have the global_symbol attribute")

        expected_name = str(global_symbol)
        code_block_entry_name = base_func.attrs.get("code_block_entry_name")
        if code_block_entry_name is not None and str(code_block_entry_name) != expected_name:
            raise ValueError("T.CUDASourceCodeKernel expects the lowered device global_symbol to match entry_name")

        kernel_names = _collect_external_cuda_kernel_names(str(code_block_source))
        if not kernel_names:
            raise ValueError("T.CUDASourceCodeKernel expects external CUDA source to declare at least one __global__ kernel")
        if expected_name not in kernel_names:
            raise ValueError(
                "T.CUDASourceCodeKernel expected device global_symbol "
                f"`{expected_name}` to match a __global__ kernel in the provided CUDA source. "
                f"Available entries: {', '.join(kernel_names)}"
            )


def _rewrite_default_launch_bounds(code: str, min_blocks_per_sm: int) -> tuple[str, int]:
    return _CUDA_DEFAULT_LAUNCH_BOUNDS_PATTERN.subn(
        rf"\g<1>{min_blocks_per_sm}\g<2>", code
    )


def _collect_cuda_launch_resources(device_mod) -> dict[str, dict[str, int]]:
    if device_mod is None:
        return {}

    launches: dict[str, dict[str, int]] = {}
    for global_var, base_func in device_mod.functions.items():
        attrs = base_func.attrs
        if not attrs:
            continue
        symbol = (
            str(attrs["global_symbol"])
            if "global_symbol" in attrs
            else global_var.name_hint
        )
        threads_per_block = 1
        if "thread_extent" not in attrs:
            continue
        try:
            for tag, extent in attrs["thread_extent"].items():
                if str(tag).startswith("threadIdx."):
                    threads_per_block *= int(extent)
            dynamic_shared_memory_bytes = int(
                attrs["dyn_shared_memory_buf"]
                if "dyn_shared_memory_buf" in attrs
                else 0
            )
        except (TypeError, ValueError):
            continue
        launches[symbol] = {
            "threads_per_block": threads_per_block,
            "dynamic_shared_memory_bytes": dynamic_shared_memory_bytes,
        }
    return launches


def _should_try_auto_launch_bounds(
    baseline: dict[str, KernelResourceUsage],
    target,
    launch_resources: dict[str, dict[str, int]],
) -> bool:
    if not baseline:
        return False
    if not launch_resources:
        return max(item.n_regs for item in baseline.values()) >= (
            _AUTO_LAUNCH_BOUNDS_MIN_REGISTERS
        )

    target_arch, _ = nvcc.get_target_arch_and_code(target)
    match = re.match(r"(?P<sm>\d+)", str(target_arch))
    limits = _CUDA_OCCUPANCY_LIMITS.get(int(match.group("sm"))) if match else None
    if limits is None:
        return max(item.n_regs for item in baseline.values()) >= (
            _AUTO_LAUNCH_BOUNDS_MIN_REGISTERS
        )

    max_threads_per_sm, shared_memory_per_sm = limits
    for name, usage in baseline.items():
        launch = launch_resources.get(name)
        if launch is None:
            continue
        threads_per_block = launch["threads_per_block"]
        block_shared_memory = (
            usage.shared_bytes
            + launch["dynamic_shared_memory_bytes"]
            + _CUDA_SHARED_MEMORY_RESERVATION_PER_BLOCK
        )
        two_blocks_fit_nonregister_resources = (
            2 * threads_per_block <= max_threads_per_sm
            and 2 * block_shared_memory <= shared_memory_per_sm
        )
        baseline_registers_limit_to_one_block = (
            usage.n_regs * threads_per_block > _CUDA_REGISTERS_PER_SM // 2
        )
        if two_blocks_fit_nonregister_resources and (
            usage.n_regs >= _AUTO_LAUNCH_BOUNDS_MIN_REGISTERS
            or baseline_registers_limit_to_one_block
        ):
            return True
    return False


def _auto_launch_bounds_candidate_is_safe(
    baseline: dict[str, KernelResourceUsage],
    candidate: dict[str, KernelResourceUsage],
    baseline_binary: bytes | bytearray,
    candidate_binary: bytes | bytearray,
) -> bool:
    if not baseline or baseline.keys() != candidate.keys():
        return False
    if len(candidate_binary) > len(baseline_binary):
        return False

    reduced = False
    for name, baseline_item in baseline.items():
        candidate_item = candidate[name]
        if candidate_item.n_regs > baseline_item.n_regs:
            return False
        reduced |= candidate_item.n_regs < baseline_item.n_regs
        if candidate_item.spill_store_bytes or candidate_item.spill_load_bytes:
            return False
        if candidate_item.local_bytes > baseline_item.local_bytes:
            return False
        if candidate_item.stack_frame_bytes > baseline_item.stack_frame_bytes:
            return False
    return reduced


def _compile_with_auto_launch_bounds(code, target, pass_config, device_mod=None):
    base_config = dict(pass_config)
    base_config.pop(PassConfigKey.TL_ENABLE_AUTO_LAUNCH_BOUNDS, None)
    selected_binary: bytes | bytearray
    selected_usage: dict[str, KernelResourceUsage]
    selected_min_blocks_per_sm = 1

    with isolated_recorder():
        baseline_binary = tilelang_callback_cuda_compile(code, target, base_config)
        baseline_usage = pop_recorded()
        selected_binary = baseline_binary
        selected_usage = baseline_usage

        candidate_code, rewrite_count = _rewrite_default_launch_bounds(code, 2)
        launch_resources = _collect_cuda_launch_resources(device_mod)
        if rewrite_count and _should_try_auto_launch_bounds(
            baseline_usage, target, launch_resources
        ):
            reset_recorder()
            try:
                candidate_binary = tilelang_callback_cuda_compile(
                    candidate_code, target, base_config
                )
            except RuntimeError as error:
                diagnostics = [
                    line.strip()
                    for line in str(error).splitlines()
                    if "ptxas fatal" in line.lower()
                ]
                logger.info(
                    "CUDA auto launch-bound candidate rejected: %s",
                    diagnostics[-1] if diagnostics else type(error).__name__,
                )
            else:
                candidate_usage = pop_recorded()
                if _auto_launch_bounds_candidate_is_safe(
                    baseline_usage,
                    candidate_usage,
                    baseline_binary,
                    candidate_binary,
                ):
                    selected_binary = candidate_binary
                    selected_usage = candidate_usage
                    selected_min_blocks_per_sm = 2
                    logger.info(
                        "CUDA auto launch bounds selected min_blocks_per_sm=2 "
                        "for %d kernel(s)",
                        rewrite_count,
                    )

    record_usage(selected_usage)
    record_auto_launch_bounds_selection(selected_min_blocks_per_sm)
    return selected_binary


def tilelang_callback_cuda_compile(code, target, pass_config=None, device_mod=None):
    from tilelang.cache.cuda_binary_cache import CUDABinaryCache

    target_arch, target_code = nvcc.get_target_arch_and_code(target)
    target_code_list = nvcc.get_target_code_list(target_code)
    gencode_code = nvcc.format_target_code_for_gencode(target_code)
    if gencode_code is None:
        arch = [f"-arch=sm_{target_arch}"]
    else:
        arch = ["-gencode", f"arch=compute_{target_arch},code={gencode_code}"]
    compile_format = "fatbin" if len(target_code_list) > 1 else "cubin"

    cfg = pass_config or {}
    enable_auto_launch_bounds = bool(
        cfg.get(PassConfigKey.TL_ENABLE_AUTO_LAUNCH_BOUNDS, False)
    )
    if enable_auto_launch_bounds and compile_format == "cubin":
        return _compile_with_auto_launch_bounds(code, target, cfg, device_mod)
    enable_fast_math = bool(cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False))
    ptxas_usage_level = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
    if ptxas_usage_level is not None:
        ptxas_usage_level = int(ptxas_usage_level)

    options = [
        "-std=c++20",
        "-I" + TILELANG_TEMPLATE_PATH,
        "-I" + CUTLASS_INCLUDE_DIR,
        "--resource-usage",
    ]
    extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
    if extra_flags:
        import shlex

        if isinstance(extra_flags, str):
            tokens = shlex.split(extra_flags)
        else:
            tokens = []
            for flag in extra_flags:
                if isinstance(flag, str):
                    tokens.extend(shlex.split(flag))
                else:
                    tokens.append(str(flag))
        options += tokens

    verbose = env.get_default_verbose()
    if enable_fast_math:
        options.append("--use_fast_math")
    if ptxas_usage_level is not None:
        options.append(f"--ptxas-options=--register-usage-level={ptxas_usage_level}")
    if verbose:
        options.append("-w")

    cache_key = CUDABinaryCache.make_key(
        code=code,
        target_kind=target.kind.name,
        target_arch=target_arch,
        target_code=target_code_list,
        compile_format=compile_format,
        options=options,
    )
    cached_binary = CUDABinaryCache.load(cache_key, compile_format)
    if cached_binary is not None:
        record_usage(CUDABinaryCache.load_resource_usage(cache_key, compile_format))
        return bytearray(cached_binary)

    pop_last_recorded()
    binary = nvcc.compile_cuda(code, compile_format, arch, options=options, verbose=verbose)
    resource_usage = pop_last_recorded()
    CUDABinaryCache.save(cache_key, compile_format, binary)
    CUDABinaryCache.save_resource_usage(cache_key, compile_format, resource_usage)
    return binary


BACKEND = register_backend(
    BackendModule(
        name="cuda",
        target_kinds=("cuda",),
        supports_target=codegen.is_plain_cuda_target,
        pipelines={"cuda": pipeline.CUDA_PIPELINE},
        device_codegens={
            "cuda": DeviceCodegen(
                "cuda",
                build=codegen.build_cuda,
                build_without_compile=codegen.build_cuda_without_compile,
            )
        },
        execution_backends=execution_backend.CUDA_EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={
            "tilelang_callback_cuda_validate": tilelang_callback_cuda_validate,
            "tilelang_callback_cuda_compile": tilelang_callback_cuda_compile,
        },
    )
)
