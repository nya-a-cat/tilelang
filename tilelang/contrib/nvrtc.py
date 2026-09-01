from __future__ import annotations

import re
import cuda.bindings.nvrtc as nvrtc
from typing import Literal
from tvm.target import Target
from .nvcc import get_target_compute_version, parse_compute_version


def get_nvrtc_version() -> tuple[int, int]:
    result, major, minor = nvrtc.nvrtcVersion()
    assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get NVRTC version: {result}"
    return (major, minor)


def get_nvrtc_arch_option(
    target_format: Literal["ptx", "cubin"],
    arch: int | str | None = None,
) -> str:
    """Return NVRTC's exact architecture option without inventing feature suffixes.

    CUDA feature-specific targets such as ``100f`` and ``100a`` are distinct
    compilation contracts.  Keep that suffix when the caller or current TVM
    target supplied one; a bare integer remains a base architecture.
    """
    if target_format not in ["cubin", "ptx"]:
        raise ValueError("target_format must be cubin or ptx")

    if arch is None:
        target = Target.current(allow_none=True)
        if target is not None and "arch" in target.attrs:
            arch = str(target.attrs["arch"])
        else:
            major, minor = parse_compute_version(get_target_compute_version(target))
            arch = major * 10 + minor

    arch_token = str(arch)
    for prefix in ("sm_", "compute_"):
        if arch_token.startswith(prefix):
            arch_token = arch_token.removeprefix(prefix)
            break
    if re.fullmatch(r"[0-9]{2,3}[af]?", arch_token) is None:
        raise ValueError("NVRTC arch must be an exact CUDA architecture token such as 80, 'sm_100', 'sm_100f', or 'sm_100a'.")

    prefix = "compute" if target_format == "ptx" else "sm"
    return f"--gpu-architecture={prefix}_{arch_token}"


def compile_cuda(
    code: str,
    target_format: Literal["ptx", "cubin"] = "ptx",
    arch: int | str | None = None,
    options: str | list[str] | None = None,
    verbose: bool = False,
) -> bytearray:
    """Compile cuda code with NVRTC.

    Parameters
    ----------
    code : str
        The cuda code.

    target_format : Literal["ptx", "cubin"]
        The target format of nvrtc compiler.

    arch : Optional[Union[int, str]]
        The exact CUDA architecture code. Feature suffixes are preserved, for
        example ``"100f"`` or ``"sm_100a"``.

    options : Optional[Union[str, List[str]]]
        The additional options.

    verbose : bool
        Whether to print the verbose output.

    Return
    ------
    result_bytes : bytearray
        The bytearray of the cubin or ptx code.
    """
    arch_option = get_nvrtc_arch_option(target_format, arch)

    file_name = "tvm_kernels"
    final_options = ["-default-device"]
    if get_nvrtc_version() >= (12, 8):
        final_options += ["-pch"]
    final_options += [arch_option]

    if options:
        if isinstance(options, str):
            final_options += [options]
        elif isinstance(options, list):
            final_options += options
        else:
            raise ValueError("options must be str or list of str")

    code = "#include <tl_templates/cuda/nvrtc_std.h>\n" + code

    if "cudaGridDependencySynchronize" in code or "cudaTriggerProgrammaticLaunchCompletion" in code:
        code = '#include "cuda_device_runtime_api.h"\n' + code

    code_bytes = bytes(code, "utf-8")
    result, program = nvrtc.nvrtcCreateProgram(code_bytes, bytes(file_name, "utf-8"), 0, [], [])
    assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to create program: {result}"

    options_bytes = [bytes(flag, "utf-8") for flag in final_options]
    compile_result = nvrtc.nvrtcCompileProgram(program, len(options_bytes), options_bytes)[0]

    if compile_result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        msg = f"{code}\nCompilation error:\n"
        if verbose:
            result, log_size = nvrtc.nvrtcGetProgramLogSize(program)
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get program log size: {result}"
            log_bytes = bytes(log_size)
            result = nvrtc.nvrtcGetProgramLog(program, log_bytes)[0]
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get program log: {result}"
            msg += f"{log_bytes.decode('utf-8')}\n"
        else:
            msg += "Turn on verbose to see the full compilation log."
        msg += f"Options: {' '.join(final_options)}\n"
        raise RuntimeError(msg)

    if target_format == "cubin":
        result, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get CUBIN size: {result}"
        result_bytes = bytes(cubin_size)
        result = nvrtc.nvrtcGetCUBIN(program, result_bytes)[0]
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get CUBIN: {result}"
    else:
        result, ptx_size = nvrtc.nvrtcGetPTXSize(program)
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get PTX size: {result}"
        result_bytes = bytes(ptx_size)
        result = nvrtc.nvrtcGetPTX(program, result_bytes)[0]
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get PTX: {result}"

    # Destroy handler
    assert nvrtc.nvrtcDestroyProgram(program)[0] == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to destroy program: {result}"

    return result_bytes
