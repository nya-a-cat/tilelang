from __future__ import annotations
from typing import Any, Generic, Literal, ParamSpec, TypeVar
from collections.abc import Callable

from tilelang.jit.adapter.utils import is_cutedsl_target, is_metal_target, is_cuda_target, is_hip_target
from tvm.tirx import PrimFunc

from tilelang import tvm
from tilelang import env
from tilelang.backend.module import BackendContext, create_backend_context
from tvm.target import Target
from tilelang.engine.param import CompiledArtifact, KernelParam
from tilelang.jit.adapter import (
    BaseKernelAdapter,
    CachedTextSource,
    CythonKernelAdapter,
    CuTeDSLKernelAdapter,
    TVMFFIKernelAdapter,
    MetalKernelAdapter,
)
from tilelang.profiler import Profiler, TensorSupplyType
from tilelang.contrib import nvcc as tl_nvcc
from tilelang.contrib.cuda_resource_info import pop_recorded as pop_cuda_resource_usage
from tilelang.contrib.cuda_resource_info import reset_recorder as reset_cuda_resource_usage
from tilelang.contrib.hip_resource_info import pop_recorded as pop_hip_resource_usage
from tilelang.contrib.hip_resource_info import reset_recorder as reset_hip_resource_usage
from tilelang.jit.abi import prepare_tvm_ffi_callee_allocated_outputs, prepare_tvm_ffi_caller_allocated_outputs
from tilelang.jit.diagnostics import jit_phase
from tilelang.transform import PassConfigKey
from tilelang.transform.pass_config import normalize_pass_configs
from tilelang.instrumentation import compile_pass_instrumentation, create_pass_instruments
from tilelang.tools.pass_timing import create_pass_timing_tool
import logging
import os
import threading

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")
TargetLike = str | dict[str, object] | Target


class JITKernel(Generic[_P, _T]):
    """
    A wrapper class for compiling and invoking TileLang (TVM TIR) functions as PyTorch-compatible functions.

    Attributes
    ----------
    artifact : CompiledArtifact
        The compiled artifact containing the runtime module and parameters.
    adapter : BaseKernelAdapter
        The adapter for the compiled function.
    torch_function : Callable
        The compiled function that can be invoked as a PyTorch-compatible function.
    """

    prim_func: PrimFunc = None
    artifact: CompiledArtifact = None
    adapter: BaseKernelAdapter = None
    execution_backend_spec = None
    torch_function: Callable = None

    # tuner result
    latency: float = None
    config: dict[str, Any] = None
    ref_latency: float = None

    def __init__(
        self,
        func: PrimFunc = None,
        out_idx: list[int] | int = None,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] = "tvm_ffi",
        target: TargetLike = "auto",
        target_host: TargetLike | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        from_database: bool = False,
        compile_flags: list[str] | None = None,
        backend_context: BackendContext | None = None,
    ):
        """
        Initializes a TorchFunction instance.

        Parameters
        ----------
        func : tvm.tirx.PrimFunc, optional
            The TileLang TIR function to compile and wrap.
        out_idx : Union[List[int], int], optional
            Index(es) of the output tensors to return (default: None).
        execution_backend : Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"], optional
            Execution backend to use for kernel execution.
        target : str, dict, or tvm.target.Target, optional
            Compilation target (default: "auto"). Use a dict for target attributes,
            for example {"kind": "cuda", "arch": "sm_90"}.
        target_host : str, dict, or tvm.target.Target, optional
            Target host for cross-compilation (default: None).
        verbose : bool, optional
            Whether to enable verbose output (default: False).
        pass_configs : dict, optional
            Additional keyword arguments to pass to the Compiler PassContext.
            Refer to `tilelang.PassConfigKey` for supported options.
        from_database : bool, optional
            Whether to create a TorchFunction from a database.
        backend_context : BackendContext, optional
            Pre-resolved context supplied by compiler infrastructure. Direct
            callers may omit it and let this public entry resolve one context.
        """
        self.prim_func = func
        self.verbose = verbose
        self._caller_allocated_kernel: JITKernel | None = None
        self._caller_allocated_kernel_lock = threading.Lock()
        self._last_bound_outputs: tuple[Any, ...] | None = None
        self._last_bound_output_call: Callable[..., Any] | None = None

        self.pass_configs = normalize_pass_configs(pass_configs)

        self.compile_flags = [compile_flags] if isinstance(compile_flags, str) else compile_flags

        if backend_context is None:
            backend_context = create_backend_context(target, target_host, execution_backend)
        self.backend_context = backend_context
        self.backend = backend_context.module
        self.target = backend_context.target
        self.target_host = backend_context.target_host
        self.execution_backend_spec = backend_context.execution_backend
        self.execution_backend = self.execution_backend_spec.name

        if self.execution_backend == "cython":
            from tilelang.contrib.cc import get_cplus_compiler

            assert get_cplus_compiler() is not None, "Cython backend requires a C++ compiler, please install or use other backends."

        if from_database:
            return

        # Print log on compilation starts
        # NOTE(Chenggang): printing could let the training/inference framework easier to know
        # whether the communication timeout is from compilation
        if env.is_print_on_compilation_enabled():
            # assert func must have "global_symbol"
            func_name = func.attrs.get("global_symbol")
            assert func_name is not None, "func must have global_symbol"
            logger.info(f"TileLang begins to compile kernel `{func_name}` with `{out_idx=}`")

        # Compile the TileLang function and create a kernel adapter for execution.
        adapter = self._compile_and_create_adapter(func, out_idx)

        if env.is_print_on_compilation_enabled():
            func_name = func.attrs.get("global_symbol")
            assert func_name is not None, "func must have global_symbol"
            logger.info(f"TileLang completes to compile kernel `{func_name}`")

        # The adapter's function is assigned as the callable function for this instance.
        self.adapter = adapter
        self.torch_function = adapter.func

    @classmethod
    def from_database(
        cls,
        func: PrimFunc,
        host_kernel_source: CachedTextSource,
        device_kernel_source: CachedTextSource,
        kernel_lib_path: str,
        params: list[KernelParam],
        target: TargetLike,
        target_host: TargetLike | None,
        out_idx: list[int] | int,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"],
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
        backend_context: BackendContext | None = None,
    ):
        """
        Alternative constructor to create a TorchFunction directly from a database.
        """
        instance = cls(
            func=func,
            out_idx=out_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            pass_configs=pass_configs,
            from_database=True,
            compile_flags=compile_flags,
            backend_context=backend_context,
        )

        instance.adapter = instance._create_adapter_from_database(
            func_or_mod=func,
            params=params,
            result_idx=out_idx,
            target=target,
            host_kernel_source=host_kernel_source,
            device_kernel_source=device_kernel_source,
            kernel_lib_path=kernel_lib_path,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )
        instance.torch_function = instance.adapter.func
        return instance

    def __call__(self, *args: _P.args, **kwds: _P.kwargs) -> _T:
        """
        Invokes the compiled function with the given arguments.

        Parameters
        ----------
        *args : Any
            Positional arguments for the function.
        **kwds : Any
            Keyword arguments for the function.

        Returns
        -------
        Any
            The result of the function execution.
        """
        return self.torch_function(*args, **kwds)

    def _get_caller_allocated_kernel(self) -> JITKernel:
        """Lazily compile the full-parameter ABI used by reusable outputs."""
        if not self.out_idx:
            raise ValueError("This kernel has no callee-allocated outputs to bind.")

        kernel = self._caller_allocated_kernel
        if kernel is not None:
            return kernel

        with self._caller_allocated_kernel_lock:
            kernel = self._caller_allocated_kernel
            if kernel is not None:
                return kernel
            if self.prim_func is None:
                raise RuntimeError("Cannot prepare reusable outputs without the source PrimFunc.")

            caller_allocated_func = prepare_tvm_ffi_caller_allocated_outputs(self.prim_func)
            # Import lazily because tilelang.cache imports JITKernel while setting
            # up its backend-specific cache dispatch table.
            from tilelang.cache import cached

            kernel = cached(
                func=caller_allocated_func,
                out_idx=None,
                target=self.target,
                target_host=self.target_host,
                execution_backend=self.execution_backend,
                verbose=self.verbose,
                pass_configs=self.pass_configs,
                compile_flags=self.compile_flags,
            )
            self._caller_allocated_kernel = kernel
            return kernel

    def _normalize_bound_outputs(self, out: Any) -> tuple[Any, ...]:
        result_count = len(self.out_idx)
        if result_count == 0:
            raise ValueError("This kernel has no callee-allocated outputs to bind.")

        if result_count == 1:
            if isinstance(out, (list, tuple)):
                if len(out) != 1:
                    raise ValueError(f"Kernel expected one output buffer, but {len(out)} are provided.")
                outputs = (out[0],)
            else:
                outputs = (out,)
        else:
            if not isinstance(out, (list, tuple)):
                raise TypeError(f"Kernel expected {result_count} output buffers as a list or tuple.")
            if len(out) != result_count:
                raise ValueError(f"Kernel expected {result_count} output buffers, but {len(out)} are provided.")
            outputs = tuple(out)

        if any(output is None for output in outputs):
            raise TypeError("Output buffers cannot be None.")
        return outputs

    def _make_bound_output_call(self, outputs: tuple[Any, ...]) -> Callable[..., Any]:
        """Create a low-overhead callable that writes into fixed output buffers."""
        caller_allocated_kernel = self._get_caller_allocated_kernel()
        caller_func = caller_allocated_kernel.torch_function
        caller_adapter = getattr(caller_allocated_kernel, "adapter", None)
        get_call_entry = getattr(caller_adapter, "get_caller_allocated_call_entry", None)
        if get_call_entry is not None:
            direct_caller_func = get_call_entry()
            if direct_caller_func is not None:
                caller_func = direct_caller_func
        result_idx = tuple(self.out_idx)
        total_params = len(self.params)
        expected_inputs = total_params - len(result_idx)
        output_result = outputs[0] if len(outputs) == 1 else list(outputs)

        # Most kernels place one result last.  Keep this path allocation-light;
        # it is the path that matters for repeated small eager kernels.
        if result_idx == tuple(range(expected_inputs, total_params)):

            def bound_output_call(*inputs: Any):
                if len(inputs) != expected_inputs:
                    raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} are provided.")
                caller_func(*inputs, *outputs)
                return output_result

            return bound_output_call

        output_by_position = dict(zip(result_idx, outputs, strict=True))
        input_slots = tuple(position for position in range(total_params) if position not in output_by_position)

        def bound_output_call(*inputs: Any):
            if len(inputs) != expected_inputs:
                raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} are provided.")
            full_args: list[Any] = [None] * total_params
            for position, output in output_by_position.items():
                full_args[position] = output
            for position, value in zip(input_slots, inputs, strict=True):
                full_args[position] = value
            caller_func(*full_args)
            return output_result

        return bound_output_call

    def bind_outputs(self, out: Any) -> Callable[..., _T]:
        """Bind caller-owned output buffers and return a reusable fast callable.

        The first binding lazily compiles a companion entry with the original
        full parameter list.  Later calls skip output allocation and return the
        supplied buffer, or a list of supplied buffers for a multi-output
        kernel.  A bound callable reuses the same storage on every invocation;
        callers coordinate overlapping launches and buffer lifetimes.

        Parameters
        ----------
        out : Any
            One output buffer, or a list/tuple matching ``out_idx``.

        Returns
        -------
        Callable
            A callable accepting the kernel's non-output arguments.
        """
        outputs = self._normalize_bound_outputs(out)
        cached_outputs = self._last_bound_outputs
        if (
            cached_outputs is not None
            and len(cached_outputs) == len(outputs)
            and all(cached is current for cached, current in zip(cached_outputs, outputs, strict=True))
        ):
            cached_call = self._last_bound_output_call
            if cached_call is not None:
                return cached_call

        bound_call = self._make_bound_output_call(outputs)
        self._last_bound_outputs = outputs
        self._last_bound_output_call = bound_call
        return bound_call

    def call_into(self, *inputs: Any, out: Any) -> _T:
        """Execute once with caller-owned output buffers.

        ``bind_outputs`` is the lower-overhead interface for repeated calls.
        This convenience method caches the most recently bound output identity.
        """
        return self.bind_outputs(out)(*inputs)

    def _compile_and_create_adapter(
        self,
        tilelang_func: PrimFunc,
        out_idx: list[int] | int | None,
    ) -> BaseKernelAdapter:
        """Compile one kernel and construct its adapter in one tool session."""
        if self.execution_backend == "tvm_ffi":
            # MakePackedAPI consumes this attribute to omit all result buffers
            # from main's packed arguments and replace them with one allocator
            # anchor.  Use a derived PrimFunc so manual out_idx does not become
            # a persistent frontend attribute on the user's function.
            tilelang_func, out_idx = prepare_tvm_ffi_callee_allocated_outputs(tilelang_func, out_idx)

        func_name = str(tilelang_func.attrs.get("global_symbol", "<unknown>"))
        timing_tool = create_pass_timing_tool(self.pass_configs)
        tools = [timing_tool] if timing_tool is not None else []
        with compile_pass_instrumentation(name=func_name, tools=tools):
            target = self.target
            target_host = self.target_host
            execution_backend = self.execution_backend
            pass_configs = dict(self.pass_configs) if self.pass_configs else {}

            compile_flags = self.compile_flags
            if compile_flags is not None:
                compile_flags_cfg = pass_configs.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS)
                pass_configs[PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = (
                    compile_flags_cfg + compile_flags if compile_flags_cfg is not None else compile_flags
                )

            capture_cuda_resource_usage = is_cuda_target(target)
            capture_hip_resource_usage = is_hip_target(target)
            if capture_cuda_resource_usage:
                reset_cuda_resource_usage()
            elif capture_hip_resource_usage:
                reset_hip_resource_usage()

            compile_metadata = {
                "kernel": func_name,
                "target": str(target),
                "target_host": str(target_host) if target_host is not None else None,
                "backend": execution_backend,
            }
            self.artifact = self._compile_artifact(tilelang_func, pass_configs, compile_metadata)
            adapter = self._create_adapter_from_artifact(
                tilelang_func,
                out_idx,
                self.artifact,
                pass_configs,
                compile_metadata,
            )

            if capture_cuda_resource_usage:
                self._resource_usage = pop_cuda_resource_usage()
            elif capture_hip_resource_usage:
                self._resource_usage = pop_hip_resource_usage()

            return adapter

    def _compile_artifact(
        self,
        tilelang_func: PrimFunc,
        pass_configs: dict[str, Any],
        compile_metadata: dict[str, Any],
    ) -> CompiledArtifact:
        """Lower one TileLang function into the host/device compilation artifact."""
        enable_host_codegen = self.execution_backend_spec.enable_host_codegen
        enable_device_compile = self.execution_backend_spec.enable_device_compile

        base_pass_instruments = []
        if pass_configs.get(PassConfigKey.TL_ENABLE_DUMP_IR):
            dump_ir_path = pass_configs.get(PassConfigKey.TL_DUMP_IR_DIR, "./dump_ir")
            base_pass_instruments.append(tvm.ir.instrument.DumpIR(dump_dir=dump_ir_path))

        pass_instrument_context = f"stage=jit-lower, kernel={compile_metadata['kernel']}, backend={compile_metadata['backend']}"
        pass_instruments = [
            *create_pass_instruments(context=pass_instrument_context),
            *base_pass_instruments,
        ]
        with (
            jit_phase("lower", verbose=self.verbose, **compile_metadata),
            tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments),
            self.target,
        ):
            from tilelang.engine.lower import lower_with_context

            artifact = lower_with_context(
                tilelang_func,
                self.backend_context,
                enable_host_codegen=enable_host_codegen,
                enable_device_compile=enable_device_compile,
            )

        return artifact

    def _create_adapter_from_artifact(
        self,
        tilelang_func: PrimFunc,
        out_idx: list[int] | int | None,
        artifact: CompiledArtifact,
        pass_configs: dict[str, Any],
        compile_metadata: dict[str, Any],
    ) -> BaseKernelAdapter:
        """Construct the selected execution adapter from one lowered artifact."""

        def create_adapter(adapter_cls: Callable[..., BaseKernelAdapter], **kwargs: Any) -> BaseKernelAdapter:
            with jit_phase("adapter", verbose=self.verbose, **compile_metadata):
                return adapter_cls(**kwargs)

        execution_backend = self.execution_backend
        target = self.target
        compile_flags = self.compile_flags

        if execution_backend == "tvm_ffi":
            # Use TVMFFIKernelAdapter for interoperability with PyTorch via DLPack.
            # But we need to ensure that the runtime is enabled and the runtime module is not None.
            assert artifact.rt_mod is not None, "tvm_ffi backend requires a runtime module."
            adapter = create_adapter(
                TVMFFIKernelAdapter,
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                rt_mod=artifact.rt_mod,
                device_kernel_source=artifact.kernel_source,
                verbose=self.verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "cython":
            adapter = create_adapter(
                CythonKernelAdapter,
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                device_kernel_source=artifact.kernel_source,
                verbose=self.verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "nvrtc":
            from tilelang.jit.adapter import NVRTCKernelAdapter

            adapter = create_adapter(
                NVRTCKernelAdapter,
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                device_kernel_source=artifact.kernel_source,
                verbose=self.verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "torch":
            assert is_metal_target(target)
            adapter = create_adapter(
                MetalKernelAdapter,
                params=artifact.params,
                result_idx=out_idx,
                # target=target,
                func_or_mod=tilelang_func,
                # host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                kernel_global_source=artifact.kernel_source,
                verbose=self.verbose,
                # pass_configs=pass_configs,
                # compile_flags=compile_flags,
            )
        elif execution_backend == "cutedsl":
            assert is_cutedsl_target(target)
            adapter = create_adapter(
                CuTeDSLKernelAdapter,
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                host_mod=artifact.host_mod,
                device_mod=artifact.device_mod,
                device_kernel_source=artifact.kernel_source,
                verbose=self.verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        else:
            # Handle invalid backend.
            raise ValueError(f"Invalid execution backend: {execution_backend}")

        return adapter

    def _create_adapter_from_database(
        self,
        params: list[KernelParam],
        result_idx: list[int] | int,
        target: TargetLike,
        func_or_mod: PrimFunc | tvm.runtime.Module,
        host_kernel_source: CachedTextSource,
        device_kernel_source: CachedTextSource,
        kernel_lib_path: str,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ) -> BaseKernelAdapter:
        target = self.target
        execution_backend = self.execution_backend

        # Create an adapter based on the specified execution backend.
        if execution_backend == "tvm_ffi":
            adapter = TVMFFIKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "cython":
            adapter = CythonKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                pass_configs=pass_configs,
            )
        elif execution_backend == "nvrtc":
            from tilelang.jit.adapter import NVRTCKernelAdapter

            adapter = NVRTCKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        elif execution_backend == "cutedsl":
            adapter = CuTeDSLKernelAdapter.from_database(
                params=params,
                result_idx=result_idx,
                target=target,
                func_or_mod=func_or_mod,
                host_kernel_source=host_kernel_source,
                device_kernel_source=device_kernel_source,
                kernel_lib_path=kernel_lib_path,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        else:
            # Handle invalid backend.
            raise ValueError(f"Invalid execution backend: {execution_backend}")

        return adapter

    @classmethod
    def from_tilelang_function(cls, tilelang_func: PrimFunc, **kwargs):
        """
        Alternative constructor to create a TorchFunction directly from a TileLang PrimFunc.

        Parameters
        ----------
        tilelang_func : tvm.tirx.PrimFunc
            The TileLang (TVM TIR) function to compile.
        **kwargs : dict
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        TorchFunction
            An instance of TorchFunction wrapping the compiled function.
        """
        return cls(func=tilelang_func, **kwargs)

    def get_profiler(self, tensor_supply_type: TensorSupplyType = TensorSupplyType.Auto) -> Profiler:
        """
        Creates a profiler to benchmark the compiled runtime module.

        Parameters
        ----------
        tensor_supply_type : TensorSupplyType, optional
            The type of input tensors to supply for profiling (default: TensorSupplyType.Auto).

        Returns
        -------
        Profiler
            A Profiler instance for benchmarking the runtime module.
        """
        return Profiler(self.params, self.out_idx, tensor_supply_type).with_default_adapter(self.adapter)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        """
        Returns the source code of the compiled kernel function.

        Returns
        -------
        str
            The source code of the compiled kernel function.
        """
        if self.execution_backend in {"cython", "nvrtc", "tvm_ffi", "cutedsl"}:
            return self.adapter.get_kernel_source(kernel_only=kernel_only)
        return self.artifact.kernel_source

    def get_host_source(self) -> str:
        """
        Returns the source code of the host function.
        """
        if self.execution_backend in {"cython", "nvrtc", "tvm_ffi", "cutedsl"}:
            return self.adapter.get_host_source()
        assert self.artifact.host_mod is not None, "host_mod is not available"
        return str(self.artifact.host_mod)

    def run_once(self, func: Callable | None = None) -> None:
        return self.get_profiler().run_once(func)

    def show_source(self, which: Literal["kernel", "host", "both"] = "kernel") -> None:
        """
        Print generated source code to stdout.

        Parameters
        ----------
        which : Literal["kernel", "host", "both"], optional
            Select which source to print. Defaults to "kernel".

        Examples
        --------
        >>> jit_kernel.show_source()            # print kernel source
        >>> jit_kernel.show_source("host")      # print host source
        >>> jit_kernel.show_source("both")      # print both sources
        """
        try:
            if which == "kernel":
                src = self.get_kernel_source()
                print(src)
            elif which == "host":
                src = self.get_host_source()
                # Host is generally C/C++
                print(src)
            elif which == "both":
                print("===== Kernel Source =====")
                ksrc = self.get_kernel_source()
                print(ksrc)
                print("===== Host Source =====")
                hsrc = self.get_host_source()
                print(hsrc)
            else:
                raise ValueError(f"Unknown option for 'which': {which}")
        except Exception as e:
            logger.error(f"Failed to show source code: {e}")

    def export_sources(self, kernel_path: str | None = None, host_path: str | None = None) -> None:
        """
        Export generated source code to files.

        Parameters
        ----------
        kernel_path : Optional[str]
            Destination file path to write the kernel source. If None, skips writing kernel code.
        host_path : Optional[str]
            Destination file path to write the host source. If None, skips writing host code.

        Examples
        --------
        >>> jit_kernel.export_sources(kernel_path="/tmp/kernel.cu")
        >>> jit_kernel.export_sources(host_path="/tmp/host.cc")
        >>> jit_kernel.export_sources(
        ...     kernel_path="/tmp/kernel.cu",
        ...     host_path="/tmp/host.cc",
        ... )
        """
        if kernel_path is None and host_path is None:
            raise ValueError("At least one of kernel_path or host_path must be provided.")
        try:
            if kernel_path is not None:
                dir_path = os.path.dirname(kernel_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(kernel_path, "w") as f:
                    f.write(self.get_kernel_source())
            if host_path is not None:
                dir_path = os.path.dirname(host_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(host_path, "w") as f:
                    f.write(self.get_host_source())
        except Exception as e:
            logger.error(f"Failed to export sources: {e}")

    # Backward compatibility alias (deprecated)
    def print_source_code(self, which: Literal["kernel", "host", "both"] = "kernel", file: str | None = None) -> None:
        """
        Deprecated: use show_source() or export_sources() instead.

        Parameters
        ----------
        which : Literal["kernel", "host", "both"], optional
            Kept for backward compatibility with printing behavior.
        file : Optional[str]
            If provided, behaves like export_sources(kernel_path=file).

        Examples
        --------
        >>> # New API (preferred)
        >>> jit_kernel.show_source("both")
        >>> jit_kernel.export_sources(kernel_path="/tmp/kernel.cu")

        >>> # Old API (still works but deprecated)
        >>> jit_kernel.print_source_code(file="/tmp/kernel.cu")
        """
        logger.warning("print_source_code is deprecated; use show_source() or export_sources() instead.")
        if file is not None:
            # Historical behavior wrote only kernel source when file provided
            self.export_sources(kernel_path=file)
        else:
            self.show_source(which=which)

    def update_tuner_result(self, latency: float, config: dict[str, Any], ref_latency: float) -> JITKernel:
        """
        Updates the tuning results for this kernel.

        Parameters
        ----------
        latency : float
            The measured latency of this kernel configuration.
        config : Dict[str, Any]
            The configuration parameters used for this kernel.
        ref_latency : float
            The reference latency to compare against.

        Returns
        -------
        None
        """
        self.latency = latency
        self.config = config
        self.ref_latency = ref_latency

        return self

    def get_tuner_result(self) -> dict[str, Any]:
        """
        Gets the tuning results for this kernel.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - latency: The measured latency of this kernel
            - config: The configuration parameters used
            - ref_latency: The reference latency for comparison
        """
        if self.latency is None:
            raise ValueError("Tuning results are not available. Please tune the kernel first.")

        return {
            "latency": self.latency,
            "config": self.config,
            "ref_latency": self.ref_latency,
        }

    @property
    def out_idx(self) -> list[int]:
        return self.adapter.result_idx

    @property
    def params(self) -> list[KernelParam]:
        return self.artifact.params if self.artifact else self.adapter.params

    @property
    def kernel_source(self) -> str:
        if self.artifact:
            return self.artifact.kernel_source
        source = getattr(self.adapter, "kernel_global_source", None)
        if source is not None:
            return source
        return self.adapter.get_kernel_source(kernel_only=True) or ""

    @property
    def host_source(self) -> str:
        if self.artifact:
            return str(self.artifact.host_mod)
        get_host_source = getattr(self.adapter, "get_host_source", None)
        if get_host_source is None:
            return ""
        return get_host_source() or ""

    @property
    def resource_usage(self) -> dict[str, Any]:
        """Per-entry compiler resource usage when supported by the target."""
        return getattr(self, "_resource_usage", {}) or {}

    def _primary_resource_usage(self):
        usage = self.resource_usage
        if not usage:
            return None
        gsym = None
        if self.prim_func is not None and self.prim_func.attrs is not None:
            gsym = self.prim_func.attrs.get("global_symbol")
        if gsym is not None and str(gsym) in usage:
            return usage[str(gsym)]
        return next(iter(usage.values()))

    @property
    def n_regs(self) -> int | None:
        info = self._primary_resource_usage()
        return info.n_regs if info is not None else None

    @property
    def n_spills(self) -> int | None:
        info = self._primary_resource_usage()
        return info.n_spills if info is not None else None

    @property
    def n_max_threads(self) -> int | None:
        info = self._primary_resource_usage()
        return info.n_max_threads if info is not None else None

    @property
    def shared_bytes(self) -> int | None:
        info = self._primary_resource_usage()
        return info.shared_bytes if info is not None else None

    @property
    def stack_frame_bytes(self) -> int | None:
        info = self._primary_resource_usage()
        return info.stack_frame_bytes if info is not None else None

    @property
    def spill_store_bytes(self) -> int | None:
        info = self._primary_resource_usage()
        return info.spill_store_bytes if info is not None else None

    @property
    def spill_load_bytes(self) -> int | None:
        info = self._primary_resource_usage()
        return info.spill_load_bytes if info is not None else None

    @property
    def constant_bytes(self) -> int | None:
        info = self._primary_resource_usage()
        return info.constant_bytes if info is not None else None

    @property
    def local_bytes(self) -> int | None:
        info = self._primary_resource_usage()
        return info.local_bytes if info is not None else None

    @property
    def barrier_count(self) -> int | None:
        info = self._primary_resource_usage()
        return info.barrier_count if info is not None else None

    def export_library(self, kernel_file: str) -> None:
        """
        Exports the compiled kernel function to a shared library file.

        Parameters
        ----------
        kernel_file : str
            The path to the shared library file to create.
        """
        # rt_module: tvm.runtime.Module = None
        # rt_params: dict = None
        # adapter: BaseKernelAdapter = None
        # torch_function: Callable = None
        # rt_module: use export_library to export
        # rt_params: use cloudpickle to serialize

        if self.artifact is None or self.artifact.rt_mod is None:
            raise AttributeError(
                'Runtime module is not available. Please compile the kernel with `execution_backend="tvm_ffi"` before exporting.'
            )

        dir_path = os.path.dirname(kernel_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        self.artifact.rt_mod.export_library(kernel_file)
        logger.info(f"Kernel library exported to {os.path.abspath(kernel_file)}")

    def _get_ptx(self, verbose: bool | None = None) -> str:
        """
        Compile and return PTX for the current kernel (CUDA only).

        Parameters
        ----------
        verbose : Optional[bool]
            Whether to enable verbose NVRTC logs. Defaults to self.verbose.

        Returns
        -------
        str
            The compiled PTX text.
        """
        if not is_cuda_target(self.target):
            raise ValueError("PTX is only available for CUDA targets.")
        # Prefer NVCC for PTX generation via contrib helper
        code = self.get_kernel_source()
        if verbose is None:
            verbose = self.verbose
        # Ensure target is set so nvcc picks correct arch via Target.current()
        with self.target:
            return tl_nvcc.get_ptx_from_source(code, compile_flags=self.compile_flags, verbose=verbose)

    def show_ptx(self) -> None:
        """
        Print compiled PTX for the kernel (CUDA only).

        Examples
        --------
        >>> jit_kernel.show_ptx()
        """
        try:
            ptx = self._get_ptx()
            print(ptx)
        except Exception as e:
            logger.error(f"Failed to show PTX: {e}")

    def export_ptx(self, path: str) -> None:
        """
        Export compiled PTX to a file (CUDA only).

        Parameters
        ----------
        path : str
            Destination file path to write PTX.

        Examples
        --------
        >>> jit_kernel.export_ptx("/tmp/kernel.ptx")
        """
        if not path:
            raise ValueError("path must be provided to export PTX")
        try:
            ptx = self._get_ptx()
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, "w") as f:
                f.write(ptx)
            logger.info(f"PTX saved to {os.path.abspath(path)}")
        except Exception as e:
            logger.error(f"Failed to export PTX: {e}")

    def _get_sass(self, verbose: bool | None = None) -> str:
        """
        Compile and return SASS for the current kernel (CUDA only).

        Parameters
        ----------
        verbose : Optional[bool]
            Whether to enable verbose tool logs. Defaults to self.verbose.

        Returns
        -------
        str
            The disassembled SASS text.
        """
        if not is_cuda_target(self.target):
            raise ValueError("SASS is only available for CUDA targets.")
        code = self.get_kernel_source()
        if verbose is None:
            verbose = self.verbose
        with self.target:
            return tl_nvcc.get_sass_from_source(code, compile_flags=self.compile_flags, verbose=verbose)

    def show_sass(self) -> None:
        """
        Print disassembled SASS for the kernel (CUDA only).

        Examples
        --------
        >>> jit_kernel.show_sass()
        """
        try:
            sass = self._get_sass()
            print(sass)
        except Exception as e:
            logger.error(f"Failed to show SASS: {e}")

    def export_sass(self, path: str) -> None:
        """
        Export disassembled SASS to a file (CUDA only).

        Parameters
        ----------
        path : str
            Destination file path to write SASS.

        Examples
        --------
        >>> jit_kernel.export_sass("/tmp/kernel.sass")
        """
        if not path:
            raise ValueError("path must be provided to export SASS")
        try:
            sass = self._get_sass()
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, "w") as f:
                f.write(sass)
            logger.info(f"SASS saved to {os.path.abspath(path)}")
        except Exception as e:
            logger.error(f"Failed to export SASS: {e}")
