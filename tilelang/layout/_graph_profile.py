"""Offline, executable-backed costs for the experimental finite layout graph.

The initial protocol measures a single CTA containing one native operation and
its input/output materialization. These inclusive costs are an explicit proxy:
they include launch and bridge overhead and do not predict pipeline overlap or
whole-kernel register pressure. Every entry retains its generated CUDA, CUBIN,
raw samples and protocol identity. Selection consumes a separately frozen table.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from ._latency_table import LatencyTable, digest


PROTOCOL = {
    "name": "isolated-single-cta-inclusive-v1",
    "launches_per_graph": 32,
    "warmup_replays": 5,
    "samples": 15,
    "includes": ["kernel launch", "input materialization", "native operation", "output materialization"],
    "free_loop_indices": 0,
    "floating_inputs": "seeded uniform [0.001, 0.126) before dtype rounding",
}


def environment(*, tilelang_revision, tvm_revision, compiler_flags=(), protocol=None):
    """Identify the actual runtime; revisions must come from the build manifest."""
    import torch
    from cuda.bindings import nvrtc

    if not tilelang_revision or not tvm_revision:
        raise ValueError("exact compiler and TVM build revisions are required")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    if len(driver) != 1:
        raise ValueError("the initial calibration protocol requires one visible GPU")
    result, major, minor = nvrtc.nvrtcVersion()
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"cannot identify the CUDA compiler: {result}")
    return dict(
        gpu_name=torch.cuda.get_device_name(),
        compute_capability=".".join(map(str, torch.cuda.get_device_capability())),
        driver_version=driver[0], cuda_version=f"NVRTC {major}.{minor}; Torch runtime {torch.version.cuda}",
        tilelang_revision=tilelang_revision, tvm_revision=tvm_revision,
        compiler_flags=list(compiler_flags), timing_protocol=digest(protocol or PROTOCOL),
    )


def make_microkernel(measurement):
    """Build one annotated operation, keeping its physical fragment layouts.

    Scalar indices belonging to the enclosing iteration are fixed at zero.
    Reducer epochs and asynchronous lifetimes require an atomic context and are
    rejected here with a saved diagnostic. No substitute cost is returned.
    """
    import tvm_ffi
    from tilelang import tvm, language as T

    tir = tvm.tirx
    if measurement["kind"] == "conversion":
        original = measurement["buffer"]
        source = tir.decl_buffer(original.shape, original.dtype, name="source", scope="local.fragment")
        target = tir.decl_buffer(original.shape, original.dtype, name="destination", scope="local.fragment")
        statement = tvm_ffi.get_global_func("tl.layout.make_conversion")(source, target)
        reads, writes = [source], [target]
        layouts = {source: measurement["source"], target: measurement["target"]}
    else:
        statement = measurement["statement"]
        reads, writes = list(measurement["reads"]), list(measurement["writes"])
        layouts = dict(measurement["layouts"])
    buffers = list(dict.fromkeys([*reads, *writes]))
    if not writes:
        raise ValueError("an isolated operation needs an observable output")
    if any(buffer.scope() in ("local.reducer", "shared.tmem", "shared.barrier") for buffer in buffers):
        raise ValueError("isolated profiling requires the complete reducer/asynchronous epoch")
    if len({buffer.data for buffer in buffers}) != len(buffers):
        raise ValueError("isolated profiling requires one view per storage family")
    if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
        if getattr(statement.value.op, "name", "") in ("tl.tileop.wgmma_gemm", "tl.tileop.tcgen05_gemm"):
            raise ValueError("isolated profiling requires an explicit asynchronous wait context")
    for buffer in buffers:
        stride = 1
        compact = True
        if buffer.strides:
            for extent, actual in zip(reversed(buffer.shape), reversed(buffer.strides)):
                compact &= isinstance(actual, tir.IntImm) and int(actual) == stride
                stride *= int(extent)
        if not compact or not isinstance(buffer.elem_offset, tir.IntImm) or int(buffer.elem_offset):
            raise ValueError("isolated profiling currently requires compact zero-offset buffers")
        if any(not isinstance(size, tir.IntImm) or int(size) < 1 for size in buffer.shape):
            raise ValueError("isolated profiling requires static operand shapes")

    thread = measurement["thread_index"]
    if not isinstance(thread, tir.Var):
        raise ValueError("profiling requires one explicit threadIdx.x axis")
    bound = measurement["thread_bounds"]
    if int(bound.min) != 0:
        raise ValueError("isolated profiling requires a zero-based thread scope")

    def materialization(buffer, name, load):
        # Ordinary local arrays contain a separate value in every thread. Give
        # that implicit axis explicit global storage when profiling them.
        if buffer.scope() == "local":
            global_buffer = tir.decl_buffer([bound.extent, *buffer.shape], buffer.dtype, name=name)
            indices = [tir.Var(f"profile_index_{i}", "int32") for i in range(len(buffer.shape))]
            if load:
                body = tir.BufferStore(buffer, tir.BufferLoad(global_buffer, [thread, *indices]), indices)
            else:
                body = tir.BufferStore(global_buffer, tir.BufferLoad(buffer, indices), [thread, *indices])
            for index, extent in reversed(list(zip(indices, buffer.shape))):
                body = tir.For(index, 0, extent, tir.ForKind.SERIAL, body)
            return global_buffer, body
        global_buffer = tir.decl_buffer(buffer.shape, buffer.dtype, name=name)
        return global_buffer, tir.Evaluate(T.copy(global_buffer, buffer) if load else T.copy(buffer, global_buffer))

    globals_ = [b for b in buffers if b.scope() == "global"]
    allocations = [b for b in buffers if b.scope() != "global"]
    params, before, after = list(globals_), [], []
    outputs = list(b for b in dict.fromkeys(writes) if b.scope() == "global")
    for buffer in dict.fromkeys(reads):
        if buffer.scope() != "global":
            materialized, copy = materialization(buffer, f"input_{buffer.name}", True)
            params.append(materialized)
            before.append(copy)
    for buffer in dict.fromkeys(writes):
        if buffer.scope() != "global":
            materialized, copy = materialization(buffer, f"output_{buffer.name}", False)
            params.append(materialized)
            outputs.append(materialized)
            after.append(copy)
    defined = [b.data for b in buffers] + [thread]
    free = tir.analysis.undefined_vars(statement, defined)
    substitutions = {}
    for variable in free:
        if variable.dtype == "handle":
            raise ValueError(f"unresolved pointer in isolated operation: {variable.name}")
        substitutions[variable] = tir.const(0, variable.dtype)
    statement = tir.stmt_functor.substitute(statement, substitutions)
    body = tir.stmt_seq(*before, statement, *after)
    annotations = {"layout_map": {b: l for b, l in layouts.items() if b in buffers}}
    block = tir.SBlock([], [], [], "root", body, alloc_buffers=allocations, annotations=annotations)
    body = tir.SBlockRealize([], True, block)
    # The ordinary kernel frame supplies all three axes. ThreadSync's shared
    # dependency analysis addresses its final three thread entries as x/y/z.
    axes = [(thread, bound.extent, "threadIdx.x"),
            (tir.Var("profile_thread_y", "int32"), 1, "threadIdx.y"),
            (tir.Var("profile_thread_z", "int32"), 1, "threadIdx.z")]
    for variable, extent, tag in reversed(axes):
        axis = tir.IterVar(tvm.ir.Range(0, extent), variable, tir.IterVar.ThreadIndex, tag)
        body = tir.AttrStmt(axis, "thread_extent", extent, body)
    function = tir.PrimFunc(params, body).with_attr("global_symbol", "layout_microkernel").with_attr("tir.noalias", True)
    function = function.with_attr("target", measurement["function"].attrs["target"])
    return function, params, outputs


def compile_microkernel(measurement, compile_flags=()):
    """Use the ordinary CUDA lowering suffix and original NVRTC adapter."""
    from tilelang.backend import PassPipeline, create_backend_context
    from tilelang.cuda.pipeline import CUDAPassPipelineAfterLayout, CUDAPassPipelineFinalize
    from tilelang.jit.kernel import JITKernel

    function, params, outputs = make_microkernel(measurement)
    context = create_backend_context(function.attrs["target"], None, "nvrtc")

    def lower(mod, target):
        return CUDAPassPipelineFinalize(CUDAPassPipelineAfterLayout(mod, target), target)

    module = replace(context.module, pipelines={"cuda": PassPipeline("cuda", lower)})
    context = replace(context, module=module)
    configs = dict(measurement.get("pass_configs", {}))
    configs["tl.layout_solver"] = "root"
    kernel = JITKernel(function, execution_backend="nvrtc", backend_context=context,
                       pass_configs=configs, compile_flags=list(compile_flags))
    return kernel, params, outputs


def measure_one(measurement, directory, protocol=None, compile_flags=()):
    """Measure a compiled CUDA graph and require it to overwrite poisoned outputs."""
    import torch
    from cuda.bindings import driver

    protocol = dict(protocol or PROTOCOL)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The NVRTC adapter requires an established primary CUDA context.
    torch.empty(1, device="cuda")
    started = time.monotonic()
    kernel, params, outputs = compile_microkernel(measurement, compile_flags)
    compile_seconds = time.monotonic() - started
    source = kernel.get_kernel_source()
    (directory / "kernel.cu").write_text(source, encoding="utf-8")
    library = Path(kernel.adapter.lib_generator.libpath)
    binary = library.read_bytes()
    (directory / "kernel.cubin").write_bytes(binary)
    generator = torch.Generator(device="cuda").manual_seed(int(digest(measurement["key"])[:16], 16))
    inputs = []
    output_ids = {b for b in outputs}
    read_buffers = set(measurement.get("reads", []))
    for buffer in params:
        dtype = getattr(torch, str(buffer.dtype))
        shape = tuple(int(size) for size in buffer.shape)
        if dtype.is_floating_point:
            value = torch.rand(shape, dtype=dtype, device="cuda", generator=generator) * 0.125 + 0.001
        else:
            value = torch.randint(1, 4, shape, dtype=dtype, device="cuda", generator=generator)
        if buffer in output_ids and buffer not in read_buffers and dtype.is_floating_point:
            value.fill_(float("nan"))
        inputs.append(value)
    launch = kernel.adapter._forward_from_prebuild_lib
    launch(*inputs, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    expected = {buffer: value.clone() for buffer, value in zip(params, inputs) if buffer in output_ids}
    if measurement["kind"] == "conversion":
        torch.testing.assert_close(inputs[-1], inputs[0], rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    launches = protocol["launches_per_graph"]
    with torch.cuda.graph(graph):
        for _ in range(launches):
            launch(*inputs, stream=torch.cuda.current_stream().cuda_stream)
    status, _, node_count = driver.cuGraphGetNodes(driver.CUgraph(graph.raw_cuda_graph()))
    if status != driver.CUresult.CUDA_SUCCESS or node_count != launches * len(kernel.adapter.kernels):
        raise RuntimeError(f"unexpected CUDA graph node count: {status}, {node_count}")
    # An output that is also an input may intentionally accumulate on replay;
    # poison/replay validation is available for write-only materialized outputs.
    safe_outputs = [(value, expected[buffer]) for buffer, value in zip(params, inputs)
                    if buffer in output_ids and buffer not in read_buffers and value.dtype.is_floating_point]
    for value, _ in safe_outputs:
        value.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for value, expected_value in safe_outputs:
        # Global regions can be partial. Only observed finite locations enter
        # this replay check; independent source correctness tests remain required.
        finite = torch.isfinite(expected_value)
        if not finite.any():
            raise RuntimeError("isolated operation produced no finite observable output")
        torch.testing.assert_close(value[finite], expected_value[finite], rtol=0, atol=0)
    for _ in range(protocol["warmup_replays"]):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(protocol["samples"]):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e6 / launches)
    entry = dict(key=measurement["key"], samples_ns=samples,
                 cost_ns=max(1, math.ceil(statistics.median(samples))),
                 executable_sha256=hashlib.sha256(binary).hexdigest(),
                 source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                 compile_seconds=compile_seconds, graph_nodes=node_count, protocol=protocol)
    (directory / "measurement.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return entry


def _read_verified_measurement(directory, expected_environment, identity, protocol):
    """Read one measurement only after validating its complete local evidence."""
    directory = Path(directory)
    entry = json.loads((directory / "measurement.json").read_text(encoding="utf-8"))
    LatencyTable(dict(schema=1, frozen=True, environment=expected_environment,
                      entries=[entry]), expected_environment)
    if digest(entry["key"]) != identity or entry.get("protocol") != protocol:
        raise ValueError("saved measurement identity or protocol differs")
    for filename, field in (("kernel.cubin", "executable_sha256"), ("kernel.cu", "source_sha256")):
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != entry.get(field):
            raise ValueError("saved measurement executable/source digest differs")
    return entry


def calibrate(collection, directory, expected_environment, protocol=None, *, resume=True, cache_directory=None):
    """Measure all requested keys, save failures, and freeze only full coverage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table_path = directory / "latency-table.json"
    table_tmp = directory / ".latency-table.json.tmp"
    protocol = dict(protocol or PROTOCOL)
    if expected_environment["timing_protocol"] != digest(protocol):
        raise ValueError("timing protocol differs from the calibration environment")
    actual = environment(tilelang_revision=expected_environment["tilelang_revision"],
                         tvm_revision=expected_environment["tvm_revision"],
                         compiler_flags=expected_environment["compiler_flags"], protocol=protocol)
    if actual != expected_environment:
        raise ValueError("calibration runtime differs from the requested environment")
    manifest = directory / "environment.json"
    if manifest.exists() and json.loads(manifest.read_text()) != expected_environment:
        raise ValueError("output directory contains measurements from another environment")
    # A request that passed environment identity validation owns the output
    # directory. Remove any prior frozen table before collecting fresh
    # evidence, so an incomplete request cannot retain a successful result.
    table_path.unlink(missing_ok=True)
    table_tmp.unlink(missing_ok=True)
    manifest.write_text(json.dumps(expected_environment, indent=2) + "\n", encoding="utf-8")
    # Identify missing profiling contexts before spending GPU time on thousands
    # of conversion pairs. Every requested key stays in the coverage accounting.
    preflight_failures = []
    for measurement in collection.measurements.values():
        try:
            make_microkernel(measurement)
        except Exception as exc:
            preflight_failures.append(dict(key=measurement["key"], stage="preflight", error=repr(exc)))
    if preflight_failures:
        (directory / "progress.json").write_text(json.dumps(dict(
            requested=len(collection.measurements), entries=[], failures=preflight_failures,
            cache_rejections=[],
        ), indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"calibration preflight failed for {len(preflight_failures)} requested measurements; see progress.json")
    # The environment and complete key jointly identify reusable measurements.
    # Copy verified files into each case so its evidence remains self-contained.
    cache = None if cache_directory is None else Path(cache_directory) / digest(expected_environment)
    entries, failures, cache_rejections = [], [], []
    for identity, measurement in collection.measurements.items():
        try:
            saved = directory / identity / "measurement.json"
            cached = None if cache is None else cache / identity / "measurement.json"
            entry = None
            # Validate all cache evidence before copying any of it into the
            # calibration directory. A damaged cache is recorded and replaced
            # by a fresh measurement without promoting unverified files.
            if resume and not saved.exists() and cached is not None and cached.parent.exists():
                if not cached.exists():
                    cache_rejections.append(dict(
                        key=measurement["key"], error="cached measurement.json is missing"))
                else:
                    try:
                        _read_verified_measurement(cached.parent, expected_environment, identity, protocol)
                    except Exception as exc:
                        cache_rejections.append(dict(key=measurement["key"], error=repr(exc)))
                    else:
                        saved.parent.mkdir(parents=True, exist_ok=True)
                        for filename in ("measurement.json", "kernel.cu", "kernel.cubin"):
                            shutil.copyfile(cached.parent / filename, saved.parent / filename)
                        entry = _read_verified_measurement(saved.parent, expected_environment, identity, protocol)
            if resume and saved.exists():
                if entry is None:
                    entry = _read_verified_measurement(saved.parent, expected_environment, identity, protocol)
            if entry is None:
                entry = measure_one(measurement, directory / identity, protocol,
                                    expected_environment["compiler_flags"])
                entry = _read_verified_measurement(saved.parent, expected_environment, identity, protocol)
            entries.append(entry)
            if cache is not None:
                destination = cache / identity
                destination.mkdir(parents=True, exist_ok=True)
                for filename in ("kernel.cu", "kernel.cubin", "measurement.json"):
                    shutil.copyfile(saved.parent / filename, destination / filename)
                _read_verified_measurement(destination, expected_environment, identity, protocol)
        except Exception as exc:
            failures.append(dict(key=measurement["key"], error=repr(exc)))
        (directory / "progress.json").write_text(json.dumps(dict(
            requested=len(collection.measurements), entries=entries, failures=failures,
            cache_rejections=cache_rejections,
        ), indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"calibration failed for {len(failures)} of {len(collection.measurements)} measurements; see progress.json")
    if not entries:
        raise ValueError("calibration collection contains no measurements")
    table = LatencyTable(dict(schema=1, frozen=True, environment=expected_environment, entries=entries), expected_environment)
    try:
        table.write(table_tmp)
        table_tmp.replace(table_path)
    finally:
        table_tmp.unlink(missing_ok=True)
    return table
