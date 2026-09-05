"""Measure unchanged upstream operator implementations with both layout policies.

Each case runs in its own process. Explicitly supplied launch streams, poisoned
outputs and graph node counts validate the measured work. This is an operator
and shared-expert block suite, with fixed schedules and no autotuning.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

UPSTREAM = "1a1df62e94d9736a2c9f13849bc4c2cf9377275b"
SOURCES = {
    "add": "examples/elementwise/example_elementwise_add.py",
    "rmsnorm": "examples/norm/rms_norm.py",
    "rmsnorm_splitk": "examples/norm/rms_norm.py",
    "layernorm": "examples/norm/layernorm.py",
    "softmax": "examples/online_softmax/online_softmax.py",
    "gemm": "examples/gemm/example_gemm.py",
    "gemv": "examples/gemv/example_gemv.py",
    "gemv_reducer": "examples/gemv/example_gemv.py",
    "attention": "examples/flash_attention/example_mha_fwd_bshd.py",
    "shared_mlp": "examples/fusedmoe/example_fusedmoe_tilelang.py",
}


def cases():
    specs = []
    def add(family, shape, **extra):
        label = family + "-" + "x".join(map(str, shape))
        if extra.get("causal") is not None:
            label += "-causal" if extra["causal"] else "-noncausal"
        specs.append(dict(id=label, family=family, shape=shape, **extra))
    for shape, dtype in [((128, 4096), "float16"), ((1024, 4096), "float32"), ((4096, 4096), "float16")]:
        add("add", shape, dtype=dtype)
    for shape in [(1, 4096), (128, 4096), (1024, 4096), (128, 8192)]:
        add("rmsnorm", shape)
    for shape in [(128, 8192), (1024, 8192)]:
        add("rmsnorm_splitk", shape)
    for shape in [(128, 768), (1024, 1024), (128, 4096)]:
        add("layernorm", shape)
    for shape in [(128, 4096), (1024, 8192), (128, 32768)]:
        add("softmax", shape)
    for shape in [(1024, 1024, 1024), (128, 4096, 4096), (1024, 2048, 2048)]:
        add("gemm", shape)
    for family in ["gemv", "gemv_reducer"]:
        for shape in [(4096, 4096), (11008, 4096)]:
            add(family, shape)
    for shape, causal in [((1, 8, 512, 64), False), ((1, 8, 2048, 64), True),
                          ((2, 8, 1024, 64), True), ((1, 8, 512, 128), True)]:
        add("attention", shape, causal=causal)
    for shape in [(128, 1024, 2816), (128, 4096, 11008)]:
        add("shared_mlp", shape)
    return specs


def load_example(root, family, shape):
    path = root / SOURCES[family]
    # Import full example modules with their original decorators and schedules.
    # The softmax file runs an 8192x8192 demo at module scope: omit only those
    # demo statements while preserving the function AST and its source file.
    sys.path.insert(0, str(path.parent))
    name = "upstream_" + family
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if family == "softmax":
        # This upstream demo declares M/N at module scope rather than T.const.
        # Supply the selected concrete dimensions before the JIT decorator runs.
        module.__dict__.update(M=shape[0], N=shape[1])
        tree = ast.parse(path.read_text(), filename=str(path))
        tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        exec(compile(tree, str(path), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(case, root):
    import torch
    import torch.nn.functional as F
    module, source_hash = load_example(root, case["family"], case["shape"])
    family, shape = case["family"], case["shape"]
    rand = lambda dims, dtype=torch.float16: torch.randn(dims, device="cuda", dtype=dtype)
    manual_outputs = None
    if family == "add":
        m, n = shape
        dtype = getattr(torch, case["dtype"])
        impl = module.elementwise_add
        config = dict(M=m, N=n, block_M=32, block_N=32, threads=128,
                      in_dtype=case["dtype"], out_dtype=case["dtype"])
        a, b = rand(shape, dtype), rand(shape, dtype)
        inputs, refs = [a, b], [a + b]
    elif family in ("rmsnorm", "rmsnorm_splitk"):
        m, n = shape
        impl = module.rms_norm if family == "rmsnorm" else module.rms_norm_splitk
        config = dict(M=m, N=n, blk_m=1)
        if family == "rmsnorm_splitk":
            config["blk_k"] = 1024
        a = rand(shape, torch.float32)
        inputs, refs = [a], [module.ref_program(a)]
    elif family == "layernorm":
        m, n = shape
        impl = module._layernorm_fwd
        config = dict(N=m, D=n, eps=1e-5, blk_m=1, threads=256, in_dtype="float16", out_dtype="float16")
        x, g, b = rand(shape), rand((n,)), rand((n,))
        xf = x.float()
        mean = xf.mean(-1)
        rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5)
        inputs = [x, g, b]
        refs = [F.layer_norm(x, (n,), g, b, eps=1e-5), mean, rstd]
    elif family == "softmax":
        m, n = shape
        impl = module.softmax_kernel
        config = dict(M=m, N=n, BLOCK_M=1, BLOCK_N=min(n, 8192), dtype="float16")
        x = rand(shape)
        inputs, refs = [x], [x.float().softmax(-1).half()]
    elif family == "gemm":
        m, n, k = shape
        impl = module.matmul
        config = dict(M=m, N=n, K=k, block_M=128, block_N=128, block_K=32)
        a, b = rand((m, k)), rand((k, n))
        inputs, refs = [a, b], [(a.float() @ b.float()).half()]
    elif family in ("gemv", "gemv_reducer"):
        n, k = shape
        a, b = rand((k,)), rand((n, k))
        if family == "gemv":
            impl = module.splitk_gemv_vectorized_tvm
            config = dict(N=n, K=k, BLOCK_N=2, reduce_threads=32)
            inputs = [a, b]
        else:
            impl = module.gemv_alloc_reducer.jit_impl
            config = dict(M=n, N=k, block_M=128, block_N=128, num_stages=2, threads=256)
            inputs = [b, a]
        refs = [(b.float() @ a.float()).half()]
    elif family == "attention":
        batch, heads, seq, dim = shape
        impl = module.flashattn.jit_impl
        config = dict(batch=batch, heads=heads, seq_len=seq, dim=dim, is_causal=case["causal"],
                      block_M=64, block_N=64, num_stages=1, threads=128)
        tensors = [rand((batch, seq, heads, dim)) for _ in range(3)]
        q, k, v = [x.transpose(1, 2).float() for x in tensors]
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=case["causal"])
        inputs, refs = tensors, [ref.transpose(1, 2).contiguous().half()]
    else:
        tokens, hidden, expert = shape
        impl = module.moe_forward_tilelang_shared
        config = dict(num_tokens=tokens, dhidden=hidden, dexpert=expert, dtype="float16",
                      block_token=128, block_dhidden=128, block_dexpert=128, threads=256, num_stages=1)
        x = rand((tokens, hidden))
        wg, wu = [rand((expert, hidden)) / hidden ** 0.5 for _ in range(2)]
        wd = rand((hidden, expert)) / expert ** 0.5
        mid = (F.silu(x.float() @ wg.float().T) * (x.float() @ wu.float().T)).half()
        final = (mid.float() @ wd.float().T).half()
        inputs, refs = [x, wg, wu, wd], [mid, final]
        manual_outputs = [4, 5]
    prim = impl.get_tir(**config)
    if manual_outputs is not None:
        outputs = manual_outputs
    elif prim.attrs and "tilelang_out_idx" in prim.attrs:
        outputs = [int(i) for i in prim.attrs["tilelang_out_idx"]]
    else:
        outputs = list(impl.out_idx)
    outputs = [i % len(prim.params) for i in outputs]
    all_args, input_iter = [], iter(inputs)
    for i, param in enumerate(prim.params):
        buf = prim.buffer_map[param]
        if i in outputs:
            all_args.append(torch.empty(tuple(int(x) for x in buf.shape), device="cuda", dtype=getattr(torch, str(buf.dtype))))
        else:
            all_args.append(next(input_iter))
    return impl, config, all_args, outputs, refs, source_hash


def run_case(case, args):
    import torch
    import tilelang
    from cuda.bindings import driver
    torch.manual_seed(20260906)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(case=case, upstream=UPSTREAM, variant=args.variant, tilelang=tilelang.__version__,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  capability=torch.cuda.get_device_capability(), tilelang_disable_cache=os.environ.get("TILELANG_DISABLE_CACHE"),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=[])
    impl, schedule, tensors, out_ids, references, source_hash = prepare(case, Path(args.examples_root))
    report.update(schedule=schedule, source_path=SOURCES[case["family"]], source_sha256=source_hash,
                  input_shapes=[list(t.shape) for t in tensors], output_indices=out_ids)
    modes = [("register-count", "root"), ("io-aware", "root")]
    if args.variant == "candidate":
        modes += [("register-count", "maxsat"), ("io-aware", "maxsat")]
    kernels = []
    repeats = 20
    def check_outputs():
        errors = []
        for index, ref in zip(out_ids, references):
            actual = tensors[index]
            tol = 3e-4 if ref.dtype == torch.float32 else 1e-2
            torch.testing.assert_close(actual, ref, atol=tol, rtol=tol)
            errors.append(float((actual.float() - ref.float()).abs().max()))
        return errors
    for model, solver in modes:
        label = f"{model}-{solver}"
        record = dict(mode=label, cost_model=model, solver=solver, samples_us=[])
        report["results"].append(record)
        print(f"COMPILE {case['id']} {label}", flush=True)
        try:
            config = dict(impl.pass_configs or {})
            config["tl.layout_cost_model"] = model
            if args.variant == "candidate":
                config.update({"tl.layout_solver": solver, "tl.layout_solver_verbose": True,
                               "tl.layout_solver_timeout_ms": 100})
            started = time.perf_counter()
            prim = impl.get_tir(**schedule)
            compiled = tilelang.compile(prim, out_idx=impl.out_idx, execution_backend="nvrtc", target="cuda", pass_configs=config)
            record["compile_seconds"] = time.perf_counter() - started
            source = compiled.get_kernel_source()
            (output / (label + ".cu")).write_text(source)
            record["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            binary = Path(compiled.adapter.lib_generator.libpath).read_bytes()
            (output / (label + ".cubin")).write_bytes(binary)
            record["binary_sha256"] = hashlib.sha256(binary).hexdigest()
            record["resources"] = {}
            status, device = driver.cuDeviceGet(torch.cuda.current_device())
            assert status == driver.CUresult.CUDA_SUCCESS
            for symbol, handle in compiled.adapter.kernels.items():
                entry = {}
                for key, attr in [("registers", "CU_FUNC_ATTRIBUTE_NUM_REGS"),
                                  ("local_bytes", "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"),
                                  ("shared_bytes", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES")]:
                    status, value = driver.cuKernelGetAttribute(getattr(driver.CUfunction_attribute, attr), handle, device)
                    assert status == driver.CUresult.CUDA_SUCCESS
                    entry[key] = value
                record["resources"][symbol] = entry
            # Invoke the original generated launcher with preallocated outputs.
            # The kernel and schedule remain unchanged; allocation is outside timing.
            launch = compiled.adapter._forward_from_prebuild_lib
            launch(*tensors, stream=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            record["max_abs_errors"] = check_outputs()
            record["direct_correct"] = True
            for _ in range(5):
                launch(*tensors, stream=torch.cuda.current_stream().cuda_stream)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    launch(*tensors, stream=torch.cuda.current_stream().cuda_stream)
            for index in out_ids:
                tensors[index].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check_outputs()
            status, _, count = driver.cuGraphGetNodes(driver.CUgraph(graph.raw_cuda_graph()))
            assert status == driver.CUresult.CUDA_SUCCESS
            expected = repeats * len(compiled.adapter.kernels)
            assert count == expected, f"Captured {count} nodes; expected {expected}"
            record.update(graph_correct=True, graph_nodes=count, invocations_per_graph=repeats,
                          kernels_per_invocation=len(compiled.adapter.kernels))
            kernels.append((record, graph, compiled))
        except Exception:
            record["error"] = traceback.format_exc()
            print(record["error"], flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2))
    rng = random.Random(20260906)
    # Warm all graphs before interleaving so compilation idle time is excluded.
    for _, graph, _ in kernels:
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()
    for _ in range(30):
        rng.shuffle(kernels)
        for record, graph, _compiled in kernels:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            record["samples_us"].append(start.elapsed_time(end) * 1000 / repeats)
    for record, _, _ in kernels:
        record.update(median_us=statistics.median(record["samples_us"]),
                      min_us=min(record["samples_us"]), max_us=max(record["samples_us"]))
        print("RESULT " + json.dumps(record), flush=True)
    (output / "results.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--examples-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case")
    parser.add_argument("--family", action="append")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    selected = [c for c in cases() if not args.family or c["family"] in args.family]
    if args.case:
        run_case(next(c for c in selected if c["id"] == args.case), args)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in selected:
        folder = output / case["id"]
        folder.mkdir(exist_ok=True)
        print("CASE " + case["id"], flush=True)
        item = dict(case=case)
        try:
            with (folder / "run.log").open("w") as log:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--variant", args.variant,
                                          "--examples-root", args.examples_root, "--output", str(folder),
                                          "--case", case["id"]], stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            item["returncode"] = process.returncode
        except subprocess.TimeoutExpired:
            item["timeout_seconds"] = args.timeout
        if (folder / "results.json").exists():
            item["report"] = json.loads((folder / "results.json").read_text())
        else:
            item["error"] = (folder / "run.log").read_text()[-12000:]
        summary.append(item)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("CASE_DONE " + case["id"], flush=True)


if __name__ == "__main__":
    main()
