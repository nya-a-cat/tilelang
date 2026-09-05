"""Controlled GPU comparison of upstream/root and bounded layout composition.

Run with the baseline or candidate wheel in a fresh process. Save every sample,
source, correctness verdict and compiler log; failures remain in the JSON report.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
import tilelang
import tilelang.language as T


def broadcast(blocks, m, n, threads, kind):
    @T.prim_func
    def kernel(S: T.Tensor((blocks, m), T.float32), A: T.Tensor((blocks, m, n), T.float32),
               Out: T.Tensor((blocks, m, n), T.float32)):
        with T.Kernel(blocks, threads=threads) as b:
            small = T.alloc_fragment((m,), T.float32)
            full = T.alloc_fragment((m, n), T.float32)
            for i in T.Parallel(m):
                small[i] = S[b, i]
            for i, j in T.Parallel(m, n):
                if kind == "broadcast":
                    full[i, j] = small[i] * 2.0
                else:
                    full[i, j] = A[b, i, j] + small[i]
            for i, j in T.Parallel(m, n):
                Out[b, i, j] = full[i, j]
    return kernel


def branches(blocks, m, n, threads, kind):
    @T.prim_func
    def kernel(S: T.Tensor((blocks, m + n), T.float32),
               Out: T.Tensor((blocks, m, n), T.float32),
               Other: T.Tensor((blocks, n, m), T.float32)):
        with T.Kernel(blocks, threads=threads) as b:
            left = T.alloc_fragment((m,), T.float32)
            right = T.alloc_fragment((n,), T.float32)
            for i in T.Parallel(m):
                left[i] = S[b, i]
            for j in T.Parallel(n):
                right[j] = S[b, m + j]
            for i, j in T.Parallel(m, n):
                Out[b, i, j] = left[i] * 2.0
            for j, i in T.Parallel(n, m):
                Other[b, j, i] = right[j] * 3.0
    return kernel


def transpose(blocks, m, n, threads, kind):
    @T.prim_func
    def kernel(A: T.Tensor((blocks, m, n), T.float32), Out: T.Tensor((blocks, n, m), T.float32)):
        with T.Kernel(blocks, threads=threads) as b:
            fragment = T.alloc_fragment((m, n), T.float32)
            for i, j in T.Parallel(m, n):
                fragment[i, j] = A[b, i, j]
            for j, i in T.Parallel(n, m):
                Out[b, j, i] = fragment[i, j] + 1.0
    return kernel


def reduction(blocks, m, n, threads, kind):
    @T.prim_func
    def kernel(A: T.Tensor((blocks, m, n), T.float32), Out: T.Tensor((blocks, m), T.float32)):
        with T.Kernel(blocks, threads=threads) as b:
            fragment = T.alloc_fragment((m, n), T.float32)
            sums = T.alloc_fragment((m,), T.float32)
            T.copy(A[b, 0:m, 0:n], fragment)
            T.reduce_sum(fragment, sums, dim=1)
            T.copy(sums, Out[b, 0:m])
    return kernel


def inputs(blocks, m, n, kind):
    x = lambda *shape: torch.randn(shape, device="cuda", dtype=torch.float32)
    if kind in ("broadcast", "fused"):
        s, a = x(blocks, m), x(blocks, m, n)
        ref = (s[:, :, None] * 2).expand_as(a) if kind == "broadcast" else a + s[:, :, None]
        return [s, a, torch.empty_like(a)], [ref], [2]
    if kind == "branches":
        s = x(blocks, m + n)
        a, b = x(blocks, m, n), x(blocks, n, m)
        return [s, a, b], [(s[:, :m, None] * 2).expand_as(a), (s[:, m:, None] * 3).expand_as(b)], [1, 2]
    a = x(blocks, m, n)
    ref = a.transpose(1, 2).contiguous() + 1 if kind == "transpose" else a.sum(2)
    return [a, torch.empty_like(ref)], [ref], [1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260906)
    rng = random.Random(20260906)
    report = {"variant": args.variant, "tilelang": tilelang.__version__, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "results": []}
    factories = {"broadcast": broadcast, "fused": broadcast, "branches": branches,
                 "transpose": transpose, "reduction": reduction}
    cases = [(kind, m, n, threads) for kind in factories for m, n, threads in
             [(2, 2560, 256), (8, 512, 128), (32, 128, 128), (64, 64, 256)]]
    modes = [("register-count", "root"), ("io-aware", "root")]
    if args.variant == "candidate":
        modes += [("register-count", "maxsat"), ("io-aware", "maxsat")]
    for kind, m, n, threads in cases:
        tensors, references, out_ids = inputs(128, m, n, kind)
        kernels = []
        for model, solver in modes:
            name = f"{kind}-{m}-{n}-{threads}-{model}-{solver}"
            record = {"case": name, "kind": kind, "m": m, "n": n, "threads": threads,
                      "cost_model": model, "solver": solver, "samples_us": []}
            report["results"].append(record)
            try:
                config = {"tl.layout_cost_model": model}
                if args.variant == "candidate":
                    config.update({"tl.layout_solver": solver, "tl.layout_solver_verbose": True,
                                   "tl.layout_solver_timeout_ms": 1000})
                print("COMPILE " + name, flush=True)
                started = time.perf_counter()
                compiled = tilelang.compile(factories[kind](128, m, n, threads, kind),
                                            target="cuda", execution_backend="nvrtc", pass_configs=config)
                record["compile_seconds"] = time.perf_counter() - started
                source = compiled.get_kernel_source()
                (output / (name + ".cu")).write_text(source)
                record["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
                compiled(*tensors)
                torch.cuda.synchronize()
                for i, ref in zip(out_ids, references):
                    torch.testing.assert_close(tensors[i], ref, atol=2e-4, rtol=2e-4)
                record["correct"] = True
                for _ in range(5):
                    compiled(*tensors)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(100):
                        compiled(*tensors)
                kernels.append((record, graph))
            except Exception as error:
                record["error"] = str(error)
                print("FAIL " + name + " " + str(error), flush=True)
        # Interleave modes within each case; graph replay suppresses Python gaps.
        for _ in range(15):
            rng.shuffle(kernels)
            for record, graph in kernels:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                record["samples_us"].append(start.elapsed_time(end) * 10)
        for record, _ in kernels:
            record["median_us"] = statistics.median(record["samples_us"])
            record["min_us"] = min(record["samples_us"])
            record["max_us"] = max(record["samples_us"])
            print(json.dumps(record), flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2))
    print("REPORT " + str(output / "results.json"), flush=True)


if __name__ == "__main__":
    main()
