"""Collect, calibrate and evaluate finite layout graphs on upstream workloads.

The source kernels and reference contracts come from layout_maxsat_real.py.
Calibration runs isolated microkernels and freezes its table before the separate
benchmark phase measures the original complete kernels. Every failed stage is
retained. Each case runs in a fresh process to contain CUDA/runtime failures.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time
import traceback

from layout_maxsat_real import cases, prepare, UPSTREAM


STRATEGIES = ("root", "maxsat", "maxsat-full", "treewidth", "local", "greedy")
TVM_REVISION = "907a88c8791ccf33b9874821bc875e7abf624367"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_case(case, args):
    import torch
    import tilelang
    from cuda.bindings import driver
    from tilelang.layout import GraphLayoutSession, LatencyTable, calibrate, calibration_environment

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260906)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(case=case, phase=args.phase, upstream=UPSTREAM, tilelang=tilelang.__version__,
                  compiler_revision=args.compiler_revision, torch=torch.__version__, results=[])

    def save():
        report["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write(output / f"{args.phase}.json", report)

    try:
        impl, schedule, tensors, out_ids, references, source_sha = prepare(case, Path(args.examples_root))
        report.update(schedule=schedule, source_sha256=source_sha)
    except Exception:
        report["preparation_error"] = traceback.format_exc()
        save()
        return
    configs = {str(k): v for k, v in dict(impl.pass_configs or {}).items()}
    flags = list(impl.compile_flags or [])
    env = calibration_environment(tilelang_revision=args.compiler_revision, tvm_revision=TVM_REVISION,
                                  compiler_flags=flags)
    report["environment"] = env
    table_path = output / "calibration/latency-table.json"
    if args.phase in ("collect", "calibrate"):
        start = time.monotonic()
        with GraphLayoutSession(output_directory=output / "collect", collect=True,
                                max_candidates=args.max_candidates) as collection:
            try:
                tilelang.compile(impl.get_tir(**schedule), out_idx=impl.out_idx, target="cuda",
                                 execution_backend="nvrtc", compile_flags=flags,
                                 pass_configs={**configs, "tl.layout_solver": "maxsat-full"})
            except Exception:
                report["collection_error"] = traceback.format_exc()
        report.update(collection_seconds=time.monotonic() - start,
                      measurement_count=len(collection.measurements),
                      regions=[dict(candidates=r["candidates"], candidate_attempts=r["candidate_attempts"],
                                    pinned_buffers=r["pinned_buffers"], rejections=len(r["rejections"]),
                                    operators=len(r["problem"]["operators"]), tensors=len(r["problem"]["tensors"]))
                               for r in collection.reports])
        save()
        if args.phase == "calibrate" and "collection_error" not in report:
            start = time.monotonic()
            try:
                table = calibrate(collection, output / "calibration", env)
                report.update(table_sha256=table.sha256, calibration_seconds=time.monotonic() - start,
                              frozen_unix_time=time.time())
            except Exception:
                report["calibration_error"] = traceback.format_exc()
            save()
        return

    def check():
        errors = []
        for index, reference in zip(out_ids, references):
            value = tensors[index]
            tolerance = 1e-2 if value.dtype == torch.float16 else 3e-4
            torch.testing.assert_close(value, reference, rtol=tolerance, atol=tolerance)
            errors.append(float((value.float() - reference.float()).abs().max()))
        return errors

    graphs = []
    for strategy in STRATEGIES:
        record = dict(strategy=strategy)
        report["results"].append(record)
        print("COMPILE", case["id"], strategy, flush=True)
        try:
            session = None
            if strategy in ("root", "maxsat"):
                context = nullcontext()
            else:
                table = LatencyTable.read(table_path, env)
                session = GraphLayoutSession(output_directory=output / strategy, table=table,
                                             max_candidates=args.max_candidates)
                context = session
            start = time.monotonic()
            with context:
                kernel = tilelang.compile(impl.get_tir(**schedule), out_idx=impl.out_idx, target="cuda",
                                         execution_backend="nvrtc", compile_flags=flags,
                                         pass_configs={**configs, "tl.layout_solver": strategy})
            record["compile_seconds"] = time.monotonic() - start
            if session is not None:
                record.update(
                    table_sha256=table.sha256,
                    unweighted_region_objective_ns=sum(r["result"]["cost"] for r in session.reports),
                    unweighted_root_objective_ns=sum(r["root"]["cost"] for r in session.reports),
                    conversions=sum(len(r["result"]["conversions"]) for r in session.reports),
                    region_statuses=[r["result"]["status"] for r in session.reports],
                    extraction_ms=sum(r["extraction_ms"] for r in session.reports),
                    solver_results=[r["result"] for r in session.reports],
                )
            folder = output / strategy
            folder.mkdir(exist_ok=True)
            source = kernel.get_kernel_source()
            binary = Path(kernel.adapter.lib_generator.libpath).read_bytes()
            (folder / "kernel.cu").write_text(source)
            (folder / "kernel.cubin").write_bytes(binary)
            record.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                          binary_sha256=hashlib.sha256(binary).hexdigest())
            status, device = driver.cuDeviceGet(torch.cuda.current_device())
            assert status == driver.CUresult.CUDA_SUCCESS
            record["kernel_resources"] = {}
            for symbol, handle in kernel.adapter.kernels.items():
                resource_values = {}
                for label, attribute in (("registers", "CU_FUNC_ATTRIBUTE_NUM_REGS"),
                                         ("local_bytes", "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"),
                                         ("shared_bytes", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES")):
                    status, value = driver.cuKernelGetAttribute(
                        getattr(driver.CUfunction_attribute, attribute), handle, device)
                    assert status == driver.CUresult.CUDA_SUCCESS
                    resource_values[label] = value
                record["kernel_resources"][symbol] = resource_values
            launch = kernel.adapter._forward_from_prebuild_lib
            launch(*tensors, stream=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            record["max_abs_errors"] = check()
            record["direct_correct"] = True
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                for _ in range(20):
                    launch(*tensors, stream=torch.cuda.current_stream().cuda_stream)
            for index in out_ids:
                tensors[index].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check()
            status, _, count = driver.cuGraphGetNodes(driver.CUgraph(graph.raw_cuda_graph()))
            assert status == driver.CUresult.CUDA_SUCCESS and count == 20 * len(kernel.adapter.kernels)
            record.update(graph_correct=True, graph_nodes=count, latency_samples_us=[], status="passed")
            graphs.append((record, graph, kernel))
        except Exception:
            record.update(status="failed", error=traceback.format_exc())
        record["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
    for _, graph, _ in graphs:
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()
    rng = random.Random(20260906)
    for _ in range(30):
        rng.shuffle(graphs)
        for record, graph, _kernel in graphs:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            record["latency_samples_us"].append(start.elapsed_time(end) * 1000 / 20)
    for record, _, _ in graphs:
        record["median_us"] = statistics.median(record["latency_samples_us"])
        record["min_us"] = min(record["latency_samples_us"])
        record["max_us"] = max(record["latency_samples_us"])
    save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compiler-revision", required=True)
    parser.add_argument("--phase", choices=["collect", "calibrate", "benchmark"], required=True)
    parser.add_argument("--case")
    parser.add_argument("--family", action="append")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-candidates", type=int, default=4096)
    args = parser.parse_args()
    if args.case:
        run_case(next(c for c in cases() if c["id"] == args.case), args)
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in cases():
        if args.family and case["family"] not in args.family:
            continue
        folder = root / case["id"]
        folder.mkdir(exist_ok=True)
        item = dict(case=case)
        print(args.phase.upper(), case["id"], flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--examples-root", args.examples_root,
                   "--output", str(folder), "--compiler-revision", args.compiler_revision,
                   "--phase", args.phase, "--case", case["id"], "--max-candidates", str(args.max_candidates)]
        with (folder / f"{args.phase}.log").open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                                        env={**os.environ, "TILELANG_DISABLE_CACHE": "1"})
                item["returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                item["timeout_seconds"] = args.timeout
        if (folder / f"{args.phase}.json").exists():
            item["report"] = json.loads((folder / f"{args.phase}.json").read_text())
        else:
            item["error"] = (folder / f"{args.phase}.log").read_text()[-12000:]
        summary.append(item)
        write(root / f"{args.phase}-summary.json", summary)


if __name__ == "__main__":
    main()
