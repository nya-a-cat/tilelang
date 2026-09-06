"""Native adapter semantics with explicitly synthetic optimization costs.

These fixtures exercise graph construction and rewriting. Their costs are
invented to force conversions and provide no performance evidence.
"""

import pytest
import torch
import tilelang
from tilelang import language as T
from tilelang.layout import GraphLayoutSession
from tilelang.layout._latency_table import ENVIRONMENT_FIELDS, LatencyTable


def kernel_function(partial):
    @T.prim_func
    def main(A: T.Tensor((8, 16), "float32"), B: T.Tensor((8, 16), "float32")):
        with T.Kernel(1, threads=32):
            x = T.alloc_fragment((8, 16), "float32")
            y = T.alloc_fragment((8, 16), "float32")
            T.copy(A, x)
            for i, j in T.Parallel(8, 16):
                y[i, j] = x[i, j] * 2
            if partial:
                for i, j in T.Parallel(4, 16):
                    y[i, j] = x[i, j] + 3
            else:
                for i, j in T.Parallel(8, 16):
                    y[i, j] = y[i, j] + x[i, j] + x[i, j]
            T.copy(y, B)
    return main


def synthetic_table(collection):
    roots = {op["configs"][0]["cost_key"] for report in collection.reports
             for op in report["problem"]["operators"] if "cost_key" in op["configs"][0]}
    environment = {key: "synthetic-test-fixture" for key in ENVIRONMENT_FIELDS}
    entries = []
    for identity, measurement in collection.measurements.items():
        cost = 1000 if identity in roots else 1
        entries.append(dict(key=measurement["key"], samples_ns=[cost] * 3,
                            cost_ns=cost, executable_sha256="0" * 64))
    return LatencyTable(dict(schema=1, frozen=True, environment=environment, entries=entries), environment)


def launch_and_check(kernel, partial):
    a = torch.arange(128, device="cuda", dtype=torch.float32).reshape(8, 16)
    b = torch.empty_like(a)
    reference = a * (2 if partial else 4)
    if partial:
        reference[:4] = a[:4] + 3
    launch = kernel.adapter._forward_from_prebuild_lib
    for _ in range(3):
        b.fill_(float("nan"))
        launch(a, b, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(b, reference, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(a, b, stream=torch.cuda.current_stream().cuda_stream)
    b.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(b, reference, rtol=0, atol=0)


@pytest.mark.parametrize("partial", [False, True])
def test_full_adapter_semantics_and_exact_solver_agreement(tmp_path, partial):
    torch.empty(1, device="cuda")  # Establish the primary context before NVRTC loading.
    main = kernel_function(partial)
    with GraphLayoutSession(output_directory=tmp_path / "collect", collect=True) as collection:
        root = tilelang.compile(main, target="cuda", execution_backend="nvrtc",
                                pass_configs={"tl.layout_solver": "maxsat-full"})
    launch_and_check(root, partial)
    assert collection.reports
    assert any(max(report["candidates"]) > 1 for report in collection.reports)
    assert all(report["mode"] == "calibration" for report in collection.reports)
    table = synthetic_table(collection)
    objectives = {}
    for algorithm in ("maxsat-full", "treewidth", "local", "greedy"):
        with GraphLayoutSession(output_directory=tmp_path / algorithm, table=table) as selected:
            kernel = tilelang.compile(main, target="cuda", execution_backend="nvrtc",
                                      pass_configs={"tl.layout_solver": algorithm})
        launch_and_check(kernel, partial)
        assert all(not report["missing_measurements"] for report in selected.reports)
        objectives[algorithm] = [report["result"]["cost"] for report in selected.reports]
        assert any(report["result"]["conversions"] for report in selected.reports)
    assert objectives["maxsat-full"] == objectives["treewidth"]


def test_missing_session_fails_explicitly():
    with pytest.raises(Exception, match="active GraphLayoutSession"):
        tilelang.compile(kernel_function(False), target="cuda", execution_backend="nvrtc",
                         pass_configs={"tl.layout_solver": "maxsat-full"})
