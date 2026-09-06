"""Exercise real calibration, frozen selection and GPU execution together."""

import json
import pytest
import torch
import tilelang
from tilelang import language as T
from tilelang.layout import GraphLayoutSession
from tilelang.layout._graph_profile import calibrate, environment
from tilelang.layout._latency_table import LatencyTable


def test_nvrtc_single_output_parameter():
    @T.prim_func
    def fill(B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = 3.0

    output = torch.full((128,), float("nan"), device="cuda")
    kernel = tilelang.compile(fill, target="cuda", execution_backend="nvrtc")
    launch = kernel.adapter._forward_from_prebuild_lib
    launch(output, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 3), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(output, stream=torch.cuda.current_stream().cuda_stream)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 3), rtol=0, atol=0)


@pytest.mark.parametrize("threads", [32, 128])
def test_measured_calibration_and_frozen_selection(tmp_path, monkeypatch, threads):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=threads):
            x = T.alloc_fragment((128,), "float32")
            y = T.alloc_fragment((128,), "float32")
            T.copy(A, x)
            for i in T.Parallel(128):
                y[i] = x[i] * 2
            T.copy(y, B)

    a = torch.arange(128, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    with GraphLayoutSession(output_directory=tmp_path / "collect", collect=True) as collection:
        tilelang.compile(main, target="cuda", execution_backend="nvrtc",
                         pass_configs={"tl.layout_solver": "maxsat-full"})
    env = environment(tilelang_revision=tilelang.__version__,
                      tvm_revision="907a88c8791ccf33b9874821bc875e7abf624367")
    table = calibrate(collection, tmp_path / "calibration", env, cache_directory=tmp_path / "cache")
    loaded = LatencyTable.read(tmp_path / "calibration/latency-table.json", env)
    assert table.sha256 == loaded.sha256
    progress = json.loads((tmp_path / "calibration/progress.json").read_text())
    assert not progress["failures"] and len(progress["entries"]) == len(collection.measurements)
    assert all(entry["graph_nodes"] == 32 for entry in progress["entries"])
    assert calibrate(collection, tmp_path / "calibration", env).sha256 == table.sha256
    def reject_measurement(*args, **kwargs):
        raise AssertionError("a verified cached measurement must be reused")
    monkeypatch.setattr("tilelang.layout._graph_profile.measure_one", reject_measurement)
    assert calibrate(collection, tmp_path / "reused", env,
                     cache_directory=tmp_path / "cache").sha256 == table.sha256
    objectives = {}
    for algorithm in ("maxsat-full", "treewidth", "local", "greedy"):
        with GraphLayoutSession(output_directory=tmp_path / algorithm, table=loaded) as session:
            kernel = tilelang.compile(main, target="cuda", execution_backend="nvrtc",
                                      pass_configs={"tl.layout_solver": algorithm})
        b.fill_(float("nan"))
        kernel.adapter._forward_from_prebuild_lib(a, b, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(b, 2 * a, rtol=0, atol=0)
        assert all(not report["missing_measurements"] for report in session.reports)
        objectives[algorithm] = [report["result"]["cost"] for report in session.reports]
    assert objectives["maxsat-full"] == objectives["treewidth"]
    with GraphLayoutSession(output_directory=tmp_path / "different-flags", table=loaded):
        with pytest.raises(ValueError, match="compiler flags differ"):
            tilelang.compile(main, target="cuda", execution_backend="nvrtc", compile_flags=["--use_fast_math"],
                             pass_configs={"tl.layout_solver": "maxsat-full"})
