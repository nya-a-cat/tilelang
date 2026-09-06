"""Calibration identity, provenance, freezing and missing-cost contracts."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "latency_table", Path(__file__).parents[3] / "tilelang/layout/_latency_table.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    env = {key: "test-" + key for key in module.ENVIRONMENT_FIELDS}
    key = module.measurement_key("conversion", operator_ir="fixture-ir", dtype="float32",
                                 shape=[256], layouts=["a", "b"], schedule={"threads": 128})
    document = dict(schema=1, frozen=True, environment=env, entries=[dict(
        key=key, samples_ns=[10.1, 11.2, 100.0], cost_ns=12, executable_sha256="a" * 64)])
    return env, key, document


def test_frozen_roundtrip_and_directed_keys(tmp_path):
    env, key, doc = fixture()
    table = module.LatencyTable(doc, env)
    assert table.cost(key) == 12
    doc["entries"][0]["cost_ns"] = 999
    assert table.cost(key) == 12
    reverse = dict(key, layouts=list(reversed(key["layouts"])))
    with pytest.raises(module.MissingMeasurement):
        table.cost(reverse)
    assert table.coverage([key, key, reverse])["measured"] == 1
    path = tmp_path / "costs.json"
    table.write(path)
    assert module.LatencyTable.read(path, env).sha256 == table.sha256


def test_environment_and_measurement_validation():
    env, key, doc = fixture()
    changed = dict(env, driver_version="different")
    with pytest.raises(ValueError, match="environment differs"):
        module.LatencyTable(doc, changed)
    mutations = [
        lambda d: d.update(frozen=False),
        lambda d: d["environment"].pop("tvm_revision"),
        lambda d: d["entries"][0].update(cost_ns=0),
        lambda d: d["entries"][0].update(samples_ns=[1, 2]),
        lambda d: d["entries"][0].update(samples_ns=[True, 2, 3]),
        lambda d: d["entries"][0].update(samples_ns=[float("nan"), 2, 3]),
        lambda d: d["entries"][0].update(executable_sha256="x" * 64),
        lambda d: d["entries"].append(copy.deepcopy(d["entries"][0])),
    ]
    for mutate in mutations:
        broken = copy.deepcopy(doc)
        mutate(broken)
        with pytest.raises(ValueError):
            module.LatencyTable(broken, env)
