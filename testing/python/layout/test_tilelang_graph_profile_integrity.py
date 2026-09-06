"""CPU checks for calibration recovery and evidence promotion."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


_ROOT = Path(__file__).parents[3]
_PACKAGE_NAME = "_cpu_graph_profile_fixture"
_PACKAGE = types.ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(_ROOT / "tilelang" / "layout")]
sys.modules[_PACKAGE_NAME] = _PACKAGE


def _load(name):
    qualified_name = f"{_PACKAGE_NAME}.{name}"
    spec = importlib.util.spec_from_file_location(
        qualified_name, _ROOT / "tilelang" / "layout" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


latency_table = _load("_latency_table")
graph_profile = _load("_graph_profile")


class _Collection:
    def __init__(self, measurements):
        self.measurements = measurements


def _fixture():
    environment = {field: f"test-{field}" for field in latency_table.ENVIRONMENT_FIELDS}
    environment["compiler_flags"] = []
    environment["timing_protocol"] = graph_profile.digest(graph_profile.PROTOCOL)
    key = latency_table.measurement_key(
        "conversion", operator_ir="fake-ir", dtype="float32", shape=[4],
        layouts=["source", "target"], schedule={"threads": 1})
    identity = latency_table.digest(key)
    return environment, key, identity, _Collection({identity: {"key": key}})


def _install_fake_runtime(monkeypatch, environment):
    monkeypatch.setattr(graph_profile, "environment", lambda **_: environment)
    monkeypatch.setattr(graph_profile, "make_microkernel", lambda measurement: None)


def _install_fake_measurement(monkeypatch, calls):
    def measure_one(measurement, directory, protocol, compile_flags):
        del compile_flags
        key = measurement["key"]
        calls.append(key)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        source = f"source:{graph_profile.digest(key)}\n".encode()
        binary = f"cubin:{graph_profile.digest(key)}\n".encode()
        (directory / "kernel.cu").write_bytes(source)
        (directory / "kernel.cubin").write_bytes(binary)
        entry = dict(
            key=key,
            samples_ns=[10.0, 11.0, 12.0],
            cost_ns=11,
            executable_sha256=hashlib.sha256(binary).hexdigest(),
            source_sha256=hashlib.sha256(source).hexdigest(),
            compile_seconds=0.001,
            graph_nodes=1,
            protocol=dict(protocol),
        )
        (directory / "measurement.json").write_text(
            json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        return entry

    monkeypatch.setattr(graph_profile, "measure_one", measure_one)


def test_failed_preflight_removes_old_table_and_recovers(monkeypatch, tmp_path):
    environment, _, _, collection = _fixture()
    _install_fake_runtime(monkeypatch, environment)
    calls = []
    _install_fake_measurement(monkeypatch, calls)
    directory = tmp_path / "calibration"

    first = graph_profile.calibrate(collection, directory, environment)
    assert (directory / "latency-table.json").is_file()

    different_environment = dict(environment, gpu_name="test-other-gpu")
    monkeypatch.setattr(graph_profile, "environment", lambda **_: different_environment)
    with pytest.raises(ValueError, match="another environment"):
        graph_profile.calibrate(collection, directory, different_environment)
    assert latency_table.LatencyTable.read(directory / "latency-table.json", environment).sha256 == first.sha256
    monkeypatch.setattr(graph_profile, "environment", lambda **_: environment)

    def reject_preflight(measurement):
        raise ValueError("unsupported profiling context")

    monkeypatch.setattr(graph_profile, "make_microkernel", reject_preflight)
    with pytest.raises(RuntimeError, match="preflight failed"):
        graph_profile.calibrate(collection, directory, environment)
    assert not (directory / "latency-table.json").exists()
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    assert progress["failures"][0]["stage"] == "preflight"

    monkeypatch.setattr(graph_profile, "make_microkernel", lambda measurement: None)
    recovered = graph_profile.calibrate(collection, directory, environment)
    assert recovered.sha256 == first.sha256
    assert len(calls) == 1


def test_corrupt_cache_is_rejected_before_remeasurement(monkeypatch, tmp_path):
    environment, _, identity, collection = _fixture()
    _install_fake_runtime(monkeypatch, environment)
    calls = []
    _install_fake_measurement(monkeypatch, calls)
    cache_root = tmp_path / "cache"
    source_directory = tmp_path / "source"
    graph_profile.calibrate(collection, source_directory, environment, cache_directory=cache_root)

    cached_case = cache_root / graph_profile.digest(environment) / identity
    (cached_case / "kernel.cubin").write_bytes(b"partially-written-cubin")

    recovered_directory = tmp_path / "recovered"
    table = graph_profile.calibrate(collection, recovered_directory, environment,
                                    cache_directory=cache_root)
    assert len(calls) == 2
    progress = json.loads((recovered_directory / "progress.json").read_text(encoding="utf-8"))
    assert progress["failures"] == []
    assert progress["cache_rejections"][0]["key"] == collection.measurements[identity]["key"]
    recovered_entry = json.loads(
        (recovered_directory / identity / "measurement.json").read_text(encoding="utf-8"))
    assert (recovered_directory / "latency-table.json").is_file()
    assert hashlib.sha256((recovered_directory / identity / "kernel.cubin").read_bytes()).hexdigest() == recovered_entry[
        "executable_sha256"]
    assert (recovered_directory / identity / "kernel.cubin").read_bytes() != b"partially-written-cubin"
    assert latency_table.LatencyTable.read(recovered_directory / "latency-table.json", environment).sha256 == table.sha256
