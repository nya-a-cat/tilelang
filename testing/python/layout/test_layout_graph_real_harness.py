"""CPU-only checks for phase-attempt isolation in the real-workload harness."""

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import types


_ROOT = Path(__file__).parents[3]
_SOURCE = _ROOT / "benchmarks" / "layout_graph_real.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
_NAMES = {"write", "_prepare_phase_attempt", "main"}
_FUNCTIONS = [node for node in _TREE.body
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _NAMES]


def _run_parent(tmp_path, monkeypatch, failure):
    namespace = {
        "__file__": str(_SOURCE),
        "argparse": argparse,
        "json": json,
        "os": os,
        "Path": Path,
        "subprocess": types.SimpleNamespace(
            run=None, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired),
        "sys": sys,
        "cases": lambda: [{"id": "case-a", "family": "fixture", "shape": [1]}],
    }
    exec(compile(ast.Module(body=_FUNCTIONS, type_ignores=[]), str(_SOURCE), "exec"), namespace)

    class FakeSubprocess:
        def run(self, command, stdout, stderr, timeout, env):
            del command, stderr, timeout, env
            stdout.write(f"current-{failure}\n")
            if failure == "timeout":
                raise subprocess.TimeoutExpired("fixture", 1)
            if failure == "missing-report":
                return types.SimpleNamespace(returncode=0)
            raise OSError("fixture process could not start")

    namespace["subprocess"].run = FakeSubprocess().run
    root = tmp_path / "reports"
    folder = root / "case-a"
    folder.mkdir(parents=True)
    (folder / "calibrate.json").write_text(
        json.dumps({"status": "success", "attempt": "old"}) + "\n", encoding="utf-8")
    (folder / "calibrate.log").write_text("old-attempt-log\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "layout_graph_real.py", "--examples-root", str(tmp_path), "--output", str(root),
        "--compiler-revision", "fixture", "--phase", "calibrate", "--timeout", "1",
    ])
    namespace["main"]()
    summary = json.loads((root / "calibrate-summary.json").read_text(encoding="utf-8"))
    return root, folder, summary[0]


def test_timeout_does_not_reuse_old_phase_report(tmp_path, monkeypatch):
    root, folder, item = _run_parent(tmp_path, monkeypatch, "timeout")

    assert item["status"] == "timeout"
    assert "report" not in item
    assert item["error"] == "current-timeout\n"
    assert json.loads((folder / "calibrate.attempt-0001.json").read_text(encoding="utf-8"))["status"] == "success"
    assert (folder / "calibrate.attempt-0001.log").read_text(encoding="utf-8") == "old-attempt-log\n"
    assert json.loads((root / "calibrate-summary.json").read_text(encoding="utf-8"))[0]["attempt"] == "calibrate.attempt-0002"


def test_start_failure_does_not_reuse_old_phase_report(tmp_path, monkeypatch):
    _, folder, item = _run_parent(tmp_path, monkeypatch, "start-failure")

    assert item["status"] == "start_failed"
    assert "report" not in item
    assert "OSError: fixture process could not start" in item["error"]
    assert json.loads((folder / "calibrate.attempt-0001.json").read_text(encoding="utf-8"))["status"] == "success"
    assert (folder / "calibrate.attempt-0001.log").read_text(encoding="utf-8") == "old-attempt-log\n"


def test_missing_report_is_explicit_failure(tmp_path, monkeypatch):
    _, _, item = _run_parent(tmp_path, monkeypatch, "missing-report")
    assert item["status"] == "missing_report"
    assert "report" not in item
    assert "without calibrate.json" in item["error"]


def test_repeated_attempts_preserve_prior_outputs(tmp_path):
    namespace = {}
    node = next(node for node in _FUNCTIONS if node.name == "_prepare_phase_attempt")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_SOURCE), "exec"), namespace)
    prepare = namespace["_prepare_phase_attempt"]
    assert prepare(tmp_path, "collect") == "collect.attempt-0001"
    for attempt in range(1, 4):
        (tmp_path / "collect.log").write_text(str(attempt), encoding="utf-8")
        assert prepare(tmp_path, "collect") == f"collect.attempt-{attempt + 1:04d}"
    for attempt in range(1, 4):
        assert (tmp_path / f"collect.attempt-{attempt:04d}.log").read_text() == str(attempt)
