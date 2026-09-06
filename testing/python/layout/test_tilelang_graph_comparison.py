"""CPU-only checks for the real-workload report's timing rank comparisons."""

import ast
import itertools
from pathlib import Path

import pytest


def load_comparison():
    source = Path(__file__).resolve().parents[3] / "benchmarks/layout_graph_real.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "comparison")
    namespace = {"itertools": itertools}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["comparison"]


@pytest.mark.parametrize("slow, expected_pairs", [(101.0, 0), (101.005, 1), (102.0, 1)])
def test_timing_tie_is_order_independent(slow, expected_pairs):
    compare = load_comparison()
    root = dict(strategy="root", median_us=100.0, binary_sha256="root")
    left = dict(strategy="local", median_us=slow, binary_sha256="local",
                unweighted_region_objective_ns=10, unweighted_root_objective_ns=10)
    right = dict(strategy="greedy", median_us=100.0, binary_sha256="greedy",
                 unweighted_region_objective_ns=20, unweighted_root_objective_ns=10)
    forward = compare({"results": [root, left, right]})
    reverse = compare({"results": [root, right, left]})
    for result in (forward, reverse):
        assert result["comparable_pairs"] == expected_pairs
        assert result["discordant_pairs"] == expected_pairs
        assert result["pairwise_rank_error"] == (1.0 if expected_pairs else None)


def test_identical_binary_remains_a_tie():
    result = load_comparison()({"results": [
        dict(strategy="root", median_us=100.0, binary_sha256="same"),
        dict(strategy="local", median_us=105.0, binary_sha256="same",
             unweighted_region_objective_ns=10, unweighted_root_objective_ns=20),
    ]})
    assert result["comparable_pairs"] == 0
    assert result["rank_pairs"][0]["measured_direction"] == 0
