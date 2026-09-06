"""Independent exact-cost checks; no native compiler or GPU is required."""

import copy
import importlib.util
import itertools
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

_path = Path(__file__).resolve().parents[3] / "tilelang/layout/_graph_solver.py"
_spec = importlib.util.spec_from_file_location("graph_layout_solver", _path)
solver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(solver)


def tensor(costs=((0, 3), (5, 0))):
    return dict(name="value", layouts=list(range(len(costs))), conversion_costs=[list(r) for r in costs])


def op(inputs, outputs, configs):
    return dict(name="operator", inputs=inputs, outputs=outputs,
                configs=[dict(inputs=i, outputs=o, cost=c) for i, o, c in configs])


def brute(problem):
    """Enumerate configurations and charge independently collected requests."""
    best = None
    for indices in itertools.product(*(range(len(o["configs"])) for o in problem["operators"])):
        produced, consumed, total = {}, {}, 0
        for operator, index in zip(problem["operators"], indices):
            config = operator["configs"][index]
            total += config["cost"]
            for t, layout in zip(operator["outputs"], config["outputs"]):
                produced[t] = layout
            for t, layout in zip(operator["inputs"], config["inputs"]):
                consumed.setdefault(t, set()).add(layout)
        feasible = True
        for t, layouts in consumed.items():
            for layout in layouts:
                cost = problem["tensors"][t]["conversion_costs"][produced[t]][layout]
                if cost is None:
                    feasible = False
                else:
                    total += cost
        if feasible and (best is None or total < best):
            best = total
    return best


def check_exact(problem):
    expected = brute(problem)
    for algorithm in ("maxsat-full", "treewidth"):
        result = solver.solve(problem, algorithm)
        if expected is None:
            assert result["status"] == "unsat", (algorithm, result, problem)
        else:
            assert result["status"] == "optimal", (algorithm, result, problem)
            assert result["cost"] == expected, (algorithm, result, problem)
            assert result["lower_bound"] == result["upper_bound"] == expected
            assert solver.evaluate(problem, result["choices"])["cost"] == expected
    return expected


def test_fanout_shared_conversion_join_and_repeated_argument():
    # Three consumers, including duplicate argument use, share one conversion.
    problem = dict(tensors=[tensor()], operators=[op([], [0], [([], [0], 0)])] +
                   [op([0, 0], [], [([1, 1], [], 2)]) for _ in range(3)])
    assert check_exact(problem) == 9
    for algorithm in ("maxsat-full", "treewidth", "local", "greedy"):
        result = solver.solve(problem, algorithm)
        assert result["conversions"] == [dict(tensor=0, source=0, target=1, cost=3)]


def test_directed_three_layout_cost_and_fixed_boundary():
    problem = dict(tensors=[tensor(((0, 2, 11), (13, 0, 7), (17, 19, 0)))], operators=[
        op([], [0], [([], [1], 0)]),
        op([0, 0], [], [([0, 2], [], 1)]),
        op([0], [], [([2], [], 0)]),
    ])
    assert check_exact(problem) == 21


def test_local_and_greedy_have_distinct_objectives():
    problem = dict(tensors=[tensor()], operators=[
        op([], [0], [([], [0], 0), ([], [1], 2)]),
        op([0], [], [([1], [], 0), ([0], [], 8)]),
    ])
    assert check_exact(problem) == 2
    assert solver.solve(problem, "local")["cost"] == 3
    assert solver.solve(problem, "greedy")["cost"] == 2


def test_random_dags_against_exhaustive():
    rng = random.Random(2026090602)
    for _ in range(250):
        tensors, operators = [], []
        for o in range(rng.randrange(1, 7)):
            inputs = [rng.randrange(len(tensors)) for _ in range(rng.randrange(4))] if tensors else []
            outputs = list(range(len(tensors), len(tensors) + rng.randrange(1, 3)))
            for _t in outputs:
                tensors.append(tensor(((0, rng.choice([None, 0, 2, 7])), (rng.choice([None, 0, 3, 11]), 0))))
            configs = [([rng.randrange(2) for _i in inputs], [rng.randrange(2) for _t in outputs], rng.randrange(8))
                       for _c in range(rng.randrange(1, 4))]
            operators.append(op(inputs, outputs, configs))
        problem = dict(tensors=tensors, operators=operators)
        check_exact(problem)
        optimum = brute(problem)
        for algorithm in ("local", "greedy"):
            result = solver.solve(problem, algorithm)
            if result["status"] == "feasible":
                assert optimum is not None and result["cost"] >= optimum


def test_budget_is_explicit_and_old_size_limit_removed():
    problem = dict(tensors=[tensor() for _ in range(70)],
                   operators=[op([], [i], [([], [0], 0), ([], [1], 1)]) for i in range(70)])
    for algorithm in ("maxsat-full", "treewidth"):
        assert solver.solve(problem, algorithm)["cost"] == 0
    assert solver.solve(problem, "treewidth", max_states=1)["status"] == "budget"


def test_zero_large_cost_unsat_and_empty_graph():
    assert check_exact(dict(tensors=[], operators=[])) == 0
    problem = dict(tensors=[tensor(((0, 2**65), (1, 0)))], operators=[
        op([], [0], [([], [0], 2**65)]), op([0], [], [([1], [], 0)])])
    assert check_exact(problem) == 2**66
    problem["tensors"][0]["conversion_costs"][0][1] = None
    assert check_exact(problem) is None
    problem["operators"][1]["configs"] = []
    assert check_exact(problem) is None


def test_value_versions_do_not_share_conversions():
    problem = dict(tensors=[tensor(), tensor()], operators=[
        op([], [0], [([], [0], 0)]), op([0], [], [([1], [], 0)]),
        op([], [1], [([], [0], 0)]), op([1], [], [([1], [], 0)])])
    assert check_exact(problem) == 6
    assert len(solver.solve(problem)["conversions"]) == 2


def test_validation_rejects_silent_coercions_and_invalid_graphs():
    problem = dict(tensors=[tensor()], operators=[op([], [0], [([], [0], 0)])])
    for value in (-1, 1.2, True, "3"):
        bad = copy.deepcopy(problem)
        bad["operators"][0]["configs"][0]["cost"] = value
        with pytest.raises(ValueError):
            solver.solve(bad)
    with pytest.raises(ValueError, match="multiple producers"):
        solver.solve(dict(tensors=[tensor()], operators=problem["operators"] * 2))
    with pytest.raises(ValueError, match="topological"):
        solver.solve(dict(tensors=[tensor()], operators=[op([0], [0], [([0], [0], 0)])]))
    bad = copy.deepcopy(problem)
    bad["operators"][0]["configs"][0]["outputs"] = [2]
    with pytest.raises(ValueError, match="outside domain"):
        solver.solve(bad)


def test_decomposition_validator_rejects_broken_running_intersection():
    with pytest.raises(ValueError, match="disconnected bags"):
        solver.validate_decomposition([{1}, {0}], [(0, 1), (1,), (0, 1)], [1, 2, -1])


def test_independent_z3_contexts():
    problem = dict(tensors=[tensor()], operators=[
        op([], [0], [([], [0], 0), ([], [1], 2)]), op([0], [], [([1], [], 0)])])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: solver.solve(problem), range(16)))
    assert all(r["status"] == "optimal" and r["cost"] == 2 for r in results)
