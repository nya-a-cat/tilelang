"""Exact finite-table tests, runnable without loading TileLang's native library."""

import importlib.util
import itertools
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

_path = Path(__file__).resolve().parents[3] / "tilelang/layout/_solver.py"
_spec = importlib.util.spec_from_file_location("layout_solver", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
solve = _module.solve_candidate_table


def exhaustive(rows, registers):
    best = None
    for choices in itertools.product(*(range(len(group)) for group in rows)):
        for layouts in itertools.product(*(range(len(costs)) for costs in registers)):
            if all(all(l < 0 or layouts[b] == l for b, l in enumerate(rows[o][c][1:])) for o, c in enumerate(choices)):
                cost = (sum(rows[o][c][0] for o, c in enumerate(choices)), sum(registers[b][l] for b, l in enumerate(layouts)))
                best = cost if best is None else min(best, cost)
    return best


def test_random_tables_against_exhaustive():
    rng = random.Random(20260906)
    for _ in range(100):
        costs = [[rng.randrange(8) for _ in range(2)] for _ in range(3)]
        rows = [[[rng.randrange(10), *(rng.randrange(-1, 2) for _ in costs)] for _ in range(3)] for _ in range(3)]
        expected = exhaustive(rows, costs)
        result = solve(rows, costs, 5000)
        if expected is None:
            assert result["status"] == "unsat"
        else:
            assert result["status"] == "optimal"
            assert (result["memory"], result["registers"]) == expected


def test_shared_register_charged_once_and_lexicographic():
    result = solve([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], [[1000000, 1]], 5000)
    assert (result["memory"], result["registers"]) == (0, 1000000)
    result = solve([[[2**40, 0], [2**40 + 1, 1]]], [[2**40, 0]], 5000)
    assert (result["memory"], result["registers"]) == (2**40, 2**40)


def test_independent_composition():
    result = solve([[[8, 0, -1], [1, 1, -1]], [[2, -1, 1], [7, -1, 0]]], [[1, 2], [1, 2]], 5000)
    assert result["choices"] == [1, 0]
    assert (result["memory"], result["registers"]) == (3, 4)


def test_budget_and_zero_objectives():
    assert solve([[[0]]] * 33, [], 5000)["status"] == "budget"
    result = solve([[[0, 0]]], [[0]], 5000)
    assert (result["memory"], result["registers"]) == (0, 0)


def test_concurrent_contexts():
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: solve([[[3, 0], [1, 1]]], [[2, 4]], 5000), range(16)))
    assert all((r["memory"], r["registers"]) == (1, 4) for r in results)
