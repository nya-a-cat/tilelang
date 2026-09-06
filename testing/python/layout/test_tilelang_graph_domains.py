"""Exhaustive ownership checks for declared generic fragment domains."""
import importlib.util
import itertools
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "graph_domains", Path(__file__).parents[3] / "tilelang/layout/_graph_domains.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("shape,threads", [([256], 128), ([16, 32], 128), ([8, 8, 4], 64), ([32], 128), ([6, 8], 32)])
def test_all_owners_form_bijection(shape, threads):
    domains = module.domain_specs(shape, threads, 16)
    assert domains
    for domain in domains:
        physical = set()
        for indices in itertools.product(*(range(n) for n in shape)):
            for replica in range(domain["replicas"]):
                thread, register = module.coordinates(indices, shape, domain, replica)
                assert 0 <= thread < threads
                assert 0 <= register < math.prod(shape) * domain["replicas"] // threads
                assert (thread, register) not in physical
                physical.add((thread, register))
        assert len(physical) == math.prod(shape) * domain["replicas"]


def test_complete_domain_budget_and_independent_axis_orders():
    domains = module.domain_specs([16, 32], 128, 32)
    assert {tuple(d["order"]) for d in domains} == {(0, 1), (1, 0)}
    assert len({tuple(d["thread_factors"]) for d in domains}) > 1
    assert len({d["vector_width"] for d in domains}) > 1
    with pytest.raises(module.DomainBudgetExceeded):
        module.domain_specs([16, 32], 128, 32, len(domains) - 1)
