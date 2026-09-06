"""Independent finite fragment domains for the CUDA graph layout experiment."""

from __future__ import annotations

import itertools
import math


class DomainBudgetExceeded(RuntimeError):
    pass


def factorisations(shape, threads):
    """All axis thread factors for the minimum-replication tensor domain."""
    participants = math.gcd(math.prod(shape), threads)

    def visit(axis, remaining, factors):
        if axis == len(shape):
            if remaining == 1:
                yield tuple(factors)
            return
        for factor in range(1, min(shape[axis], remaining) + 1):
            if shape[axis] % factor == 0 and remaining % factor == 0:
                yield from visit(axis + 1, remaining // factor, [*factors, factor])

    yield from visit(0, participants, [])


def domain_specs(shape, threads, dtype_bits, max_candidates=4096):
    """Enumerate axis orders, thread factorizations and register vector chunks.

    The root/native layout is supplied separately. This finite generic domain
    uses the minimum replication compatible with the launch and vector widths
    in {1, 2, 4, 8}, capped by a 128-bit vector. Every configuration in this
    declared domain is enumerated or an explicit budget error is raised.
    """
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("fragment domain requires positive static dimensions")
    if type(threads) is not int or threads <= 0 or threads > 1024:
        raise ValueError("invalid CUDA thread count")
    if type(dtype_bits) is not int or dtype_bits <= 0:
        raise ValueError("invalid element bit width")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("candidate budget must be positive")
    factors = list(factorisations(shape, threads))
    result = []
    for order in itertools.permutations(range(len(shape))):
        for thread_factors in factors:
            axis = order[-1]
            for width in (1, 2, 4, 8):
                if width * dtype_bits > 128 or (shape[axis] // thread_factors[axis]) % width:
                    continue
                result.append(dict(order=order, thread_factors=thread_factors,
                                   vector_width=width, replicas=threads // math.prod(thread_factors)))
                if len(result) > max_candidates:
                    raise DomainBudgetExceeded(f"complete fragment domain exceeds {max_candidates} candidates")
    return result


def coordinates(indices, shape, spec, replica=0):
    """Forward map shared by native expression construction and integer tests."""
    order, factors = spec["order"], spec["thread_factors"]
    thread = 0
    register = 0
    for axis in order:
        chunk = spec["vector_width"] if axis == order[-1] else 1
        index = indices[axis]
        thread = thread * factors[axis] + index // chunk % factors[axis]
        physical = index // (chunk * factors[axis]) * chunk + index % chunk
        register = register * (shape[axis] // factors[axis]) + physical
    thread = thread + replica * math.prod(factors)
    return thread, register


def make_fragment(shape, spec):
    from .fragment import Fragment

    def forward(*arguments):
        if spec["replicas"] > 1:
            return coordinates(arguments[:-1], shape, spec, arguments[-1].var)
        return coordinates(arguments, shape, spec)

    return Fragment(shape, forward_fn=forward, replicate=spec["replicas"])
