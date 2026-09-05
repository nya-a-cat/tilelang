"""Bounded weighted MaxSAT for finite, conversion-free layout candidates.

Rows are [memory_cost, layout_id_for_buffer_0, ...]. A -1 layout ID means
the operator does not access that buffer. Register costs are indexed by
buffer and layout ID, and charged once per buffer. Objectives preserve the
compiler's lexicographic (memory, registers) ordering. Each call owns its
Z3 context, including when compilation is concurrent.
"""

from __future__ import annotations


def solve_candidate_table(rows, register_costs, timeout_ms=100):
    """Return a certified optimum or a status without an assignment.

    This is exact only over the supplied rows. No conversions or unseen
    candidate layouts are introduced. Costs must be nonnegative integers.
    """
    import z3

    rows = [[[int(v) for v in row] for row in group] for group in rows]
    register_costs = [[int(v) for v in costs] for costs in register_costs]
    timeout_ms = int(timeout_ms)
    if not 1 <= timeout_ms <= 60_000:
        raise ValueError("layout solver timeout must be in 1..60000 ms")
    if not rows or len(rows) > 32 or sum(map(len, rows)) > 256 or len(register_costs) > 64:
        return {"status": "budget"}
    if any(not group for group in rows) or any(not costs for costs in register_costs):
        return {"status": "unsat"}
    nbuf = len(register_costs)
    if any(cost < 0 for costs in register_costs for cost in costs):
        raise ValueError("negative register cost")
    for group in rows:
        for row in group:
            if len(row) != nbuf + 1 or row[0] < 0:
                raise ValueError("invalid layout candidate row")
            if any(l < -1 or l >= len(register_costs[b]) for b, l in enumerate(row[1:])):
                raise ValueError("layout ID outside candidate domain")

    ctx = z3.Context()
    opt = z3.Optimize(ctx=ctx)
    opt.set(timeout=timeout_ms, priority="lex")
    selected = [[z3.Bool(f"op_{o}_{c}", ctx) for c in range(len(group))] for o, group in enumerate(rows)]
    layouts = [[z3.Bool(f"buf_{b}_{l}", ctx) for l in range(len(costs))] for b, costs in enumerate(register_costs)]
    for group in selected + layouts:
        opt.add(z3.PbEq([(x, 1) for x in group], 1))
    # Use separate weighted-soft groups, preserving lexicographic order.
    memory_objective = opt.add_soft(z3.BoolVal(True, ctx), weight=1, id="memory")
    register_objective = opt.add_soft(z3.BoolVal(True, ctx), weight=1, id="registers")
    for o, group in enumerate(rows):
        for c, row in enumerate(group):
            active = selected[o][c]
            if row[0]:
                opt.add_soft(z3.Not(active), weight=row[0], id="memory")
            for b, l in enumerate(row[1:]):
                if l >= 0:
                    opt.add(z3.Implies(active, layouts[b][l]))
    for b, costs in enumerate(register_costs):
        for l, cost in enumerate(costs):
            if cost:
                opt.add_soft(z3.Not(layouts[b][l]), weight=cost, id="registers")
    status = opt.check()
    if status != z3.sat:
        return {"status": "unsat" if status == z3.unsat else "unknown"}
    model = opt.model()
    choices = [next(c for c, active in enumerate(group) if z3.is_true(model.eval(active))) for group in selected]
    chosen_layouts = [next(l for l, active in enumerate(group) if z3.is_true(model.eval(active))) for group in layouts]
    mem = sum(rows[o][c][0] for o, c in enumerate(choices))
    regs = sum(costs[l] for costs, l in zip(register_costs, chosen_layouts))
    # Independently rescore the returned model; no timeout incumbent is
    # presented as optimal, and no ambiguous soft-objective grouping is accepted.
    encoded = [model.eval(expr).as_long() for expr in opt.objectives()]
    if encoded != [mem, regs]:
        raise RuntimeError(f"layout objective mismatch: {encoded} vs {[mem, regs]}")
    for objective, cost in zip((memory_objective, register_objective), (mem, regs)):
        bounds = (objective.lower(), objective.upper())
        if any(not z3.is_int_value(bound) or bound.as_long() != cost for bound in bounds):
            return {"status": "unknown"}
    return {"status": "optimal", "memory": mem, "registers": regs, "choices": choices}


def register_solver():
    """Register the narrow integer-table interface used by the C++ pass."""
    import tvm_ffi

    def solve(rows, register_costs, timeout_ms):
        result = solve_candidate_table(rows, register_costs, timeout_ms)
        if result["status"] != "optimal":
            return result["status"]
        return " ".join(map(str, ["optimal", result["memory"], result["registers"], *result["choices"]]))

    tvm_ffi.register_global_func("tl.layout.solve_candidate_table", solve)
