"""Finite layout selection with shared, directed conversions.

This module has no native TileLang dependency. A problem contains ``tensors``
(a name, a nonempty layout-name domain, and a square ``conversion_costs`` table)
and topologically ordered ``operators``. Each operator has ordered ``inputs``
and ``outputs`` tensor IDs and ``configs`` with matching input/output layout
IDs and a nonnegative integer ``cost``. Every tensor has exactly one producer.
Input/output boundary constraints are represented by zero-cost source/sink
operators with the permitted configurations. Repeated inputs remain distinct.

None denotes a forbidden conversion. Missing measurement data must be rejected
by the cost-table reader before it constructs this problem. The objective is
operator cost plus one conversion per (tensor value, distinct consumer layout),
as in Eisenhofer et al., arXiv:2608.21555v1, Sections 2 and 3.2.
"""

from __future__ import annotations

from collections import defaultdict
import time


class _BudgetExceeded(Exception):
    pass


def _integer(value, what):
    if type(value) is not int or value < 0:
        raise ValueError(f"{what} must be a nonnegative integer")
    return value


def validate(problem):
    """Validate the graph and return producer and argument-position use lists."""
    tensors, operators = problem["tensors"], problem["operators"]
    producer = [None] * len(tensors)
    uses = [[] for _ in tensors]
    for t, tensor in enumerate(tensors):
        domain = tensor["layouts"]
        if not domain or len(set(domain)) != len(domain):
            raise ValueError(f"tensor {t}: empty or duplicate layout domain")
        table = tensor["conversion_costs"]
        if len(table) != len(domain) or any(len(row) != len(domain) for row in table):
            raise ValueError(f"tensor {t}: invalid conversion table dimensions")
        for a, row in enumerate(table):
            for b, cost in enumerate(row):
                if cost is not None:
                    _integer(cost, "conversion cost")
                if a == b and cost != 0:
                    raise ValueError("identity conversion must cost zero")
    for o, op in enumerate(operators):
        for t in op["inputs"] + op["outputs"]:
            if type(t) is not int or not 0 <= t < len(tensors):
                raise ValueError(f"operator {o}: invalid tensor ID")
        if len(set(op["outputs"])) != len(op["outputs"]):
            raise ValueError("an output value must be unique; version mutable buffers first")
        for i, t in enumerate(op["inputs"]):
            if producer[t] is None:
                raise ValueError("operators must be topological and every input must have a producer")
            uses[t].append((o, i))
        for i, t in enumerate(op["outputs"]):
            if producer[t] is not None:
                raise ValueError("multiple producers; version mutable buffers first")
            producer[t] = (o, i)
        for config in op["configs"]:
            _integer(config["cost"], "operator cost")
            for side in ("inputs", "outputs"):
                if len(config[side]) != len(op[side]):
                    raise ValueError(f"operator {o}: configuration arity mismatch")
                for t, layout in zip(op[side], config[side]):
                    if type(layout) is not int or not 0 <= layout < len(tensors[t]["layouts"]):
                        raise ValueError(f"operator {o}: layout outside domain")
    if any(p is None for p in producer):
        raise ValueError("every tensor must have a producer, including external inputs")
    return producer, uses


def _score(problem, choices, producer, uses, partial=False):
    """Recompute without solver expressions; return None for illegal plans."""
    ops = problem["operators"]
    if not partial and len(choices) != len(ops):
        raise ValueError("assignment length does not match operator count")
    if any(type(c) is not int or not 0 <= c < len(ops[o]["configs"]) for o, c in enumerate(choices)):
        raise ValueError("configuration index outside domain")
    cost = sum(ops[o]["configs"][c]["cost"] for o, c in enumerate(choices))
    conversions = []
    for t, (o, out) in enumerate(producer):
        if o >= len(choices):
            continue
        source = ops[o]["configs"][choices[o]]["outputs"][out]
        requested = sorted({ops[u]["configs"][choices[u]]["inputs"][i] for u, i in uses[t] if u < len(choices)})
        for target in requested:
            penalty = problem["tensors"][t]["conversion_costs"][source][target]
            if penalty is None:
                return None
            cost += penalty
            if source != target:
                conversions.append({"tensor": t, "source": source, "target": target, "cost": penalty})
    return cost, conversions


def evaluate(problem, choices):
    """Independent plan validation, objective recomputation and conversion list."""
    producer, uses = validate(problem)
    result = _score(problem, choices, producer, uses)
    if result is None:
        return {"status": "infeasible"}
    cost, conversions = result
    return {"status": "feasible", "cost": cost, "choices": list(choices), "conversions": conversions}


def _maxsat(problem, timeout_ms, check):
    import z3

    producer, uses = validate(problem)
    ctx = z3.Context()
    opt = z3.Optimize(ctx=ctx)
    # Z3 4.15.4's default maxres can return a model whose evaluated cost differs
    # from its closed bounds on shared-conversion DAGs. wmax passes the same
    # exhaustive regression corpus; the independent certificate check remains.
    opt.set(timeout=timeout_ms, maxsat_engine="wmax")
    selected = []
    objective = opt.add_soft(z3.BoolVal(True, ctx), weight=1)
    for o, op in enumerate(problem["operators"]):
        check()
        group = [z3.Bool(f"op_{o}_{c}", ctx) for c in range(len(op["configs"]))]
        if not group:
            return {"status": "unsat"}
        opt.add(z3.PbEq([(v, 1) for v in group], 1))
        selected.append(group)
        for active, config in zip(group, op["configs"]):
            if config["cost"]:
                opt.add_soft(z3.Not(active), weight=str(config["cost"]))
    for t, tensor in enumerate(problem["tensors"]):
        check()
        o, out = producer[t]
        for target in range(len(tensor["layouts"])):
            requested = z3.Bool(f"requested_{t}_{target}", ctx)
            consumers = [selected[u][c] for u, i in uses[t] for c, config in enumerate(problem["operators"][u]["configs"])
                         if config["inputs"][i] == target]
            # Equivalence makes zero-cost conversion plans deterministic to decode.
            opt.add(requested == z3.Or(*consumers, z3.BoolVal(False, ctx)))
            for active, config in zip(selected[o], problem["operators"][o]["configs"]):
                source = config["outputs"][out]
                penalty = tensor["conversion_costs"][source][target]
                clause = z3.Or(z3.Not(active), z3.Not(requested))
                if penalty is None:
                    opt.add(clause)
                elif penalty:
                    opt.add_soft(clause, weight=str(penalty))
    remaining = check()
    opt.set(timeout=max(1, int(remaining)), maxsat_engine="wmax")
    answer = opt.check()
    if answer == z3.unsat:
        return {"status": "unsat"}
    if answer != z3.sat:
        return {"status": "unknown", "reason": opt.reason_unknown()}
    model = opt.model()
    choices = [next(c for c, active in enumerate(group) if z3.is_true(model.eval(active))) for group in selected]
    result = evaluate(problem, choices)
    if result["status"] != "feasible":
        raise RuntimeError("MaxSAT returned an infeasible assignment")
    value = model.eval(opt.objectives()[0]).as_long()
    if value != result["cost"]:
        raise RuntimeError("MaxSAT objective differs from independently computed cost")
    bounds = (objective.lower(), objective.upper())
    if any(not z3.is_int_value(b) or b.as_long() != value for b in bounds):
        return {"status": "unknown", "reason": "optimality bounds are not closed",
                "bounds": [str(b) for b in bounds], "evaluated_cost": value}
    return dict(result, status="optimal", lower_bound=value, upper_bound=value)


def _adjacency(problem, producer, uses):
    graph = [set() for _ in problem["operators"]]
    for t, (p, _) in enumerate(producer):
        for u, _ in uses[t]:
            graph[p].add(u)
            graph[u].add(p)
    return graph


def tree_decomposition(problem, check=lambda: None):
    """Deterministic min-fill decomposition, with an empty synthetic root.

    The decomposition need not have minimum width for the DP to be exact.
    Child and parent indices describe a rooted tree; all original graph edges
    and the connected occurrence set of each vertex are checked independently.
    """
    producer, uses = validate(problem)
    original = _adjacency(problem, producer, uses)
    graph = [set(n) for n in original]
    remaining = set(range(len(graph)))
    order, bags = [], []
    while remaining:
        check()
        def key(v):
            neighbors = sorted(graph[v] & remaining)
            missing = sum(b not in graph[a] for i, a in enumerate(neighbors) for b in neighbors[i + 1:])
            return missing, len(neighbors), v
        v = min(remaining, key=key)
        neighbors = graph[v] & remaining
        bags.append(tuple(sorted({v} | neighbors)))
        order.append(v)
        for a in neighbors:
            graph[a].update(neighbors - {a})
        remaining.remove(v)
    position = {v: i for i, v in enumerate(order)}
    parents = []
    for i, v in enumerate(order):
        rest = set(bags[i]) - {v}
        parents.append(min((position[u] for u in rest), default=len(bags)))
    bags.append(())
    parents.append(-1)
    validate_decomposition(original, bags, parents)
    return bags, parents


def validate_decomposition(graph, bags, parents):
    if len(bags) != len(parents) or not bags:
        raise ValueError("invalid decomposition size")
    roots = [i for i, p in enumerate(parents) if p == -1]
    if len(roots) != 1:
        raise ValueError("decomposition must have one root")
    tree = [set() for _ in bags]
    for i, parent in enumerate(parents):
        if parent == -1:
            continue
        if not 0 <= parent < len(bags) or parent == i:
            raise ValueError("invalid decomposition parent")
        tree[i].add(parent)
        tree[parent].add(i)
    visited = {roots[0]}
    pending = [roots[0]]
    while pending:
        i = pending.pop()
        for j in tree[i] - visited:
            visited.add(j)
            pending.append(j)
    if len(visited) != len(bags):
        raise ValueError("disconnected or cyclic decomposition")
    vertices = set(range(len(graph)))
    if set().union(*(set(b) for b in bags)) != vertices:
        raise ValueError("decomposition vertex coverage mismatch")
    for v, neighbors in enumerate(graph):
        occurrences = {i for i, b in enumerate(bags) if v in b}
        reached = {min(occurrences)}
        pending = list(reached)
        while pending:
            i = pending.pop()
            for j in (tree[i] & occurrences) - reached:
                reached.add(j)
                pending.append(j)
        if reached != occurrences:
            raise ValueError("disconnected bags for a vertex")
        if any(not any(v in b and u in b for b in bags) for u in neighbors):
            raise ValueError("decomposition misses an edge")


def _treewidth(problem, timeout_ms, check, max_states):
    """Appendix B DP: bag labels plus forgotten-consumer request bitsets.

    Adjacent original bags are expanded to introduce/forget paths on demand.
    Equal bags are joined by unioning requests. Each entry retains a persistent
    derivation node, allowing linear backtracking after the root optimum.
    """
    producer, uses = validate(problem)
    ops, tensors = problem["operators"], problem["tensors"]
    if any(not op["configs"] for op in ops):
        return {"status": "unsat"}
    bags, parents = tree_decomposition(problem, check)
    children = [[] for _ in bags]
    for i, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(i)
    peak = 0
    processed = 0

    def outputs(bag):
        return tuple(t for o in bag for t in ops[o]["outputs"])

    def put(table, key, cost, derivation):
        nonlocal peak, processed
        processed += 1
        if processed % 256 == 0:
            check()
        prior = table.get(key)
        if prior is None or cost < prior[0]:
            table[key] = (cost, derivation)
            peak = max(peak, len(table))
            if len(table) > max_states:
                raise _BudgetExceeded("treewidth state budget")

    def transform(table, bag, target):
        for v in sorted(set(bag) - set(target)):
            check()
            newbag = tuple(o for o in bag if o != v)
            tracked, newtracked = outputs(bag), outputs(newbag)
            result = {}
            for (labels, masks), (cost, derivation) in table.items():
                assignment, requests = dict(zip(bag, labels)), dict(zip(tracked, masks))
                config = ops[v]["configs"][assignment[v]]
                total = cost + config["cost"]
                valid = True
                for t, source in zip(ops[v]["outputs"], config["outputs"]):
                    requested = requests[t]
                    for u, i in uses[t]:
                        if u in assignment and u != v:
                            requested |= 1 << ops[u]["configs"][assignment[u]]["inputs"][i]
                    for layout in range(len(tensors[t]["layouts"])):
                        if requested & (1 << layout):
                            penalty = tensors[t]["conversion_costs"][source][layout]
                            if penalty is None:
                                valid = False
                                break
                            total += penalty
                    if not valid:
                        break
                if not valid:
                    continue
                for t, layout in zip(ops[v]["inputs"], config["inputs"]):
                    if t in newtracked:
                        requests[t] |= 1 << layout
                key = (tuple(assignment[o] for o in newbag), tuple(requests[t] for t in newtracked))
                put(result, key, total, ("forget", v, assignment[v], derivation))
            table, bag = result, newbag
        for v in sorted(set(target) - set(bag)):
            check()
            newbag = tuple(sorted((*bag, v)))
            tracked, newtracked = outputs(bag), outputs(newbag)
            result = {}
            for (labels, masks), (cost, derivation) in table.items():
                assignment, requests = dict(zip(bag, labels)), dict(zip(tracked, masks))
                for c in range(len(ops[v]["configs"])):
                    assignment[v] = c
                    key = (tuple(assignment[o] for o in newbag), tuple(requests.get(t, 0) for t in newtracked))
                    put(result, key, cost, derivation)
            table, bag = result, newbag
        return table

    # Elimination bags' parents have larger indices, so this order is bottom-up.
    tables = {}
    for i, bag in enumerate(bags):
        check()
        table = None
        for child in children[i]:
            branch = transform(tables.pop(child), bags[child], bag)
            if table is None:
                table = branch
                continue
            grouped = defaultdict(list)
            for (labels, masks), value in branch.items():
                grouped[labels].append((masks, value))
            joined = {}
            for (labels, masks), (cost, derivation) in table.items():
                for other_masks, (other_cost, other_derivation) in grouped[labels]:
                    key = (labels, tuple(a | b for a, b in zip(masks, other_masks)))
                    put(joined, key, cost + other_cost, ("join", derivation, other_derivation))
            table = joined
        if table is None:
            table = transform({((), ()): (0, None)}, (), bag)
        tables[i] = table
    entry = tables[len(bags) - 1].get(((), ()))
    if entry is None:
        return {"status": "unsat", "treewidth": max(map(len, bags)) - 1}
    cost, derivation = entry
    choices = [None] * len(ops)
    pending = [derivation]
    while pending:
        node = pending.pop()
        if node is None:
            continue
        if node[0] == "join":
            pending.extend(node[1:])
        else:
            _, o, c, child = node
            if choices[o] is not None:
                raise RuntimeError("operator charged twice by tree decomposition")
            choices[o] = c
            pending.append(child)
    result = evaluate(problem, choices)
    if result.get("cost") != cost:
        raise RuntimeError("treewidth objective differs from independent cost")
    return dict(result, status="optimal", lower_bound=cost, upper_bound=cost,
                treewidth=max(map(len, bags)) - 1, peak_table_states=peak, transitions=processed)


def _heuristic(problem, algorithm, check):
    producer, uses = validate(problem)
    choices = []
    for o, op in enumerate(problem["operators"]):
        check()
        candidates = []
        for c, config in enumerate(op["configs"]):
            if algorithm == "local":
                candidates.append((config["cost"], c))
            else:
                scored = _score(problem, [*choices, c], producer, uses, partial=True)
                if scored is not None:
                    candidates.append((scored[0], c))
        if not candidates:
            return {"status": "infeasible", "reason": "no feasible heuristic continuation"}
        choices.append(min(candidates)[1])
    scored = _score(problem, choices, producer, uses)
    if scored is None:
        return {"status": "infeasible", "reason": "local choices require forbidden conversion"}
    if algorithm == "greedy":
        improved = True
        while improved:
            improved = False
            for o, op in enumerate(problem["operators"]):
                for c in range(len(op["configs"])):
                    check()
                    trial = choices.copy()
                    trial[o] = c
                    candidate = _score(problem, trial, producer, uses)
                    if candidate is not None and candidate[0] < scored[0]:
                        choices, scored, improved = trial, candidate, True
    return dict(evaluate(problem, choices), algorithm=algorithm)


def solve(problem, algorithm="maxsat-full", timeout_ms=60_000, max_states=1_000_000):
    """Solve a complete supplied finite domain; never hide budget truncation."""
    if algorithm not in ("maxsat-full", "treewidth", "local", "greedy"):
        raise ValueError("unknown graph layout strategy")
    if type(timeout_ms) is not int or timeout_ms <= 0 or type(max_states) is not int or max_states <= 0:
        raise ValueError("solver budgets must be positive integers")
    validate(problem)
    started = time.monotonic()
    def check():
        remaining = timeout_ms - (time.monotonic() - started) * 1000
        if remaining <= 0:
            raise _BudgetExceeded("total solver time budget")
        return remaining
    try:
        if algorithm == "maxsat-full":
            result = _maxsat(problem, timeout_ms, check)
        elif algorithm == "treewidth":
            result = _treewidth(problem, timeout_ms, check, max_states)
        else:
            result = _heuristic(problem, algorithm, check)
    except _BudgetExceeded as exc:
        result = {"status": "budget", "reason": str(exc)}
    return dict(result, algorithm=algorithm, elapsed_ms=(time.monotonic() - started) * 1000)


def register_solver():
    """Register a versioned JSON boundary without narrowing integer costs."""
    import json
    import tvm_ffi

    def solve_json(problem, algorithm, timeout_ms, max_states):
        result = solve(json.loads(str(problem)), str(algorithm), int(timeout_ms), int(max_states))
        return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)

    tvm_ffi.register_global_func("tl.layout.solve_graph_v1", solve_json)
