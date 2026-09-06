"""Finite intrakernel layout graph construction and value-version rewriting."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
import hashlib
import json
import time
from pathlib import Path

from ._graph_domains import DomainBudgetExceeded, domain_specs, make_fragment
from ._graph_solver import solve, evaluate
from ._latency_table import LatencyTable, MissingMeasurement, digest, measurement_key
from .partial_fragment import PartialFragment


_active_session = contextvars.ContextVar("tilelang_graph_layout_session", default=None)
_compile_flags = contextvars.ContextVar("tilelang_graph_compile_flags", default=())


@contextmanager
def compiler_configuration(flags):
    """Carry the actual JIT flags into every requested measurement identity."""
    flags = list(flags)
    session = _active_session.get()
    if session is not None and session.table is not None:
        if flags != session.table.environment["compiler_flags"]:
            raise ValueError("compiler flags differ from the frozen calibration")
    token = _compile_flags.set(tuple(flags))
    try:
        yield
    finally:
        _compile_flags.reset(token)


class GraphLayoutSession:
    """Collect calibration cases or select layouts using a frozen latency table.

    Collection explicitly preserves the root statements and records unmeasured
    costs as null. Selection requires every legal requested measurement. The
    output directory contains the finite domains, exclusions, costs and plans.
    """

    def __init__(self, *, output_directory, table=None, collect=False,
                 max_candidates=4096, timeout_ms=60_000, max_states=1_000_000):
        if not collect and not isinstance(table, LatencyTable):
            raise ValueError("layout selection requires a frozen LatencyTable")
        if collect and table is not None:
            raise ValueError("calibration collection and frozen selection are separate modes")
        for value in (max_candidates, timeout_ms, max_states):
            if type(value) is not int or value < 1:
                raise ValueError("layout budgets must be positive integers")
        self.output_directory = Path(output_directory)
        self.table, self.collect = table, collect
        self.max_candidates, self.timeout_ms, self.max_states = max_candidates, timeout_ms, max_states
        self.measurements = {}
        self.reports = []
        self._token = None

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("layout session is already active")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._token = _active_session.set(self)
        return self

    def __exit__(self, *_):
        _active_session.reset(self._token)
        self._token = None

    def lookup(self, key, description):
        identity = digest(key)
        if identity not in self.measurements:
            self.measurements[identity] = dict(key=key, **description)
        if self.collect:
            return None
        try:
            return self.table.cost(key)
        except MissingMeasurement:
            return None

    def record(self, report):
        index = len(self.reports)
        self.reports.append(report)
        path = self.output_directory / f"region-{index:05d}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _layout_identity(layout):
    return "fixed" if layout is None else hashlib.sha256(str(layout).encode()).hexdigest()


class _Region:
    def __init__(self, session, statements, function, layouts, pins, annotations,
                 thread_bounds, thread_index, bindings, in_pipeline, divergent):
        import tvm_ffi
        from tilelang import tvm

        self.tvm = tvm
        self.session, self.statements, self.function = session, list(statements), function
        self.layouts, self.pins, self.annotations = dict(layouts), dict(pins), annotations
        self.thread_bounds, self.thread_index, self.bindings = thread_bounds, thread_index, bindings
        self.threads, self.in_pipeline, self.divergent = int(thread_bounds.extent), bool(in_pipeline), bool(divergent)
        self.target = function.attrs["target"]
        self.access = tvm_ffi.get_global_func("tl.layout.graph_accesses")
        self.infer = tvm_ffi.get_global_func("tl.layout.graph_infer")
        self.remap = tvm_ffi.get_global_func("tl.layout.graph_remap")
        self.convert = tvm_ffi.get_global_func("tl.layout.make_conversion")
        self.conversion_kind = tvm_ffi.get_global_func("tl.layout.conversion_kind")
        self.accesses = [self.access(stmt, annotations, bindings) for stmt in statements]
        self.buffers = []
        for access in self.accesses:
            if not access["supported"]:
                raise ValueError("native graph region contains an unsupported operator")
            for buffer in [*access["reads"], *access["writes"]]:
                if buffer not in self.buffers:
                    self.buffers.append(buffer)
        # All known aliases participate, including views used outside this region.
        families = {}
        for buffer in dict.fromkeys([*self.layouts, *self.buffers]):
            families.setdefault(buffer.data, []).append(buffer)
        self.family = {b: members[0] for members in families.values() for b in members}
        self.aliases = {members[0]: members for members in families.values()}
        asynchronous = set()
        for statement, access in zip(self.statements, self.accesses):
            if isinstance(statement, tvm.tirx.Evaluate) and isinstance(statement.value, tvm.tirx.Call):
                name = getattr(statement.value.op, "name", "")
                if name in ("tl.tileop.wgmma_gemm", "tl.tileop.tcgen05_gemm"):
                    asynchronous.update(self.family[b] for b in [*access["reads"], *access["writes"]])
        self.domains, self.fixed_reasons = {}, {}
        for buffer in self.buffers:
            family = self.family[buffer]
            if family in self.domains:
                continue
            layout = self.layouts.get(family)
            self.domains[family] = [layout]
            reason = None
            if len(self.aliases[family]) > 1:
                reason = "aliased storage"
            elif family.scope() != "local.fragment" or isinstance(layout, PartialFragment):
                reason = "native storage or reducer state"
            elif family in self.pins:
                reason = "explicit layout annotation"
            elif self.divergent:
                reason = "thread-dependent or multidimensional thread scope"
            elif family in asynchronous:
                reason = "explicit asynchronous operation lifetime"
            if reason:
                self.fixed_reasons[family] = reason
                for alias in self.aliases[family]:
                    if alias in self.layouts:
                        self.pins[alias] = self.layouts[alias]
        self.configurations = []
        self.rejections = []
        self.attempts = 0
        self.schedule = dict(target=str(self.target), threads=self.threads,
                             thread_min=str(thread_bounds.min), in_pipeline=self.in_pipeline,
                             divergent_scope=self.divergent,
                             compiler_flags=list(_compile_flags.get()),
                             pass_configs={str(k): str(v) for k, v in tvm.transform.PassContext.current().config.items()
                                           if str(k) not in ("tl.layout_solver", "tl.layout_solver_verbose",
                                                             "tl.layout_solver_timeout_ms")})

    def layout_id(self, buffer, layout):
        family = self.family[buffer]
        if family in self.fixed_reasons:
            expected = self.layouts.get(buffer)
            if expected is not None and not expected.is_equal(layout):
                raise ValueError("candidate modifies fixed storage layout")
            return 0
        domain = self.domains[family]
        for index, known in enumerate(domain):
            if known.is_equal(layout):
                return index
        domain.append(layout)
        return len(domain) - 1

    def candidates(self):
        tir = self.tvm.tirx
        generic = {}
        for family in self.domains:
            if family not in self.fixed_reasons:
                shape = [int(n) for n in family.shape]
                specs = domain_specs(shape, self.threads, self.tvm.DataType(family.dtype).bits,
                                     self.session.max_candidates)
                generic[family] = [make_fragment(shape, spec) for spec in specs]
        for op_index, (stmt, access) in enumerate(zip(self.statements, self.accesses)):
            operands = list(dict.fromkeys([*access["reads"], *access["writes"]]))
            root = {b: self.layouts[b] for b in operands if b in self.layouts}
            seeds = [("root", stmt, root)]
            unbound = stmt
            pinned_loop = isinstance(stmt, tir.For) and bool(stmt.annotations.get("tl.graph_pinned_loop", False))
            if isinstance(stmt, tir.For) and not pinned_loop:
                attrs = dict(stmt.annotations)
                for key in ("parallel_loop_layout", "parallel_loop_predicate", "parallel_loop_requires_padding_guard"):
                    attrs.pop(key, None)
                unbound = tir.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind, stmt.body,
                                  thread_binding=stmt.thread_binding, annotations=attrs,
                                  step=stmt.step)
            if not self.divergent:
                seeds.append(("native", unbound, {}))
                for buffer in operands:
                    family = self.family[buffer]
                    for layout in generic.get(family, []):
                        seeds.append(("generic", unbound, {buffer: layout}))
            rows, signatures = [], set()
            for origin, candidate_stmt, seed in seeds:
                self.attempts += 1
                try:
                    result = self.infer(candidate_stmt, self.annotations, self.target, self.thread_bounds,
                                        self.thread_index, seed, self.pins, self.bindings, self.in_pipeline)
                    if not result["valid"]:
                        raise ValueError(str(result["reason"]))
                    candidate_layouts = dict(result["layouts"])
                    ids = {}
                    for buffer in operands:
                        if buffer in self.layouts:
                            if buffer not in candidate_layouts:
                                raise ValueError(f"candidate omitted layout for {buffer.name}")
                            ids[buffer] = self.layout_id(buffer, candidate_layouts[buffer])
                        else:
                            ids[buffer] = 0
                    signature = (tuple(ids[b] for b in operands), self.tvm.ir.save_json(result["statement"]))
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    rows.append(dict(ids=ids, statement=result["statement"], layouts=candidate_layouts,
                                     origin=origin))
                    if len(rows) > self.session.max_candidates:
                        raise DomainBudgetExceeded("operator candidate budget exceeded")
                except Exception as exc:
                    if origin == "root" or isinstance(exc, DomainBudgetExceeded):
                        raise
                    self.rejections.append(dict(operator=op_index, origin=origin, reason=str(exc)))
            if not rows:
                raise ValueError("operator has no validated configuration")
            self.configurations.append(rows)

    def key(self, kind, buffers, layouts, statement):
        ir = self.tvm.ir.save_json(statement)
        return measurement_key(kind, operator_ir=hashlib.sha256(ir.encode()).hexdigest(),
                               dtype=",".join(str(b.dtype) for b in buffers) or "int32",
                               shape=[int(n) for b in buffers for n in b.shape] or [1],
                               layouts=layouts or ["fixed"], schedule=self.schedule)

    def measure(self, key, **description):
        return self.session.lookup(key, dict(function=self.function, thread_bounds=self.thread_bounds,
                                            thread_index=self.thread_index, bindings=self.bindings,
                                            annotations=self.annotations,
                                            pass_configs=dict(self.tvm.transform.PassContext.current().config),
                                            **description))

    def build(self):
        tensors, operators = [], []
        current, written = {}, set()
        self.values, self.operation_nodes = [], []
        conversion_tables, conversion_keys = {}, {}
        for family, layouts in self.domains.items():
            matrix, keys = [], []
            for source_id, source in enumerate(layouts):
                row, key_row = [], []
                for target_id, target in enumerate(layouts):
                    if source_id == target_id:
                        row.append(0)
                        key_row.append(None)
                        continue
                    try:
                        kind = self.conversion_kind(source, target, self.threads)
                    except Exception as exc:
                        self.rejections.append(dict(buffer=str(family.name), conversion=[source_id, target_id], reason=str(exc)))
                        row.append(None)
                        key_row.append(None)
                        continue
                    # The direction and both layouts are part of the identity.
                    statement = self.tvm.tirx.Evaluate(0)
                    key = self.key("conversion", [family], [_layout_identity(source), _layout_identity(target)], statement)
                    key_row.append(digest(key))
                    row.append(self.measure(key, kind="conversion", buffer=family, source=source, target=target,
                                            communication=int(kind)))
                matrix.append(row)
                keys.append(key_row)
            conversion_tables[family] = matrix
            conversion_keys[family] = keys

        def value(family):
            index = len(tensors)
            tensors.append(dict(name=f"{family.name}@{index}", layouts=[_layout_identity(l) for l in self.domains[family]],
                                conversion_costs=conversion_tables[family], conversion_keys=conversion_keys[family]))
            self.values.append(family)
            return index

        for index, (access, configurations) in enumerate(zip(self.accesses, self.configurations)):
            reads, writes = list(access["reads"]), list(dict.fromkeys(self.family[b] for b in access["writes"]))
            for buffer in reads:
                family = self.family[buffer]
                if family not in current:
                    current[family] = value(family)
                    operators.append(dict(name=f"input:{family.name}", inputs=[], outputs=[current[family]],
                                          configs=[dict(inputs=[], outputs=[0], cost=0)]))
            inputs = [current[self.family[b]] for b in reads]
            outputs = [value(family) for family in writes]
            rows = []
            for candidate in configurations:
                ids = candidate["ids"]
                input_layouts = [ids[b] for b in reads]
                output_layouts = [next(ids[b] for b in access["writes"] if self.family[b] == family) for family in writes]
                operands = [*reads, *access["writes"]]
                layout_keys = [_layout_identity(candidate["layouts"].get(b)) for b in operands]
                key = self.key("operator", operands, layout_keys, candidate["statement"])
                cost = self.measure(key, kind="operator", statement=candidate["statement"], layouts=candidate["layouts"],
                                    reads=reads, writes=list(access["writes"]))
                rows.append(dict(inputs=input_layouts, outputs=output_layouts, cost=cost, cost_key=digest(key)))
            self.operation_nodes.append(len(operators))
            operators.append(dict(name=f"operator:{index}:{access['kind']}", inputs=inputs, outputs=outputs, configs=rows))
            current.update(zip(writes, outputs))
            written.update(writes)
        self.sinks = [current[family] for family in current if family in written]
        if self.sinks:
            operators.append(dict(name="region:exit", inputs=self.sinks, outputs=[],
                                  configs=[dict(inputs=[0] * len(self.sinks), outputs=[], cost=0)]))
        self.problem = dict(tensors=tensors, operators=operators)
        return self.problem

    def apply(self, choices):
        tir = self.tvm.tirx
        storage, available, sources = {}, {}, {}
        allocations, new_layouts, output = [], {}, []
        serial = len(self.session.reports)

        def physical(family, layout_id):
            if layout_id == 0:
                return family
            key = (family, layout_id)
            if key not in storage:
                buffer = tir.decl_buffer(family.shape, family.dtype,
                                         name=f"{family.name}_layout_{serial}_{layout_id}", scope="local.fragment")
                storage[key] = buffer
                allocations.append(buffer)
                new_layouts[buffer] = self.domains[family][layout_id]
            return storage[key]

        def require(tensor, layout_id):
            family = self.values[tensor]
            if (tensor, layout_id) not in available:
                source_layout = sources.get(tensor, 0)
                if source_layout != layout_id:
                    output.append(self.convert(physical(family, source_layout), physical(family, layout_id)))
                available[tensor, layout_id] = physical(family, layout_id)
            return available[tensor, layout_id]

        for op_index, node_index in enumerate(self.operation_nodes):
            node = self.problem["operators"][node_index]
            choice = choices[node_index]
            row = node["configs"][choice]
            candidate = self.configurations[op_index][choice]
            for tensor, layout_id in zip(node["inputs"], row["inputs"]):
                require(tensor, layout_id)
            remapping = {}
            for buffer, layout_id in candidate["ids"].items():
                if layout_id:
                    remapping[buffer] = physical(self.family[buffer], layout_id)
            output.append(self.remap(candidate["statement"], remapping))
            for tensor, layout_id in zip(node["outputs"], row["outputs"]):
                sources[tensor] = layout_id
                available[tensor, layout_id] = physical(self.values[tensor], layout_id)
        for tensor in self.sinks:
            require(tensor, 0)
        return dict(statement=tir.stmt_seq(*output), allocations=allocations, layouts=new_layouts)


def select_region(*arguments):
    session = _active_session.get()
    if session is None:
        raise RuntimeError("full layout strategies require an active GraphLayoutSession")
    started = time.monotonic()
    region = _Region(session, *arguments)
    region.candidates()
    problem = region.build()
    missing = [key for key, value in session.measurements.items()
               if session.collect or _missing(session.table, value["key"])]
    report = dict(mode="calibration" if session.collect else "selection", problem=problem,
                  candidate_attempts=region.attempts, candidates=[len(c) for c in region.configurations],
                  rejections=region.rejections,
                  pinned_buffers={str(b.name): reason for b, reason in region.fixed_reasons.items()},
                  missing_measurements=missing, extraction_ms=(time.monotonic() - started) * 1000,
                  table_sha256=None if session.collect else session.table.sha256)
    if session.collect:
        session.record(report)
        return dict(statement=region.tvm.tirx.stmt_seq(*region.statements), allocations=[], layouts={})
    if missing:
        session.record(report)
        raise MissingMeasurement(f"frozen calibration lacks {len(missing)} requested measurements")
    algorithm = str(region.tvm.transform.PassContext.current().config["tl.layout_solver"])
    result = solve(problem, algorithm, session.timeout_ms, session.max_states)
    report["result"] = result
    report["root"] = evaluate(problem, [0] * len(problem["operators"]))
    session.record(report)
    if result["status"] not in ("optimal", "feasible"):
        raise RuntimeError(f"full layout selection failed explicitly: {result}")
    return region.apply(result["choices"])


def _missing(table, key):
    try:
        table.cost(key)
        return False
    except MissingMeasurement:
        return True


def register_selection():
    import tvm_ffi
    tvm_ffi.register_global_func("tl.layout.select_graph_region_v1", select_region)
