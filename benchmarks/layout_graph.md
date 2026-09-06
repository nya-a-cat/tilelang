# Experimental finite layout graphs

This CUDA experiment applies a finite layout-selection objective to straight-line
TileLang operator regions. `maxsat-full`, `treewidth`, `local`, and `greedy` share
the same candidate configurations, directed conversion table, and objective.
The default `root` strategy and the earlier `maxsat` experiment remain available.

## Objective and implementation

For each operator, choose one jointly valid configuration for all ordered input
and output ports. The objective is the sum of operator costs plus one directed
conversion cost for each distinct `(tensor value, requested layout)` pair that
differs from its producer layout. Repeated consumers share that conversion.
Every write creates a new value version. Partial writes also consume the previous
version, and multiple outputs belong to the same configuration decision.

The weighted MaxSAT solver and the tree-decomposition dynamic program optimize
this finite objective. The latter retains a request set for every live produced
value, unions request sets at joins, and charges conversions when their producer
is forgotten. Both independently recompute the returned assignment's cost.
`optimal` refers to the supplied finite problem. Time and state budgets produce
explicit unsuccessful results when optimality cannot be established.

Candidate generation starts from native inference and independent fragment axis
orders, thread factorizations, and vector widths. Native operator inference
validates complete port configurations, including GEMM and reductions. Explicit
layout annotations, aliased storage, reducer state, asynchronous lifetimes, and
multidimensional or divergent thread scopes constrain the admitted domain.
Rejection reasons and budgets are recorded for each region.

The pass runs after layout inference and before reducer materialization and
TileOp lowering. It preserves statement order, launch geometry, tile sizes, and
pipeline scheduling. Structured control flow separates regions; written fragment
storage returns to its original layout at each boundary. Conversions use register
permutations, warp shuffles, or synchronized shared-memory exchange. Physical
storage is reused across versions only after source-order dependencies permit it.

## Offline calibration

```python
import tilelang
from tilelang.layout import (
    GraphLayoutSession, calibrate, calibration_environment,
)

# `main` is the unchanged source PrimFunc. Initialize the CUDA context first.
with GraphLayoutSession(output_directory="results/collect", collect=True) as collection:
    tilelang.compile(main, target="cuda", execution_backend="nvrtc",
                     pass_configs={"tl.layout_solver": "maxsat-full"})

env = calibration_environment(
    tilelang_revision="<exact compiler build revision>",
    tvm_revision="<exact TVM build revision>",
    compiler_flags=[],
)
table = calibrate(collection, "results/calibration", env)

with GraphLayoutSession(output_directory="results/selected", table=table) as session:
    kernel = tilelang.compile(main, target="cuda", execution_backend="nvrtc",
                              pass_configs={"tl.layout_solver": "treewidth"})
```

Collection records unmeasured costs as `null` and preserves the original region.
Selection requires a frozen table with complete coverage. Missing measurements,
changed compilation flags, solver failures, and compilation failures remain
visible; they do not receive a substitute zero cost or an implicit root plan.
Full strategies bypass the ordinary kernel cache so that every session executes
its collection or selection pass.

Keys retain operator IR, ordered layouts, operand shapes and dtypes, target,
thread geometry, pipeline context, pass configuration, and compilation flags.
The table also identifies GPU, driver, CUDA compiler/runtime, source revisions,
and timing protocol. Each measurement retains CUDA source, CUBIN, their hashes,
raw samples, and compilation time. Resume verifies these files before reusing a
measurement. Calibration freezes the table before complete-kernel evaluation.

The current protocol measures an isolated single-CTA microkernel containing one
native operation plus input/output materialization. Thirty-two invocations are
captured in a CUDA graph; the measured node count must match the actual number
of kernels. Five warmup replays precede fifteen samples. The cost is the positive
integer-nanosecond ceiling of the sample median. Conversion profiles require
exact value preservation, and floating outputs undergo poisoned replay checks.

These are inclusive proxy costs. They include launch and materialization overhead;
enclosing loop indices are fixed at zero. Register pressure, occupancy, spills,
cache state, and pipeline overlap can change the ranking of complete kernels.
Standalone profiling currently rejects reducer epochs, asynchronous wait
contexts, unresolved pointers, and unsupported storage views. Such failures
prevent table freezing and remain in `progress.json`.

## Verification and real workloads

The solver tests compare both exact methods with independent enumeration on 250
random DAGs and cover fanout sharing, joins, repeated ports, multiple outputs,
versions, large costs, concurrency, invalid inputs, and explicit budgets. Native
GPU tests exercise bit-preserving conversion roundtrips and graph rewriting.
Synthetic costs force conversions in semantics tests; those tests provide no
performance measurements for layout optimization.

`layout_graph_real.py` uses the same 28 upstream configurations and independent
references as `layout_maxsat_real.py`, including normalization, GEMM, GEMV,
attention, and the complete shared-expert MLP block. The source kernels and fixed
schedules are retained. Run calibration and evaluation as separate phases:

```bash
python benchmarks/layout_graph_real.py --examples-root /path/to/pinned/upstream \
  --output results/real --compiler-revision EXACT_BUILD_SHA --phase calibrate \
  --calibration-cache results/measurement-cache
python benchmarks/layout_graph_real.py --examples-root /path/to/pinned/upstream \
  --output results/real --compiler-revision EXACT_BUILD_SHA --phase benchmark
```

Each configuration runs in a separate process with a recorded timeout. Evaluation
retains all six strategies, correctness failures, missing tables, generated code,
resource attributes, process peak RSS, extraction/solver/compilation time, and
thirty shuffled timing samples. Preallocated outputs, explicit CUDA streams,
poisoned graph replay, and graph node counts validate the measured execution.
The optional calibration cache reuses measurements only under the same complete
environment and measurement key, after validating source and executable hashes.
Each case receives a separate copy of its evidence and a separately frozen table.

Each benchmark report compares full-kernel timings with the unweighted sum of
region costs. Pairwise rank errors exclude equal proxy costs, identical binaries,
and measured differences within 1% of the faster time. This symmetric band describes timing ties and supplies
no confidence interval. Raw gaps, all ties, speedups and regressions are retained.
Enclosing loop multiplicity is absent from the proxy sum; these diagnostics
measure that proxy's ranking, with its stated limitations.

The finite solver guarantee, correctness of the compiler transformation, and
usefulness of the latency proxy are separate claims. The implementation does not
establish a new treewidth bound for TileLang graphs, whole-kernel latency
optimality, global resource feasibility, or a reproduction of the paper's
Trainium experiments.
