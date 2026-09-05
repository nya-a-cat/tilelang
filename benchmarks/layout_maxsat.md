# Experimental layout candidate composition

Reference: Eisenhofer et al., [Tensor Seeks Layout: Formalizing Layout Selection
for ML Compilers](https://arxiv.org/abs/2608.21555v1), Section 3.2.2.

This experiment starts from upstream commit
`1a1df62e94d9736a2c9f13849bc4c2cf9377275b`. The default layout selection
remains `root`, with upstream's default `register-count` cost model.

## Implementation

`tl.layout_solver="maxsat"` retains successful free-inference root attempts,
projects each attempt onto each operator and its accessed fragment buffers,
and solves a finite candidate table. Each operator selects one snapshot. All
operators accessing the same buffer select the same layout for that buffer.
Weighted soft constraints minimize the existing memory score, followed by
the total register-slot score. Each buffer's register slots are charged once.
Costs cross the compiler/solver boundary as 64-bit integers. Both optimization
objectives must have matching lower and upper bounds at the returned cost.

The selected composition is re-inferred against the complete layout map and
strict annotations. Its memory and register scores must match a fresh full
score. Adoption requires a strict improvement over the best original root
attempt. Ties preserve the original selection. Solver failures, timeouts,
inconsistent compositions and unsupported components preserve that selection.

The compiler currently collects candidates for components of at most 16
operators, bounding the table to 256 rows. Composition supports Parallel and
Copy operators, at most 64 mapped buffers, closed layouts, and no buffer aliases.
Reducer operations retain root selection. Every solver call owns a private Z3
context. `tl.layout_solver_timeout_ms` defaults to 100 and applies to Z3's
optimization check; candidate generation and validation have separate costs.
Verbose traces include those phase times and original/selected scores:

```python
kernel = tilelang.compile(
    func,
    pass_configs={
        "tl.layout_cost_model": "io-aware",
        "tl.layout_solver": "maxsat",
        "tl.layout_solver_timeout_ms": 1000,
        "tl.layout_solver_verbose": True,
    },
)
```

This is a conversion-free adaptation of global finite-domain layout selection.
The candidate pool comes from TileLang's existing inference. It does not add
layout-conversion operators, the paper's target-specific cost tables, or unseen
layout configurations. An optimal solver result applies to that finite table.
Measured outcomes therefore describe this integration and its candidate pool.

## Validation and measurement

The standalone solver tests compare 100 deterministic random tables against
exhaustive search and cover lexicographic priorities, shared buffer charging,
independent composition, zero objectives, budget rejection and concurrent calls.

```sh
python -m pytest --confcutdir=testing/python/layout \
  testing/python/layout/test_tilelang_layout_solver.py -q
```

The fork workflow builds the pinned upstream baseline and experimental source
with the same Python, CUDA and dependency environment. It records both source
revisions, all submodules, dependency versions and wheel SHA-256 hashes.

`layout_maxsat.py` runs 20 configurations from five families: broadcast,
fused elementwise, connected branches, transpose, and reduction. Shapes and
thread counts are fixed in the script, with 128 blocks per case. The connected
branch construction shares a pinned replicated scalar fragment across both
branches. Reduction cases exercise the current eligibility restriction.

Run each installed wheel in a fresh process with a distinct cache directory:

```sh
TILELANG_CACHE_DIR=/content/layout-cache-baseline \
  python layout_maxsat.py --variant baseline --output /content/layout-baseline
TILELANG_CACHE_DIR=/content/layout-cache-candidate \
  python layout_maxsat.py --variant candidate --output /content/layout-candidate
```

The baseline includes register-count/root and io-aware/root. The candidate
includes those controls plus register-count/maxsat and io-aware/maxsat.
All outputs are checked against PyTorch. Each mode records compilation time,
generated CUDA and its hash. CUDA Driver API queries record register counts,
local and shared memory sizes; CUBIN files and hashes are retained when these
queries succeed. Query failures are recorded independently of numerical checks.
Successful kernels are captured as 100 launches in
a CUDA graph; 15 replay measurements are interleaved across modes within a case.
Raw per-launch microsecond samples, median, minimum and maximum are retained.
This timing measures repeated execution with warm caches. Compilation includes
initialization costs; trace phase times separately describe solver overhead.
Failures remain in the report. Source equality between corresponding baseline
and candidate root controls must be checked before attributing timing changes.

GPU measurements are pending. No runtime speedup is established by the solver
tests or compilation alone.
