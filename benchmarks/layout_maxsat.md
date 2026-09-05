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
TILELANG_DISABLE_CACHE=1 TILELANG_CACHE_DIR=/content/layout-cache-baseline \
  python layout_maxsat.py --variant baseline --output /content/layout-baseline
TILELANG_DISABLE_CACHE=1 TILELANG_CACHE_DIR=/content/layout-cache-candidate \
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
The launcher receives the current stream explicitly. Each graph must contain
100 nodes and reproduce the reference after its outputs are filled with NaNs.
This guards against empty captures: the pinned upstream NVRTC adapter tests
whether the target string starts with `cuda`, while this TVM revision renders
the target as JSON. Automatic stream selection therefore used stream 0 during
the initial capture attempt. Those initial timing samples are invalid.
Raw per-launch microsecond samples, median, minimum and maximum are retained.
This timing measures repeated execution with warm caches. Compilation includes
initialization costs; trace phase times separately describe solver overhead.
Failures remain in the report. Source equality between corresponding baseline
and candidate root controls must be checked before attributing timing changes.

## T4 results (2026-09-06)

The results in this section concern constructed layout components. A subsequent
[real-workload evaluation](layout_maxsat_real.md) tested 28 upstream operator
and shared-expert configurations: 23 passed all numerical/graph checks, and
MaxSAT adopted no new layouts. The positive constructions below therefore
provide a limited mechanism demonstration; general model-workload gains remain
unestablished.

[Workflow 33992154463](https://github.com/nya-a-cat/tilelang/actions/runs/33992154463)
built baseline `1a1df62e94d9736a2c9f13849bc4c2cf9377275b` and compiler candidate
`2a8c5e5b56b13953c238b6b4b8ebbb19cf6c7676`. The final measurement script is
`415bda7b1fba89fbdb3af81e9777eecdd0e4993e`. Wheels and build provenance are in
the [fork prerelease](https://github.com/nya-a-cat/tilelang/releases/tag/layout-maxsat-2a8c5e5b56b1).
The Colab environment used Tesla T4, driver 580.82.07, Python 3.13.15 and
PyTorch 2.11.0+cu128. Compilation caching was disabled for the final run.

Baseline passed 38/40 mode configurations; candidate passed 76/80. All 114
successful records passed direct and graph-replay numerical checks, with 100
graph nodes and 15 timing samples each. Their saved source and CUBIN hashes
were independently checked. The 38 successful baseline/root controls had
identical source and CUBIN in the candidate wheel. The reduction shape
`(2,2560)` with 256 threads failed with `no available layout found` under both
baseline models and all four candidate modes; these failures were retained.

MaxSAT solved 32 eligible tables to optimality and retained root selection in
six ineligible reduction cases. Three io-aware connected-branch configurations
produced strictly improved costs and changed source/CUBIN:

| Branch shape (m,n,threads) | Root median us | MaxSAT median us | Speedup | CUDA registers root / MaxSAT |
|---|---:|---:|---:|---:|
| (8,512,128) | 35.194 | 21.865 | 1.610x | 46 / 48 |
| (32,128,128) | 32.276 | 21.402 | 1.508x | 52 / 48 |
| (64,64,256) | 24.844 | 20.068 | 1.238x | 62 / 42 |

An earlier graph-validated round also favored all three compositions, with
speedups of 1.286x, 1.357x and 1.597x respectively. Ratios vary across rounds;
these measurements establish benefits for these constructions on this T4.
The other 35 successful root/MaxSAT comparisons had identical source and CUBIN.
Timing variation for those binaries does not establish an optimization benefit.
The default register-count model selected no improved composition in this suite.

For the three changed configurations, memory proxy costs decreased from
217088/165888/159744 to 102400/65536/57344. Register-slot proxy costs changed
from 13/10/6 to 17/17/9; these proxy counts differ from physical CUDA registers.
Across 32 eligible solves, median extraction, solver and validation times were
3.920, 9.204 and 3.112 ms. These phases exclude initial root inference and
do not capture every source of total compilation overhead.

The initial empty-graph timing run is invalid and was preserved with that
label. The final report, all raw samples, generated source/CUBIN, failures and
both valid measurement rounds were saved locally. Colab was terminated and
the server reported no active sessions after delivery.
