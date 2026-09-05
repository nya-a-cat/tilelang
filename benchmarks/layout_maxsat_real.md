# Real-workload layout selection measurements

The bounded MaxSAT integration adopted no new layouts in 28 upstream workload
configurations tested on T4. All 50 available root/MaxSAT CUBIN pairs were
identical, including 46 numerically valid comparisons. These results support
keeping the feature opt-in. No runtime improvement is established by this suite.

## Setup

Date: 2026-09-06. Baseline compiler:
`1a1df62e94d9736a2c9f13849bc4c2cf9377275b`; candidate compiler:
`2a8c5e5b56b13953c238b6b4b8ebbb19cf6c7676`. Both wheels were built by
[Actions run 33992154463](https://github.com/nya-a-cat/tilelang/actions/runs/33992154463)
and are available in the
[fork release](https://github.com/nya-a-cat/tilelang/releases/tag/layout-maxsat-2a8c5e5b56b1).
The experiment reused the existing compiler wheels without algorithm changes.

Environment: Tesla T4 (sm_75), driver 580.82.07, CUDA 12.8.93,
Python 3.13.15, Torch 2.11.0+cu128 and Z3 4.15.4.0. Compilation caching and
TF32 were disabled. Source file hashes match the pinned upstream examples.

[The harness](layout_maxsat_real.py) imports the original kernel bodies and
preserves their pass configurations. Autotuned wrappers use fixed schedules.
Each case has a fresh process and deterministic inputs. Timing uses preallocated
outputs, explicit launch streams, and 30 shuffled interleaved CUDA graph replay
samples, each containing 20 invocations. Direct and NaN-poisoned replay outputs
must match Torch references, with every element passing atol=rtol=0.01 for FP16
or 0.0003 for FP32. Graphs must have 20 nodes per kernel; the shared-expert block
contains two kernels and therefore 40 nodes. Timing describes warm GPU execution.

Baseline runs register-count/root and io-aware/root. Candidate also runs MaxSAT
under each model. The primary harness was `5a252998`; three Softmax loading
failures were superseded by reruns with harness `bade5b76`, which supplies the
upstream module's M/N globals while preserving the kernel AST. The latter
harness also provides optional randomized repeated compilation.

## Results

| Outcome | Baseline modes | Candidate modes |
|---|---:|---:|
| Passed direct and graph checks | 46 | 92 |
| Numerical check failed | 4 | 8 |
| NVRTC compilation failed | 6 | 12 |
| Total | 56 | 112 |

Twenty-three cases passed every mode. Two GEMV reducer cases produced matching
binaries with numerical failures; three online Softmax cases failed compilation.
All 50 available baseline/candidate root CUBIN controls also matched.
Every saved CUDA/CUBIN hash was independently checked (300 files).

The table shows candidate io-aware medians in microseconds. Every row with
timings has identical root and MaxSAT CUBIN, so timing differences provide no
evidence of layout improvement. Shape order follows the case name; attention
uses B,H,S,D and shared MLP uses tokens,hidden,expert.

| Case | Root us | MaxSAT us |
|---|---:|---:|
| add-128x4096 FP16 | 10.074 | 10.085 |
| add-1024x4096 FP32 | 217.562 | 217.782 |
| add-4096x4096 FP16 | 438.978 | 438.906 |
| rmsnorm-1x4096 | 5.223 | 5.291 |
| rmsnorm-128x4096 | 12.698 | 13.088 |
| rmsnorm-1024x4096 | 145.764 | 145.685 |
| rmsnorm-128x8192 | 36.835 | 37.010 |
| rmsnorm_splitk-128x8192 | 47.253 | 47.402 |
| rmsnorm_splitk-1024x8192 | 411.468 | 411.803 |
| layernorm-128x768 | 10.495 | 10.557 |
| layernorm-1024x1024 | 41.730 | 41.854 |
| layernorm-128x4096 | 18.030 | 18.066 |
| softmax-128x4096 | compile failure | compile failure |
| softmax-1024x8192 | compile failure | compile failure |
| softmax-128x32768 | compile failure | compile failure |
| gemm-1024x1024x1024 | 160.789 | 161.019 |
| gemm-128x4096x4096 | 324.794 | 324.692 |
| gemm-1024x2048x2048 | 638.975 | 634.682 |
| gemv-4096x4096 | 129.118 | 129.001 |
| gemv-11008x4096 | 336.586 | 336.799 |
| gemv_reducer-4096x4096 | numerical failure | numerical failure |
| gemv_reducer-11008x4096 | numerical failure | numerical failure |
| attention-1x8x512x64-noncausal | 100.149 | 100.122 |
| attention-1x8x2048x64-causal | 419.654 | 419.690 |
| attention-2x8x1024x64-causal | 236.237 | 236.339 |
| attention-1x8x512x128-causal | 101.462 | 101.385 |
| shared_mlp-128x1024x2816 | 462.751 | 462.326 |
| shared_mlp-128x4096x11008 | 5948.423 | 5945.586 |

The original shared-expert block includes gate/up GEMMs, SiLU, multiplication
and down GEMM. This suite measures operators and that complete block. Full-model
serving, training/backward, distributed workloads, other GPU architectures and
BF16 remain unmeasured.

## Coverage and failure analysis

The candidate emitted 234 component traces: six optimal ties in elementwise add
and 228 ineligible components. Zero components were adopted. These counts
exclude repeated-compilation diagnostics and differ from workload counts.

Current candidate collection is limited to components with at most 16 operators.
Composition requires multiple successful root attempts, Parallel/Copy operators,
closed layouts, no aliases, at most 64 mapped buffers and at most 256 candidate
rows. GEMM/reducer-containing components are unsupported. Existing constraints
can fix layouts before free selection. The solver composes layouts already
found by root inference and introduces no conversions. These limits explain the
narrow opportunity set; per-component eligibility traces do not distinguish
every rejection reason.

The earlier positive branch constructions used a fixed replicated scalar anchor
to connect branches. Their 1.238-1.610x final-round gains apply to those
constructions. This evaluation supplies no evidence of broader workload gains.
It evaluates this bounded integration; the paper's complete method remains
unreproduced here.

GEMV reducer failed the strict FP32-dot-reference check at 9/4096 and 26/11008
elements (maximum reported absolute discrepancies 0.0363159 and 0.0374146).
All baseline/candidate modes had the same failures and CUBIN. The upstream
profiler permits a 1% mismatched-element fraction by default; this harness
permits zero. No tolerance was relaxed, and these cases have no accepted timing.

After fixing module loading, all three FP16 online Softmax cases failed NVRTC
compilation in both versions. Verbose recompilation of the 128x4096 generated
CUDA identified missing `std::numeric_limits` and subsequent type/infinity
errors at `std::numeric_limits<half_t>::infinity()`. The compiler and upstream
kernel were left unchanged. One LayerNorm comparison changed temporary
workspace names in generated source while preserving identical CUBIN.

## Compilation cost

Four representative cases initialized all four candidate modes, then performed
three randomized blocks of fresh compilations per mode (48 repeated builds).
Each repeated CUBIN matched its initial mode. Full compilation medians and
observed ranges are in seconds. Three repeats and visible CPU variance limit
interpretation of incremental overhead.

| Case | Cost model | Root median [min,max] s | MaxSAT median [min,max] s |
|---|---|---:|---:|
| add-128x4096 | register-count | 0.318 [0.306,0.322] | 0.338 [0.312,0.342] |
| add-128x4096 | io-aware | 0.327 [0.319,0.346] | 0.326 [0.326,0.344] |
| rmsnorm-128x4096 | register-count | 0.369 [0.350,0.394] | 0.359 [0.355,0.365] |
| rmsnorm-128x4096 | io-aware | 0.381 [0.366,0.386] | 0.374 [0.349,0.391] |
| attention-1x8x512x64-noncausal | register-count | 4.468 [3.716,4.796] | 3.724 [3.700,3.740] |
| attention-1x8x512x64-noncausal | io-aware | 3.793 [3.736,4.763] | 3.914 [3.892,4.751] |
| shared_mlp-128x1024x2816 | register-count | 6.604 [6.588,6.829] | 7.095 [6.880,7.898] |
| shared_mlp-128x1024x2816 | io-aware | 7.637 [7.328,7.719] | 7.661 [6.744,7.742] |

For six eligible primary add solves, median extraction/solver/validation times
were 1.512/5.395/2.111 ms. These phases omit existing root inference and other
compilation work. Initial one-off compile-order effects are excluded above.

## Reproduction

Install the matching compiler wheel and use the pinned upstream examples:

```sh
TILELANG_DISABLE_CACHE=1 python benchmarks/layout_maxsat_real.py \
  --variant candidate --examples-root /path/to/pinned/tilelang \
  --output results-candidate
```

Run `--variant baseline` in the baseline environment. Add
`--case add-128x4096 --compile-repeats 3` for the compilation diagnostic.
The harness records concrete schedules, shapes, output indices, script/source
hashes, errors, CUDA/CUBIN, resources and all samples. Inspect per-mode errors;
a completed case process can contain failed modes.

The complete local evidence archive contains 503 manifest-verified files and
has SHA-256 `6f5c2ad8481f9af16bcfdc4f229965b881c53a1a918253a84fcc7db6a788814b`.
Raw evidence remains an internal local output. The GPU session was terminated;
the Colab server reported no active sessions after download verification.
