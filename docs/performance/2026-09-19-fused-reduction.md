# Fused final reduction — 2026-09-19

The additional Triton kernel combines partial-sum reduction and normalization.
It removes one GPU launch from the existing custom loss. The matched training
smoke measured 28.00 ms per steady Adam epoch for Triton versus 27.04 ms for
eager, a 3.6% slowdown in this session. Eager remains the default.

## Kernel comparison

RTX PRO 6000 Blackwell, PyTorch 2.12.0+cu130, Triton 3.7.0, float64, four CPU
threads, deterministic algorithms and deterministic memory initialization enabled.
The benchmark reconstructs the previous final sum/division while sharing the
same residual and backward kernels. Six repeats alternate implementation order;
compilation and warmup are excluded.

| Collocation points | Reduction GPU µs, previous → fused | Loss forward GPU µs, previous → fused | Loss forward/backward host µs, previous → fused |
|---:|---:|---:|---:|
| 1,024 | 2.89 → 1.35 | 5.46 → 3.85 | 382.25 → 392.16 |
| 8,000 | 3.40 → 1.54 | 6.73 → 4.93 | 385.89 → 377.36 |
| 40,001 | 3.92 → 2.34 | 10.35 → 8.73 | 471.36 → 465.32 |

At 8,000 points, the isolated reduction is 2.21x faster and the loss forward is
1.36x faster on the GPU. Profiling confirms four forward kernels become three:
one temporary-buffer initialization plus two calculation kernels. Reusing the
first partial-sum element for the scalar result avoids another initialization.
The reduction uses bounded 1,024-element tiles, with no atomics or dtype casts.

GPU times use 64 CUDA graph replays of 32 calls per sample. The isolated
reduction uses preallocated outputs on both sides. Complete-loss measurements
include normal allocations. Host times synchronize batches of 64 calls and
include Python dispatch and autograd. Host improvements are small and
inconsistent; the trainers do not use CUDA graphs.

## Matched training and validation

Two repetitions per backend ran in reversed order through the versioned runner.
Both use `configs/kernel_smoke.json`: Normal MYC hybrid, two conditions, 1,024
collocation points, 4-by-256 state networks, 60 Adam epochs, two L-BFGS calls and
one refinement call. Observations, initializations, references and scientific
settings match. Steady timing covers Adam epochs 11–60.

| Backend | Steady Adam ms/epoch, median | Repeat values, ms/epoch | Training seconds, first / cached repeat |
|---|---:|---|---|
| eager | 27.04 | 26.72, 27.37 | 4.00 / 3.82 |
| triton | 28.00 | 28.24, 27.77 | 5.93 / 4.61 |

All 51 kernel checks passed, including both dtypes, nonuniform grids,
in-place reduction, more than 1,024 partials, tiny float64 values, first/second
derivatives and biological/neural gradients. Lint passed. All four training
runs completed with identical recovered parameters and checkpoint tensors;
the maximum relative physics-loss difference was 2.03e-16. Each used 43 L-BFGS
closures and 23 refinement closures. Source snapshots and reference arrays
were verified.

The GPU was shared. Two training repeats do not establish a general performance
regression or long-run speedup, and this short session does not establish
converged inverse recovery. The training comparison is eager versus the final
Triton backend; the isolated benchmark compares the previous and new reductions.

[Raw timings, configurations and validation](2026-09-19-fused-reduction.json)
retain all samples and source hashes. [Reproduction commands](../experiments.md#optional-physics-kernels)
cover both benchmarks. Local logs and checkpoints remain below
`runs/20260919_232844_fused_reduction/` and the training run IDs in the JSON.
