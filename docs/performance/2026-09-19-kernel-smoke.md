# Physics-kernel smoke results — 2026-09-19

Keep eager execution as the default. The custom Triton loss did not meaningfully
accelerate the usual ten-condition workload. Compiling the complete physics RHS
and loss improved steady training time in the smaller case, with substantial
compilation overhead.

## Matched protocol

RTX PRO 6000 Blackwell, PyTorch 2.12.0+cu130, Triton 3.7.0, float64, four CPU
threads. Normal regime, MYC hybrid, 4-by-256 state networks, T=150, 150 sparse
observations per condition and 1,000 reference points. Each run used 60 Adam
epochs, two L-BFGS calls and one refinement call. Two repetitions ran in reversed
backend order, using the same observations and initialization. Steady timings
cover Adam epochs 11–60 and include adaptive updates and checkpoint writes.

| Conditions / points | Backend | Steady ms/epoch | Speedup | First training run, s | Cached repeat, s |
|---|---|---:|---:|---:|---:|
| 2 / 1,024 | eager | 28.01 | 1.000x | 4.20 | 3.84 |
| 2 / 1,024 | triton | 26.53 | 1.056x | 5.53 | 4.40 |
| 2 / 1,024 | compiled | 19.43 | 1.442x | 62.46 | 10.77 |
| 10 / 8,000 | eager | 141.56 | 1.000x | 14.35 | 14.50 |
| 10 / 8,000 | triton | 141.15 | 1.003x | 16.37 | 15.31 |

The full-workload Triton difference is only 0.3%, inside observed timing variation.
The small-case Triton median improves by 5.3%, with overlapping ranges. Compiled
physics reduces steady small-case time by 30.6%; its full-workload speed was not
measured. Neither optimized backend reduced total time in these short runs.

## Validation and evidence

75 numerical, model, runner, checkpoint/resume, legacy and CLI checks passed.
After the final compiler-wrapper change, all 28 kernel checks passed again.
Across ten successful runs, maximum relative differences were 1.20e-9 for
recovered parameters, 5.01e-10 for physics loss and 5.85e-9 for checkpoint tensor
norms. Observation hashes, references, scientific settings and source snapshots
match within each workload. All backends made the same number of L-BFGS closure
evaluations within a workload.

[Recorded measurements](2026-09-19-kernel-smoke.json) include configurations,
every run timing, recovered parameters, parity checks and source hashes.
[Reproduction commands](../experiments.md#optional-physics-kernels) use new
timestamped run directories. Original logs and checkpoints remain locally under
`runs/20260919_153746_kernel_comparison/` and the run IDs in the JSON record.

The GPU was shared, and two repetitions are insufficient for a statistical
performance claim. These short runs establish execution and numerical parity;
they do not establish converged parameter recovery or long-run speedup.
