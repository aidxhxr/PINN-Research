# Running experiments

The versioned runner executes the active integral inverse PINN and hybrid
trainers. It preserves the existing training stages and writes new runs below
`runs/`. Historical scripts still use their original defaults. The untouched
`PINN-inverse-solve/` baseline is outside this runner.

The runner requires a POSIX system with tmux and process signals.

Run jobs in tmux and record the session, purpose, output/log path, date and
status in `AGENTS.md`. Start with the CPU execution check:

```bash
tmux new-session -s research-smoke
python experiments/run.py configs/smoke_cpu.json
```

The returned directory contains `config.json`, `manifest.json`, per-regime logs,
reference arrays, every-start metrics, checkpoints and the selected models.
The smoke configuration has two conditions, a small network and six Adam
epochs followed by one L-BFGS step. It checks execution, not model accuracy.

## Configurations and resources

Configurations are JSON. Missing settings receive documented defaults from
`src/wnt_pinn/runs/config.py`; unknown keys and invalid values fail before a
worker starts. The resolved configuration is saved with each run.

| Profile | Purpose |
| --- | --- |
| `configs/smoke_cpu.json` | Small CPU execution check |
| `configs/integral.json` | Four regimes, ten conditions, integral inverse PINN |
| `configs/hybrid_myc.json` | Same protocol with learned beta-catenin to MYC activation |
| `configs/kernel_smoke.json` | GPU performance smoke check: optional Triton integral loss, MYC hybrid, two conditions |
| `configs/kernel_smoke_full.json` | Same short GPU training with all ten conditions and 8,000 collocation points |

`resources.device` accepts `cpu`, `auto`, `cuda` or `cuda:N`. `threads` limits
threads per worker; `concurrency` limits simultaneous regime workers. GPU
workers share the selected device. GPU memory determines a safe concurrency.
References are generated serially within each worker with Radau, using
`rtol=1e-10` and `atol=1e-12`.

The runner sets every supported hybrid environment value explicitly. Ambient
`HYBRID_*` settings cannot change the experiment. It currently exposes the
mechanistic control, RA to HOXA5, beta-catenin to MYC and APC mutation terms.
Other exploratory terms continue to use their existing drivers.

### Optional physics kernels

`training.physics_backend` selects `eager` (the default), `triton`, or
`compiled` in either maintained integral or hybrid pipeline. `triton` fuses
the seven-state trapezoidal residual, weighting and squared-error reduction,
with a custom backward for states and RHS values. It requires CUDA,
PyTorch >= 2.6 and Triton; install the `kernels` extra in a compatible CUDA
environment. Imports remain lazy for ordinary CPU runs. Collocation intervals
and state weights must be fixed. Float64 remains float64; there is no implicit
mixed precision or change to the biological equations. The kernel uses
deterministic reductions and no atomic gradient accumulation. Higher-order
gradients use a differentiable PyTorch backward fallback.

The final block sums and normalization share one additional Triton kernel.
It reuses the temporary partial-sum buffer for its scalar output. The custom
loss forward uses two calculation kernels, plus one buffer initialization in
PyTorch's deterministic mode. Its reduction tile is capped at 1,024 values and
loops over larger arrays while retaining the input dtype.

`compiled` applies `torch.compile` to the original RHS and integral loss,
including active learned mechanisms. It uses dynamic row counts and disables
CUDA graphs to accommodate L-BFGS closures. The state networks and optimizer
remain unchanged. Its compilation-variant budget is scoped to 128 to cover
multiple protocols and training stages; final scoring uses eager PyTorch.
Both optimized backends require the integral residual;
ordinary derivative-residual runs retain the eager path.

For a matched smoke comparison, launch this command in a recorded tmux session:

```bash
PYTHONPATH=src python3 experiments/benchmark_kernels.py \
  --out runs/<new-timestamp>_kernel_comparison --repeats 2
```

Use `--config configs/kernel_smoke_full.json --backends eager triton --skip-micro`
to compare the custom kernel at the usual condition count and collocation size.
The full-workload profile retains the usual 4-by-256 state networks; only the
training duration and reference sampling resolution are shortened.

The script runs the same configuration with each backend through the versioned
runner, reversing their order on alternate repeats. All data, initialization
and collocation seeds match. Each child gets a new timestamped run directory
with its normal manifest and checkpoints. The comparison saves every timing,
observation hash, parameter comparison and closure count in `comparison.json`,
plus `report.md`. A separate process measures forward-plus-backward loss time
and exports a baseline physics profile. Compiler caches live inside the
comparison directory; training and microbenchmarks have separate caches.

Per-start `timings` in `*_starts.json` record synchronized stage boundaries,
the first Adam epoch, and steady Adam time after ten epochs. Compilation or
cache loading is included in first-call and total time. Checkpoint writes are
included in training-stage measurements; process wall time additionally
includes setup, references, evaluation and output files. GPU sharing and
compilation overhead can outweigh a kernel speedup in a short run. This smoke
configuration checks execution and speed, not converged parameter recovery.

The [recorded GPU comparison](performance/2026-09-19-kernel-smoke.md) includes
per-run measurements and the limits of the observed speedups.

To isolate the final reduction change against its previous implementation,
run this command in a recorded tmux session:

```bash
PYTHONPATH=src python3 experiments/benchmark_reduction.py \
  --out runs/<new-timestamp>_reduction/comparison.json
```

The script compares 1,024, 8,000 and 40,001 collocation points in float64,
reverses implementation order over six repeats, verifies values and gradients,
and profiles forward kernel counts. It measures the reduction alone with
preallocated outputs, then the actual loss forward and forward/backward with
their normal allocations. Synchronized host timings include dispatch;
CUDA graph replay timings isolate device work. The maintained trainers do not
use CUDA graphs, so the latter timings do not predict training speed.
The [recorded reduction comparison](performance/2026-09-19-fused-reduction.md)
includes the matched eager/Triton training smoke and numerical checks.

An optional `data.reference_cache` accepts an NPZ file with arrays named
`<regime with underscores>__<condition>__t` and
`<regime with underscores>__<condition>__y`. Arrays must span the configured
horizon, have the configured reference point count, and use the seven-state
order. Input paths are relative to the repository root or absolute. Pickled
reference caches are not accepted by the new runner.

## Seeds and comparisons

`independent-seeds-v1` separates three random streams:

| Stream | Seed rule |
| --- | --- |
| Observations | `seeds.data + condition index`, held fixed across starts |
| Network and parameter initialization | `seeds.initialization + 1000 * start index` |
| Collocation | `seeds.collocation + 1000 * start index` |

Integral collocation uses a fixed grid and draws no random samples. Its seed
is recorded so a protocol change cannot introduce an implicit stream later.
Learned terms are still constructed after state networks. Start zero uses the
original fixed biological parameter guess; subsequent starts jitter it.

Every start records its seeds, observation hashes, physics loss, recovered
parameters and the number within 10% of truth. The selected start has the
lowest final physics loss; the first start wins a tie. Selection does not use
known synthetic parameter errors. Condition order is part of the configuration
because it affects initialization and observation seeds.

Historical defaults couple observation and network seeds across starts. They
are preserved when new seed arguments are omitted. Results produced by the
new protocol must be labeled separately from historical published comparisons.
The legacy drivers also accept `--device`, `--data-seed`, `--init-seed` and
`--collocation-seed`; their shell launchers accept `PINN_DEVICE`, `PINN_THREADS`
and `PINN_CONCURRENCY` and return failure if a child process fails.

## Provenance

A manifest records the resolved configuration hash, Git revision, tracked
changes and their patch hash, source-file hashes and a source snapshot,
installed distribution versions, Python/platform details, CPU count, resource
settings, external input hashes, parent run ID, worker commands, logs,
attempt timestamps, outcomes and output hashes. Per-regime effective settings
also record device, CUDA device name when applicable, dtype, initial state,
parameter values, condition forcing, unknown parameter order and solver settings.

`source/` includes untracked Python source used by a run, so a dirty checkout
does not lose its actual implementation. `dirty.diff` supplements that snapshot
with tracked changes. These records support inspection; they do not create a
portable environment lock or promise identical floating-point arithmetic on
different devices.

## Interruption and resume

Send SIGINT or SIGTERM to the runner, or press Ctrl-C in its tmux pane. Workers
finish the current Adam epoch, save a checkpoint and exit with a nonzero status.
A forced kill can lose work since the last checkpoint. Checkpoint and reference
cache writes use a durable journal; resume finishes an interrupted replacement
only when its bytes match the recorded checksum. If a pending write is incomplete,
a verified older checkpoint is retained. Changed committed bytes are refused. Resume with the same
configuration and existing run directory:

```bash
python experiments/run.py configs/smoke_cpu.json --resume runs/<run-id>
```

Adam checkpoints contain every condition's state network, shared biological
parameters, learned-term weights, Adam optimizer, cosine scheduler, adaptive
physics weight, loss history and Python, NumPy, CPU/CUDA Torch and collocation
RNG state. Completed starts and regimes are reused.

L-BFGS and frozen-state refinement resume by replaying those stages from the
saved end of Adam. Internal L-BFGS line-search history is not checkpointed.
This can repeat computation; it is not continuation from an arbitrary closure.
The end-of-Adam checkpoint is always written before those stages begin.

Resume rejects a changed resolved configuration, source or input hash, Python/numerical package versions, effective device, missing
saved outputs, or changed checkpoints/reference arrays. A lock prevents two
writers from executing the same run. Workers inherit that lock, so killing
the supervisor does not permit a second writer while its workers remain active. Changing architecture, training duration,
resources or protocol requires a new run and optional `parent_run` reference.

The runner's Python API is `load_config(path)` and
`run_experiment(config_path, root=repository_path, resume=run_directory)`.
It returns the run directory and raises `RunFailed` or `RunInterrupted` when
execution does not complete. Call it from the main thread so signal handling
can propagate interruption to workers.
