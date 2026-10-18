# 10 · Inverse PINN with a derivative-free integral residual

Can the inverse PINN itself recover more of the 36 kinetic parameters if the
physics residual stops differentiating the network in time? This folder adds
four levers to the ten-condition inverse PINN of
[09](../09-inverse-excitation/): a trapezoidal *integral* residual that uses
network values only, relative state weighting, a fixed collocation grid and
multiple starts. Together they raise recovery from 37 to 50 of 144
regime-parameter pairs. A later matched ablation, adding the levers in steps on
shared benchmark data for the final paper, shows that the weighting and fixed
collocation carry the gain; on those data the integral residual adds nothing
on top of them. Every variant stays 10 to 92 times above the Cramér-Rao bound
at the median, which is the "estimator gap" closed in
[17](../17-sensitivity-pinn/).

| | |
|---|---|
| **Status** | maintained (the `wnt-pinn` runner trains this pipeline: `configs/integral.json`, see [`docs/experiments.md`](../../docs/experiments.md)); the integral PINN is the starting point of the SGN-PINN in [17](../17-sensitivity-pinn/) |
| **Period** | 2026-07 (original runs), 2026-10 (matched ablation) |
| **Builds on** | [09 · inverse-excitation](../09-inverse-excitation/) |
| **Paper** | [Fig 5, Section 2.5; Methods 4.5](../../paper/main.pdf) |

## Result

Parameters within 10% of truth, ten conditions, original runs (registry record
`inverse_recovery`, all 36 parameters counted):

| inverse PINN | Normal | Early | Advanced | Severe | total / 144 |
|---|---:|---:|---:|---:|---:|
| autodiff residual ([09](../09-inverse-excitation/)) | 16 | 9 | 6 | 6 | 37 |
| integral residual + weighting + fixed collocation + 2 starts | 17 | 16 | 10 | 7 | 50 |

That is a gain of 13 pairs, not a doubling. Because four things changed at
once, this comparison does not isolate the residual.

Matched ablation for the paper: benchmark data (seed 42), one start each,
counted over the 34 parameters outside the degenerate (thetaP, deltaP1) pair:

| variant (`--variant`) | Normal | Early | Advanced | Severe | median error / CRLB sd |
|---|---:|---:|---:|---:|---|
| autodiff, unit weights, random collocation (`base`) | 14 | 10 | 7 | 2 | 43-92x |
| + relative weights + fixed collocation (`plumb`) | 21 | 19 | 10 | 10 | 10-22x |
| + integral residual (`integral`) | 17 | 15 | 10 | 7 | 13-33x |
| CRLB-expected count of an efficient estimator | 34.0 | 34.0 | 33.4 | 31.1 | 0.67 |

The integral PINN's median relative errors are 8.5%, 12.2%, 15.6% and 30.6%. On
the held-out replicate dataset (seed 7) it recovers 18, 16, 13 and 9 of 34. On
the benchmark data a converged classical fit recovers all 34 in every regime
(see [17](../17-sensitivity-pinn/)).

## Method

- One Fourier-feature GELU network per condition (width 256, depth 4, 16
  frequencies at scale 4; 207,879 weights each), with one shared vector of 36
  log-parameters. Ten conditions, 150 noisy observations per condition
  (sigma = 0.002), 8,000 collocation points.
- Training: Adam for 2,000 epochs (learning rate 1e-3 for the networks, 5e-3
  for the parameters, cosine schedule), then 150 L-BFGS steps, then 600
  parameter-refinement steps with the networks frozen. Loss weights are
  adaptive and the initial-condition weight is 20.
- `residual_mode=integral`: R_k = (z_{k+1} - z_k)/dt - (F_k + F_{k+1})/2 on the
  sorted grid. The time derivative of the network is never used.
- `rel_weight`: data and residual terms are divided by a per-state scale, so
  the small states (b, h5, h13, m) are not drowned out by beta-catenin and APC.
- `colloc_mode=fixed`: a fixed grid replaces fresh random collocation draws
  at each step.
- `n_starts`: start 0 is the common guess (1.5x truth, W = 1.2, thetaP = 0.6);
  further starts are jittered log-normally around it. The start with the lowest
  final physics residual is kept, so selection never uses the truth.
- Variants in `run_boost.py`: `base` (all levers off), `plumb` (weighting and
  fixed collocation), `integral` (`plumb` plus the integral residual), `siren`
  (`plumb` with a SIREN network).

## Contents

| path | purpose |
|---|---|
| `run_boost.py` | single-regime driver; `--variant {base,plumb,integral,siren}`, `--starts`, `--data-seed`, `--device`, `--adam/--lbfgs/--refine` |
| `run_boost.sh` | runs one variant over all four regimes in parallel, then writes `summary.txt` with `aggregate.py` |
| `run_queue.sh` | runs several variants one after another |
| `run_paper_ablation*.sh` | the matched `base`/`plumb` ablation for the paper (outputs go to `../17-sensitivity-pinn/runs/20261017_paper/ablation_seed42/`) |
| `training.py`, `model.py`, `residual.py` | training stages and levers; the residual and networks come from `../../src/wnt_pinn/` |
| `config.py`, `odes.py`, `reference.py`, `prep_refs.py` | regimes, conditions, Radau references (cached once per run) |
| `aggregate.py`, `identifiability.py`, `plot_error.py`, `regen_plots.py` | count recoveries, per-parameter verdicts, error-versus-iteration plots, standard figures from a finished run |
| `recover_odefit.py` | classical multiple-condition ODE fit (see [07](../07-inverse-six-condition/)) |
| `profile_identifiability.py` | exploratory profile likelihood (needs the `identifiability` package); no completed profile output is saved |
| `pinn_core_dynamics.py`, `pinn_inverse_core_dynamics*.png` | inverse-PINN state trajectories for the core proteins, `ctrl` condition |
| `main.py`, `run.sh`, `plotting.py`, `metrics.py` | older single-process driver with built-in plotting |

## Reproduce

```bash
pip install -e '.[ml,publications]'      # from the repository root: shared model, torch, matplotlib
cd experiments/10-inverse-integral-residual
bash run_boost.sh integral 2      # 4 regimes in parallel -> runs/<timestamp>_integral/
python3 aggregate.py runs/<timestamp>_integral/
python3 regen_plots.py runs/<timestamp>_integral/

# matched ablation used in the paper (benchmark data, 1 start per rung)
bash run_paper_ablation_cpu.sh    # queues base and plumb for data seeds 42 and 7, 4 at a time, skips finished runs
bash run_paper_ablation_plumb.sh  # plumb rung, seed 42 only
```

The paper uses the seed-42 `base` jobs of the first script and the `plumb`
rung of the second. The seed-7 ablation was not run. The `run_paper_ablation*`
scripts put this checkout's `src/` on `PYTHONPATH` themselves.
Alternatively, install the package with
`pip install -e '.[ml]'`. Both scripts pass `--device cuda` and need a GPU, despite the `_cpu` name of the first.

`PINN_DEVICE`, `PINN_THREADS` and `PINN_CONCURRENCY` control the device,
threads per regime and number of parallel regimes. Run long jobs in a
persistent terminal session. Training is float64. On CPU with 16 threads,
one start of one regime took 40-43 minutes in the paper's timing (shared
node). On a shared GPU, one start of the July 2-start runs took about 50 minutes (2026-07-02) and 2-3 hours (the 2026-07-11 retraining).
The integral rung of the ablation is the 3-start run
`../17-sensitivity-pinn/runs/20261002_integral_seed42/` (the paper scores its
start 0), and the replicate is `20261002_integral_seed7/`. Both are tracked.
This folder's own `runs/` directories are git-ignored. Their files are listed
with hashes in [`artifacts/catalog.json`](../../artifacts/catalog.json). The
registered result reads `runs/20260711_203325_integral/`, a retraining of
`runs/20260702_160102_integral/` after the regimes were renamed, which gives
the same counts. See [`results/registry.json`](../../results/registry.json)
and the [reproduction map](../../paper/REPRODUCE.md).

## Notes

- The 37-versus-50 comparison changes the residual, weighting, collocation and
  number of starts together. The paper's stepwise ablation (one
  start, one dataset) attributes the gain to weighting and fixed collocation.
- `aggregate.py` prints a "baseline" of 10/4/5/7. That is an earlier
  six-condition PINN, not a matched control.
- Stage 3 (frozen-network refinement) is a no-op in these runs: the physics
  loss and all parameters are identical from refinement step 50 to step 600
  (for example Normal, start 0: Lp = 3.71e-8, 17/36 throughout).
- The July `plumb` and `siren` runs stopped with an autograd error before
  producing counts, so no SIREN result exists. The `plumb` and `base` rungs
  were run successfully in October.
- Before these runs, the end of the ATRA treatment window `tau2` was changed
  from 80 to 88. Reference trajectories therefore differ slightly from those
  of the earlier folders.
- Accurate state trajectories can coexist with large parameter errors. The
  shortfall comes from the optimiser, not from the data: see
  [11](../11-fisher-information/) and [17](../17-sensitivity-pinn/).
