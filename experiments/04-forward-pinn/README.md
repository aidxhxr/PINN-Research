# 04 · Forward PINN for the seven-state model

With the parameters known, can a PINN reproduce all seven states of the
reduced WNT–RA–HOX model in all four regimes, including the circadian
retinoid input and the ATRA pulse? Two sub-experiments use the same network
architecture.
`forward_pinn_train/` is a dense supervised baseline: 100 evenly spaced labels
plus the ODE residual. `forward-pinn-train-hybrid/` is the sparse PINN: 40
observation times (t = 0 plus 39 random), an initial-condition loss and the
ODE residual, so the physics must fill the gaps. Both reach low error: a mean
relative L2 error of 2.41% for the sparse PINN and 1.06% for the dense
baseline. Representation is therefore not what limits the inverse experiments
that follow.

| | |
|---|---|
| **Status** | complete |
| **Period** | 2026-06 to 2026-07 (registered runs 2026-07-11/12) |
| **Builds on** | [02 · original PINN](../02-original-pinn/) |

## Result

Mean relative L2 error (%) over the seven states, against the Radau reference
on 6000 points:

| | Normal | Early | Advanced | Severe | grand mean | without APC | grand NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense, 100 labels (`runs/20260711_203325`) | 0.49 | 0.65 | 1.17 | 1.91 | 1.06 | 0.60 | 0.56 |
| Sparse, 40 obs. (`runs/20260712_204546`) | 2.83 | 1.76 | 2.09 | 2.95 | 2.41 | 1.59 | 1.68 |

The largest error is APC in the high-WNT regimes, where APC settles near zero:
13.53% relative L2 but 1.26% NRMSE (sparse, Severe APC Loss). Quote NRMSE for
APC.

## Method

- Reference: SciPy Radau, rtol 1e-10, atol 1e-12, T = 150, ATRA pulse on
  τ ∈ [40, 88].
- Network (`model.py`): random Fourier-feature embedding of time (16
  frequencies, σ = 4.0, plus raw time), then a GELU MLP of 4 layers × 256
  units (207,879 parameters). Without the Fourier features the MLP's spectral
  bias flattened the circadian and ATRA modes into near-straight lines.
- Sparse PINN: 40 noise-free observations (seed 42); loss
  λ_data·L_data + λ_phys·L_phys + λ_ic·L_ic with 1, 1 and 20.
- Dense baseline: 100 evenly spaced labels, λ_data = 0.7, λ_phys = 0.3, no
  separate IC term.
- Both: 50,000 random collocation points, 1000 Adam epochs (lr 1e-3), then
  500 L-BFGS steps; one network per regime.
- Scoring: `score_forward.py` computes per-state relative L2 and NRMSE
  (RMSE over the reference range).

## Contents

| path | purpose |
|---|---|
| `forward-pinn-train-hybrid/main.py`, `training.py` | sparse PINN: settings, sparse sampling, IC loss |
| `forward_pinn_train/main.py`, `training.py` | dense supervised baseline |
| `*/model.py`, `config.py`, `odes.py`, `residual.py`, `reference.py` | network, parameters and regimes, equations, Radau reference |
| `*/score_forward.py` | writes `forward_error_table.{txt,json}` for a run |
| `*/compare.py`, `*/regen_plots.py` | comparison grid and per-state plots from saved checkpoints |
| `forward-pinn-train-hybrid/pinn_core_dynamics.py` | core-dynamics figure (`pinn_forward_core_dynamics*.png`) |
| `forward_pinn_train/compare_grid6.py`, `scipy_core_dynamics*.py` | six-panel overlay; reference-only figures |
| `forward-pinn-train-hybrid/architecture.{tex,pdf}` | network diagram |
| `forward_pinn_train/runs/20260711_203325/`, `forward-pinn-train-hybrid/runs/20260712_204546/` | the two tracked runs (dense, sparse): final checkpoints, histories, error tables |
| `*.ipynb` (top level) | early prototypes: a seven-state forward PINN, a 27-state windowed PINN, SciPy solves, an environment probe |

## Reproduce

```bash
cd experiments/04-forward-pinn/forward-pinn-train-hybrid   # or ../forward_pinn_train
bash run.sh        # launch inside tmux; writes runs/<timestamp>/ (git-ignored)
python3 score_forward.py runs/<timestamp> runs/<timestamp>/forward_error_table
python3 pinn_core_dynamics.py      # figure from the newest complete run
```

One GPU; the four regimes train in sequence. The registered runs took 8 to
10 h each on a shared GPU. To recompute the registered numbers from the saved
tables without training, run
`wnt-pinn reproduce --result forward_accuracy --output runs/<new-dir>` from
the repository root ([docs/reproduction.md](../../docs/reproduction.md)).

## Notes

- The two formulations differ in objective as well as data, so this is not
  an isolated data-size ablation (registry result `forward_accuracy`).
- Observations are noise-free here. The inverse experiments use noisy data.
- History: the dense folder first used 3000 labels, which made it supervised
  interpolation rather than a physics-driven solve. Straight-line plots from
  that period were stale pre-Fourier PNGs, not a convergence failure.
- Before 2026-08 the only core-dynamics figure script was in
  `forward_pinn_train/`, so earlier "forward PINN" figures showed the
  100-label baseline. The sparse PINN's figure is the one used in the root
  README.
- `forward_pinn_train/` scripts pick the newest `runs/` directory by
  modification time and fail on an empty one. The sparse copies skip
  directories without final checkpoints.
- Figure 2 of the paper ([paper/](../../paper/)) is built from these two
  runs. A Bayesian forward PINN is in [14](../14-bayesian-forward/).
