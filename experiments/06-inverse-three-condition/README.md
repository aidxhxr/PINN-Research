# 06 · Inverse PINN with three conditions and better conditioning

Can the single-condition inverse PINN of [05](../05-inverse-single-condition/)
be rescued by more experiments and better-conditioned optimization? This
experiment fits three experimental conditions (control, no ATRA, early ATRA),
each with its own state network, sharing one set of 36 parameters. It adds a
log parameterization, decoupled learning-rate schedules that fix a
learning-rate freeze bug, adaptive gradient-norm loss weighting and
per-parameter identifiability reports. The gain is small: 30/144
parameter–regime pairs within 10%, against 26/144 from one condition. In the
Normal regime W and θP still ended 21% and 27% off (05 had both within 1.2%
when only those two were unknown). Controlled diagnostics run from this
folder located the cause: gradient starvation and autodiff-derivative bias.

| | |
|---|---|
| **Status** | superseded by [07 · six conditions](../07-inverse-six-condition/) |
| **Period** | 2026-06 |
| **Builds on** | [05 · single condition](../05-inverse-single-condition/) |

## Result

Run `runs/20260629_162528` (36 unknowns, 3 conditions), parameters within
10% of truth:

| Regime | within 10% | W error | θP error |
|---|---:|---:|---:|
| Normal | 8/36 | 21.2% | 26.8% |
| Early Adenoma | 12/36 | 3.2% | 11.6% |
| Advanced Adenoma | 6/36 | 30.2% | 28.2% |
| Severe APC Loss | 4/36 | 44.8% | 149.7% |
| **Total** | **30/144** | | |

The signature is low loss at the wrong parameters. In the Normal regime W
fell from its start of 1.19 to 0.974 within 100 Adam epochs and stayed at
0.974–0.986 for the remaining 2900 epochs while the total loss fell almost
400-fold (1.2e-1 to 3.0e-4).

Diagnostics (Normal regime; scripts and logs in this folder):

- **Gradient matching** (`_test_gradmatch.py`): networks fitted to data
  alone (trajectory RMSE 0.0045–0.0055), then the parameters fitted to the
  residual of the frozen networks, give 6/36 within 10% and W = 0.504
  (true 0.80, 37.0% error), although the residual was far lower.
- **More RA conditions** (`_test_ncond.py`, same protocol): 3, 6 and 8
  conditions give 5, 5 and 7 of 36, while the W error worsens from 35.1% to
  38.1% to 39.0%.
- **Denser, cleaner data** (`_test_dense.py`, 300 points, noise 0.001): W
  still sits at 0.989 by Adam epoch 700, where the log ends.

## Method

- Conditions (`config.py`): `ctrl`; `noATRA` (DR = 0); `earlyATRA` (DR = 2.5,
  τ 20–55). Each changes only the known retinoid forcing.
- One Fourier-feature state network per condition (4 × 256, output scaled per
  state), one shared `InverseParams`. Positive parameters are nominal × exp(s);
  θP stays sigmoid-bounded.
- 60 observations per condition, noise sd 0.005, 12,000 collocation points,
  3000 Adam epochs then 500 L-BFGS steps, network lr 1e-3, parameter lr 5e-3.
  The parameter cosine schedule has a floor of 0.3 × lr so it cannot freeze.
  The physics weight is nudged toward the gradient-norm ratio; λ_ic = 20.
- Start: 1.5 × the baseline value of every shared parameter, with W = 1.20
  and θP = 0.60 in all regimes.
- `identifiability.py` combines an a-priori relative RHS sensitivity with the
  recovery error into YES (≤ 5%), MARGINAL (5–20%) or NO (> 20%)
  (`identifiability.csv/.json`).

## Contents

| path | purpose |
|---|---|
| `main.py`, `training.py` | settings; multi-condition training, decoupled schedules, adaptive weights |
| `model.py`, `config.py` | log-parameterized `InverseParams`, conditions, initial guess |
| `identifiability.py`, `plot_error.py` | verdict table; error-vs-iteration plot from `*_history.json` |
| `_test_gradmatch.py`, `_test_ncond.py`, `_test_dense.py` + `_test_*.log` | diagnostics above |
| `architecture.{tex,pdf}` | network diagram |
| `odes.py`, `residual.py`, `reference.py`, `metrics.py`, `plotting.py` | model, residual, Radau references, figures |

## Reproduce

```bash
cd experiments/06-inverse-three-condition
bash run.sh                          # launch inside tmux; one GPU, about 2 h; writes runs/<timestamp>/
python3 identifiability.py runs/<timestamp>/
python3 plot_error.py runs/<timestamp>/
python3 _test_ncond.py 6             # diagnostic: number of conditions as argument
```

`_test_dense.py` writes to `/tmp/pinn_test_dense` unless `TEST_DENSE_OUT` is set.

## Notes

- The saved runs use the old regime names (`Cancer-like` = Advanced Adenoma,
  `Strong_APC-mutant` = Severe APC Loss) and the ATRA pulse ending at τ = 80.
  `config.py` was synced to the new names and τ = 88 on 2026-07-12; a rerun
  on that date (`runs/20260712_204546`) gave the same 8/12/6/4.
- At the time the failure was read as structural ill-posedness. Later work
  showed the data carry far more information than the PINN extracted, so this
  is an estimator gap ([07](../07-inverse-six-condition/),
  [10](../10-inverse-integral-residual/),
  [17](../17-sensitivity-pinn/)).
- Historical run directories are git-ignored and indexed in
  `artifacts/catalog.json` ([docs/artifacts.md](../../docs/artifacts.md)).
  The full progression table is in
  [publications/papers/2026-08-pinn-identifiability/](../../publications/papers/2026-08-pinn-identifiability/).
