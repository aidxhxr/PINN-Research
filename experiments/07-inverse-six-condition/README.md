# 07 · Six conditions, frozen-network refinement and a classical ODE fit

Does recovery improve with six retinoid-protocol conditions and a final
refinement of the parameters on frozen state networks? And does a classical
estimator, given exactly the same data, do better than the PINN? The PINN
reaches 26/144 parameter–regime pairs within 10%. A multiple-condition ODE
fit (`recover_odefit.py`) reaches 61/144. At the time this was read as two
ceilings: one specific to the PINN (about 8/36) and one set by the
information in the data (about 13–18/36 "by any method"). The second reading
was later shown to be wrong, because the ODE fit was under-converged (see
Notes).

| | |
|---|---|
| **Status** | superseded (information lever: [09](../09-inverse-excitation/); PINN estimator: [10](../10-inverse-integral-residual/); converged classical fit: [17](../17-sensitivity-pinn/)) |
| **Period** | 2026-06 to 2026-07 |
| **Builds on** | [06 · three conditions](../06-inverse-three-condition/) |

## Result

Parameters within 10% of truth, out of 36 per regime (same data for both
estimators):

| Estimator | Normal | Early | Advanced | Severe | Total /144 |
|---|---:|---:|---:|---:|---:|
| PINN + Stage-3 refine (`runs/20260630_042144`) | 10 | 4 | 5 | 7 | 26 |
| ODE fit (`runs/20260630_032142_odefit`) | 18 | 17 | 13 | 13 | 61 |

For the ODE fit, the error on W is 2.4 / 3.1 / 23.7 / 28.4% and on θP is
16.7 / 5.7 / 58.4 / 187% (Normal to Severe).

Profile likelihood, Normal regime only (`runs/20260701_053210_ident`): 10
parameters IDENT (tight 95% CI), 22 WEAK, 4 NON-IDENT (unbounded: `deltaP1`,
`etaB13`, `kappaM13`, `aC`).

## Method

- Six conditions, all varying only the known retinoid forcing: `ctrl`,
  `noATRA` (DR = 0), `earlyATRA` (DR 2.5, τ 20–55), `lateATRA` (DR 2.0,
  τ 90–130), `lowATRA` (DR 0.6, τ 40–80), `strongCirc` (AR 0.12, DR 0).
- Data: 150 points per condition, noise sd 0.002, seed 42.
- PINN: one 4 × 256 Fourier-feature state network per condition, 36 shared
  parameters, 10,000 collocation points, 2500 Adam epochs, 600 L-BFGS steps.
  Stage 3 then runs 500 L-BFGS steps on the parameters alone against the
  physics residual, with the networks frozen.
- ODE fit: integrate the ODE under every condition (LSODA, rtol 1e-7,
  atol 1e-9), stack residuals against the same observations, and solve with
  `scipy.optimize.least_squares` (trust-region `trf`, `max_nfev=40`, one
  start at +50%) in the same log/sigmoid coordinates.
- Profile likelihood (`profile_identifiability.py`, `identifiability`
  v0.5.0 on lmfit): 7 scan points per parameter over a factor-25 window,
  95% CI, started from the ODE-fit estimate.

## Contents

| path | purpose |
|---|---|
| `main.py`, `training.py` | PINN settings; multi-condition training with Stage-3 refinement |
| `recover_odefit.py` | classical multiple-condition ODE fit; writes `*_odefit_recovered.json`, `odefit_summary.csv` |
| `profile_identifiability.py` | profile-likelihood CIs per parameter |
| `identifiability.py`, `plot_error.py` | sensitivity/recovery verdict table; error-vs-iteration plot |
| `config.py`, `model.py`, `odes.py`, `residual.py`, `reference.py`, `plotting.py`, `metrics.py` | as in [06](../06-inverse-three-condition/), with six conditions |

## Reproduce

```bash
cd experiments/07-inverse-six-condition
mkdir -p runs/<timestamp>_odefit
python3 recover_odefit.py runs/<timestamp>_odefit      # CPU, about 20 min (four regimes in sequence)
bash run.sh                                            # PINN; one GPU, about 3 h; launch inside tmux
python3 profile_identifiability.py runs/<timestamp>_odefit Normal   # pip install identifiability==0.5.0
```

`recover_odefit.py` does not create its output directory. The profile script
starts from `<regime>_odefit_recovered.json` if it finds one in the output
directory.

## Notes

- **The "13–18/36 ceiling" is not a property of the data.** It was traced to
  the ten-condition version of this script in [09](../09-inverse-excitation/)
  (`max_nfev=80`, finite-difference Jacobians on an LSODA solve at rtol
  1e-7). This version stops even earlier (`max_nfev=40`, one start). A
  converged maximum-likelihood fit on the ten-condition benchmark recovers
  35/34/33/20 = 122/144 from the same +50% start; the Severe fit stops in a
  local minimum ([17](../17-sensitivity-pinn/)).
- The PINN's "about 8/36" ceiling was broken later by changing the estimator,
  not the network ([10](../10-inverse-integral-residual/),
  [17](../17-sensitivity-pinn/)).
- `deltaP1` cannot be identified in Normal by construction: θP = 1 multiplies
  it by zero in δP = 1 + δP1(1 − θP). The profile CIs are centred on the
  under-converged fit, so read them as qualitative.
- A 2026-07-12 PINN rerun under the synced config (τ = 88, new regime names,
  `runs/20260712_204546`) gives 9/7/6/5 = 27/144.
- The profile-likelihood queue for the remaining regimes
  (`runs/20260705_020527_profile_ident`) never started.
- Historical run directories are git-ignored and indexed in
  `artifacts/catalog.json`. Progression table:
  [publications/papers/2026-08-pinn-identifiability/](../../publications/papers/2026-08-pinn-identifiability/).
