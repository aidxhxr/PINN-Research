# 09 · Parameter-exciting conditions: perturbing WNT and MYC directly

Every condition in [06](../06-inverse-three-condition/)–[08](../08-inverse-small-networks/)
varied the retinoid input, which reaches the WNT/HOX/MYC subnetwork through a
single term (ηR·r/(κR + r) in the HOXA5 equation). Do conditions that perturb
that subnetwork directly carry more information? This experiment adds two
known input channels (a WNT drive into β-catenin and a MYC drive) and four
conditions that use them, for ten conditions in total. The classical ODE fit
rises from 61/144 to 82/144 parameter–regime pairs within 10%. The autodiff
inverse PINN on the same data reaches 37/144: the autodiff side of the
registered 37-vs-50 comparison in [10](../10-inverse-integral-residual/).
Severe APC Loss stays hard.

| | |
|---|---|
| **Status** | complete; PINN path continued in [10](../10-inverse-integral-residual/), classical estimator superseded by [17](../17-sensitivity-pinn/) |
| **Period** | 2026-07 |
| **Builds on** | [07 · six conditions](../07-inverse-six-condition/) |

## Result

Parameters within 10% of truth, out of 36 per regime:

| Estimator, conditions | Normal | Early | Advanced | Severe | Total /144 |
|---|---:|---:|---:|---:|---:|
| ODE fit, 6 RA conditions ([07](../07-inverse-six-condition/)) | 18 | 17 | 13 | 13 | 61 |
| ODE fit, 10 conditions (`runs/20260701_203130_odefit_excite`) | 24 | 23 | 21 | 14 | 82 |
| Autodiff PINN, 10 conditions (`runs/20260702_153336`) | 16 | 9 | 6 | 6 | 37 |

- ODE fit, W error 1.5 / 7.9 / 0.0 / 30.7% and θP error 0.9 / 23.3 / 58.0 /
  264.3% (Normal to Severe). The de-saturating WNT-inhibition step was aimed
  at the high-WNT regimes; θP in Severe still does not recover.
- Fourteen parameters recover within 10% in all four regimes: λC, ηRC, κRC,
  κR, κB13, κBC, κBM, ηB13, ηBM, ε13, εC, εM, εR, ρ5.
- Profile likelihood, Advanced Adenoma only: 20 IDENT, 16 WEAK, 0 NON-IDENT
  of 36 (`runs/20260705_020111_profile_ident`; other regimes did not finish).

## Method

- Inputs (`config.py`, `odes.py`): `DW` adds to the WNT drive W in the
  β-catenin equation and `DM` to basal MYC production, each as a smooth tanh
  box like the ATRA pulse. Both are zero by default, so the base model is
  unchanged. They are known inputs and are not estimated.
- New conditions: `wntPulse` (DW = 1.0, τ 30–70); `wntInhib` (DW = −0.7 from
  τ = 50, sustained); `mycPulse` (DM = 1.2, τ 30–70); `wntRAcombo` (DW = 0.8
  from τ = 25, with an ATRA pulse DR = 2.0 on τ 40–80), plus the six
  retinoid conditions of [07](../07-inverse-six-condition/). Data: 150 points
  per condition, noise sd 0.002, seed 42.
- ODE fit (`recover_odefit.py`): as in [07](../07-inverse-six-condition/),
  plus relative-error weighting with a 0.05 scale floor, 4 starts jittered
  around the +50% guess (sd 0.35 in unconstrained space), lowest cost kept,
  `max_nfev=80`.
- PINN (`main.py`): the [07](../07-inverse-six-condition/) PINN unchanged
  (autodiff residual, Stage-3 refinement), one start.

## Contents

| path | purpose |
|---|---|
| `config.py` | ten conditions and the `DW`/`DM` channels |
| `recover_odefit.py` | ten-condition ODE fit with weighting and multi-start; regimes in parallel |
| `main.py`, `training.py` | autodiff inverse PINN over ten conditions |
| `profile_identifiability.py` | profile-likelihood CIs (`identifiability` v0.5.0) |
| `identifiability.py`, `plot_error.py` | verdict table; error-vs-iteration plot |
| `model.py`, `odes.py`, `residual.py`, `reference.py`, `plotting.py`, `metrics.py` | as in [07](../07-inverse-six-condition/), with the extra inputs |

## Reproduce

```bash
cd experiments/09-inverse-excitation
mkdir -p runs/<timestamp>_odefit
python3 recover_odefit.py runs/<timestamp>_odefit    # CPU, 4 regimes in parallel, about 75 min
bash run.sh                                          # autodiff PINN; launch inside tmux (registered run: about 14.5 h on a shared GPU)
python3 profile_identifiability.py runs/<timestamp>_odefit "Advanced Adenoma"
```

`wnt-pinn reproduce --result inverse_recovery --output runs/<new-dir>` (from
the repository root) recomputes the registered 37 vs 50 from saved files.

## Notes

- **The classical fit here is under-converged.** `max_nfev=80` with
  finite-difference Jacobians on an LSODA solve at rtol 1e-7 is the source of
  the old "13–18/36 ceiling". A converged maximum-likelihood fit on the
  matched ten-condition benchmark recovers 35/34/33/20 = 122/144 from the
  same start ([17](../17-sensitivity-pinn/)). The gain from excitation is
  real for this estimator, but 82/144 is not an information limit.
- **The 37 vs 50 comparison** ([docs/decisions/0002-reported-results.md](../../docs/decisions/0002-reported-results.md))
  is a gain of 13 pairs, not a doubling. The residual, relative weighting,
  fixed collocation and multiple starts changed together. A later matched
  one-start ablation, scored on the 34 parameters outside the θP–δP1 pair,
  gives autodiff 14/10/7/2, relative weights plus fixed collocation
  21/19/10/10, and the integral residual on top 17/15/10/7. The weighting
  carries the gain; the integral residual adds nothing over it (scripts in
  [10](../10-inverse-integral-residual/), outputs in
  [17](../17-sensitivity-pinn/)). The often-quoted "+24" mixes this
  comparison with the 6 → 10 condition change.
- The registered autodiff run used the ATRA pulse ending at τ = 80, while the
  integral run in [10](../10-inverse-integral-residual/) used τ = 88. A
  2026-07-12 rerun here at τ = 88 (`runs/20260712_204546`) gives
  17/8/7/6 = 38/144.
- Saved runs before 2026-07-12 use the old regime names (`Cancer-like`,
  `Strong_APC-mutant`), so the profile script will not find their fits under
  the new names; run it on a fresh fit. Run directories are git-ignored and
  indexed in `artifacts/catalog.json`.
- These ten conditions became the base design of the later inverse work
  ([10](../10-inverse-integral-residual/), [11](../11-fisher-information/),
  [17](../17-sensitivity-pinn/), [paper/](../../paper/)).
