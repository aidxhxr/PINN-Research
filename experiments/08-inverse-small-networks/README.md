# 08 · Smaller, regularized networks and alternating optimization

Is the inverse PINN's low recovery caused by over-capacity state networks that
absorb the residual at wrong parameter values? This experiment keeps the
six-condition setup of [07](../07-inverse-six-condition/) and attacks
capacity directly. Each state network is shrunk from 4 × 256 to 3 × 128
(207,879 to 38,279 weights), weight decay is applied to the networks only,
and joint training is replaced by alternating network and parameter updates.
The answer is no: 25/144 parameter–regime pairs within 10%, against 26/144
for the unmodified run. Network capacity is not the lever. This is a
negative result.

| | |
|---|---|
| **Status** | complete (negative result) |
| **Period** | 2026-07 |
| **Builds on** | [07 · six conditions](../07-inverse-six-condition/) |

## Result

Run `runs/20260701_052831`, parameters within 10% of truth:

| Regime | this experiment | [07](../07-inverse-six-condition/) PINN |
|---|---:|---:|
| Normal | 9/36 | 10/36 |
| Early Adenoma | 10/36 | 4/36 |
| Advanced Adenoma | 2/36 | 5/36 |
| Severe APC Loss | 4/36 | 7/36 |
| **Total** | **25/144** | **26/144** |

The diagnostic that matters: during the frozen-network refinement in Normal,
W sat at 0.613 (true 0.80, 23.4% error) while the physics loss stayed at
4.5e-5 for all 500 steps. The optimizer was at a minimum of the residual, and
that minimum was not at the true parameters.

## Method

- Same 6 retinoid-protocol conditions, data (150 points per condition, noise
  0.002), collocation (10,000) and stage lengths (2500 Adam, 600 L-BFGS, 500
  Stage-3 refinement) as [07](../07-inverse-six-condition/).
- Smaller network: width 256 → 128, depth 4 → 3, weight decay 1e-4 on the
  network weights only, never on the physical parameters.
- Alternating two-timescale optimization: cycles of 5 network-only steps
  (data + physics + IC) then 5 parameter-only steps (physics residual, networks
  frozen), with separate optimizers and cosine schedules.

## Contents

| path | purpose |
|---|---|
| `main.py` | settings: `width=128`, `depth=3`, `net_weight_decay=1e-4` |
| `training.py` | alternating optimization (`net_steps`, `par_steps`), separate network/parameter optimizers |
| `identifiability.py`, `plot_error.py` | verdict table; error-vs-iteration plot |
| `recover_odefit.py` | unchanged copy of the ODE fit in [07](../07-inverse-six-condition/); not run here |
| `config.py`, `model.py`, `odes.py`, `residual.py`, `reference.py`, `plotting.py`, `metrics.py` | copied unchanged from [07](../07-inverse-six-condition/) |

## Reproduce

```bash
cd experiments/08-inverse-small-networks
bash run.sh                          # launch inside tmux; one GPU, about 2 h; writes runs/<timestamp>/
python3 identifiability.py runs/<timestamp>/
```

## Notes

- Shrinking the network did not reduce the "low loss, wrong parameter"
  behaviour; it only moved where W got stuck (0.613 here, about 0.98 in
  [06](../06-inverse-three-condition/)). This points at the estimator's
  residual (autodiff derivative bias) and at the data, which
  [09](../09-inverse-excitation/) and [10](../10-inverse-integral-residual/)
  address.
- A 2026-07-12 rerun under the synced config (τ = 88, new regime names,
  `runs/20260712_204546`) gives 8/9/3/4 = 24/144. Three runs from 2026-07-02
  in `runs/` are incomplete.
- Saved runs before 2026-07-12 use the old regime names (`Cancer-like`,
  `Strong_APC-mutant`). Historical run directories are git-ignored and
  indexed in `artifacts/catalog.json`.
- Reported in the historical manuscript
  [publications/papers/2026-08-pinn-identifiability/](../../publications/papers/2026-08-pinn-identifiability/).
