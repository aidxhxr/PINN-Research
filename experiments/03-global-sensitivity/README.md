# 03 · Global sensitivity analysis of the 27-state model

Which rate constants of the full 27-state WNT–retinoid–HOX model matter for
its outputs, and which could be fixed? This experiment applies the same three
methods as the seven-state screen in [02](../02-original-pinn/) (local
elasticity, Morris elementary effects and Sobol variance decomposition) to 30
parameters of the 27-state model, then combines them with a conservative
consensus rule. The answer is 22 keep and 8 fix. Unlike the nearly additive
seven-state model, the 27-state model is strongly interacting.

| | |
|---|---|
| **Status** | complete |
| **Period** | 2026-06 |
| **Builds on** | [01 · model prototypes](../01-model-prototypes/) (model port), [02 · original PINN](../02-original-pinn/) (seven-state SA methods) |

## Result

Saved run of 2026-06-29 (Morris N = 24, Sobol M = 32): **22 KEEP / 8 FIX** of 30.

- **FIX** (below 5% of each method's maximum under all three methods):
  `v17, k15, k22, kp4, km1, kp9, km4, km3`.
- **Most influential KEEP** (consensus score): `MCsynth` 0.999, `v16` 0.830,
  `kp5` 0.612, `gamma1` 0.527, `kp6` 0.464, `k14` 0.408, `k8` 0.351.
- The rule is conservative near the threshold: `v17` (consensus 0.0341) is
  fixed while `k23` (0.0329) is kept, because `k23` clears 5% under Morris.
- Interactions dominate: Σ ST runs from 1.9 to 4.9 against Σ S1 of 0.6 to 1.5
  across outputs. On stemness, `k8` has S1 = −0.173 but ST = 0.429. `gamma1`
  is the clean exception, almost purely additive.

## Method

- Model: `model_big.py` is a copy of
  [`../01-model-prototypes/matlab/modelsys_2025_k16dynamic_converted.py`](../01-model-prototypes/matlab/modelsys_2025_k16dynamic_converted.py)
  with the solver settings exposed as arguments. It has 27 states: retinoid
  (13: Ro, Ra, A, R = RA, B, Br, N, Nr, C = CYP26A1, Cr, Dc, Dn, Bc), WNT
  (8: V, Di, Db, Bp, Da, P = APC, Ba = β-catenin, X) and HOX (6: H5, M = MYC,
  Mi, Ca, H13, Ci), plus a derived β-catenin/TCF readout.
- Forcing: a WNT step on t ∈ [3000, 25000]; no ATRA dose. Outputs are scored
  as time averages over the settled window [6000, 12000]
  (`sa_core.AVG_LO`/`AVG_HI`).
- Solver for screening: BDF, rtol 1e-3, atol 1e-6, t_N = 12000 (about 35 s
  per solve). The original reference uses Radau, rtol 1e-6, atol 1e-9,
  t_N = 30000; settled outputs agree to < 1%.
- Parameters (30, ±30% box): the 27-element rate vector (`kp1 … k23`,
  `MCsynth`) plus the coupling knobs `gamma1` (dynamic K16 scale, 1.0),
  `retef` (RA → WNT, 0.0015) and `wntef` (WNT → RA, 100). APC functionality is
  held at 1.0.
- Outputs (9): RA, CYP26A1, APC, β-catenin, HOXA5, MYC, HOXA13, β-cat/TCF and
  a stemness surrogate S = Ba(1 + H13)/(1 + P + H5).
- Consensus (`synthesize.py`): FIX only if the max-normalized,
  output-aggregated score of every method is below 5% of that method's
  maximum.

## Contents

| path | purpose |
|---|---|
| `model_big.py` | 27-state model with tunable solver settings |
| `sa_core.py` | parameters, outputs, scoring window, parallel solve helpers |
| `run_sa.py` | local + Morris + Sobol driver |
| `synthesize.py` | solve-free consensus verdict from the three CSVs |
| `sa.ipynb` | notebook that mirrors `run_sa.py` cell by cell |
| `results/` | `baseline_outputs.csv`, `local_elasticity.csv`, `morris_all.csv`, `morris_overall_ranking.csv`, `sobol_all.csv`, `combined_ranking.csv`, `run_full.log` |
| `sa_plots/` | per-output local/Morris/Sobol figures, heatmaps, `combined_ranking.png` |

## Reproduce

```bash
cd experiments/03-global-sensitivity
python3 run_sa.py                          # local + Morris (N=24) + Sobol (M=32)
python3 run_sa.py --morris-N 24 --skip-sobol
python3 run_sa.py --sobol-N 32 --workers 60
python3 synthesize.py                      # consensus verdict, writes combined_ranking.*
```

CPU only, one solve per worker (`multiprocessing`). With about 60 workers:
local (61 solves) about 1 min, Morris (N·31, e.g. 744) about 8 min, Sobol
(M·32, e.g. 1024) about 11 min. Requires SALib (pinned in
`requirements-lock.txt`).

## Notes

- 56 of 744 Morris solves and 87 of 1024 Sobol solves failed and were
  imputed with the column mean (`results/run_full.log`). Treat small indices
  as approximate.
- At this sample size the Morris flag σ > μ\*/2 fires for 27 to 30 of 30
  parameters on every output, so it is saturated and carries no information.
- The CYP26A1 baseline settles at 7.6e-6, so its Sobol indices are
  numerically meaningless.
- The FIX list concerns the 27-state rate constants. The PINN and
  identifiability experiments ([04](../04-forward-pinn/) onward) use the
  seven-state model's 36 parameters, so this verdict did not define their
  unknown sets.
- Written up in the historical manuscript
  [publications/papers/2026-08-pinn-identifiability/](../../publications/papers/2026-08-pinn-identifiability/).
