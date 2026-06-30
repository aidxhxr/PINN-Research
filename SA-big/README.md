# SA-big — Sensitivity analysis of the big (27-state) Wnt / Retinoid / HOX model

This is the large-system counterpart of the 7-ODE screening in
[`PINN/sensitivity_analysis.ipynb`](../PINN/sensitivity_analysis.ipynb).
It applies the same suite of methods — **local elasticity**, **Morris
elementary-effects screening**, and **Sobol variance decomposition** — to the
full **27-state** nondimensional Wnt / Retinoid / HOX reaction model
(`modelsys_2025_k16dynamic`, converted from the MATLAB driver
`matlab_code/normasysl_call.m`).

## The model

`model_big.py` is a copy of `matlab_code/modelsys_2025_k16dynamic_converted.py`
with the solver settings exposed as keyword arguments so the analysis can trade
fidelity for speed. State vector (27 species + 1 derived readout):

| block | states |
|-------|--------|
| RAS / retinoid (13) | Ro, Ra, A(ALDH), **R = RA**, B(CRABP), Br, N(RAR), Nr, **C = CYP26A1**, Cr, Dc, Dn, Bc |
| Wnt (8) | V, Di, Db, Bp, Da, **P = APC**, **Ba = β-catenin**, X |
| HOX (6) | **H5 = HOXA5**, **M = MYC**, Mi(MIZ1), Ca, **H13 = HOXA13**, Ci |
| derived | **Bcat/TCF** active complex |

The Wnt drive `W(t)` is a step on `t ∈ [3000, 25000]`; the ATRA treatment dose
is 0 here. Under the constant Wnt step the system settles by `t ≈ 10000`, so
outputs are scored as the **time-average over the settled window `[6000, 12000]`**
(see `sa_core.AVG_LO/AVG_HI`).

### Solver

The original reference run uses `Radau, rtol=1e-6, atol=1e-9, t_N=30000`
(minutes per solve — the `gtild` forcing oscillates ~19 000 times over the full
horizon). For screening we use **`BDF, rtol=1e-3, atol=1e-6, t_N=12000`**
(≈ 35 s per solve). Settled output values agree across solvers to <1 %.

## Parameters varied (30)

The **27-element reaction-rate vector `p`** (`kp1, km1, k2, kp3, km3, kp4, km4,
kp5, km5, kp6, km6, k7, k8, kp9, km9, k10, k14, k15, v16, v17, v18, v19, k20,
k21, k22, k23, MCsynth`) plus three **cross-pathway coupling knobs**:

| knob | meaning | baseline |
|------|---------|----------|
| `gamma1` | scales the dynamic K16 (β-cat/TCF availability) | 1.0 |
| `retef`  | RA → Wnt effect (APC-degradation modulation)   | 0.0015 |
| `wntef`  | Wnt → RA effect (CYP synthesis modulation)     | 100 |

Each is varied in a **±30 % box** around baseline (same convention as the
7-ODE screening). `funcpercent` (APC functionality) is held at 1.0 to avoid the
discontinuous `APCdeg` branch confounding the gradients; the integer/boolean
and characteristic-concentration arguments are fixed.

## Outputs scored (9)

`RA, CYP26A1, APC, β-catenin, HOXA5, MYC, HOXA13, Bcat-TCF`, and a **stemness**
surrogate `S = Ba·(1+H13)/(1+P+H5)` (the big-model analog of the 7-ODE
stemness).

## How to run

```bash
cd SA-big
python3 run_sa.py                       # local + Morris (+ Sobol)
python3 run_sa.py --morris-N 24 --skip-sobol
python3 run_sa.py --sobol-N 32 --workers 60
python3 synthesize.py                   # cross-method reduction verdict
```

`synthesize.py` is a fast, solve-free post-step: it reads the three method CSVs
and consolidates them into one per-parameter verdict (run it after `run_sa.py`).

Solves run in parallel (`multiprocessing`, one solve per worker). Total solves:

| stage | solves | wall (≈60 workers) |
|-------|--------|--------------------|
| local | `2·P+1`  ≈ 61   | ~1 min |
| Morris (`N`) | `N·(P+1)` e.g. 24·31 = 744 | ~8 min |
| Sobol (`M`) | `M·(P+2)` e.g. 32·32 = 1024 | ~11 min |

The notebook [`sa.ipynb`](sa.ipynb) mirrors `run_sa.py` cell-by-cell.

## Outputs

* `sa_plots/` — all figures:
  * `local_<output>.png`, `local_heatmap_all.png`
  * `morris_<output>.png`, `morris_mustar_heatmap.png`, `morris_overall_ranking.png`
  * `sobol_<output>.png`, `sobol_ST_heatmap.png`
  * `combined_ranking.png` — cross-method influence bars (FIX candidates shaded)
* `results/` — all numbers:
  * `baseline_outputs.csv`
  * `local_elasticity.csv` (params × outputs)
  * `morris_all.csv` (μ*, σ, μ per param per output), `morris_overall_ranking.csv`
  * `sobol_all.csv` (S1, ST + confidence intervals)
  * `combined_ranking.csv` — per-param `local/morris/sobol` aggregate scores,
    `consensus`, and `fix`/`keep` verdict (the reduction recommendation)

### Reading the results

* **Morris μ\*** ranks overall influence; **σ > μ\*/2** flags nonlinear /
  interacting parameters (orange).
* **Local elasticity** gives the signed, normalized response at baseline
  (linearized); sign disagreements with Morris indicate nonlinearity.
* **Sobol Sₜ** is the total-effect variance share (includes interactions);
  `Sₜ ≫ S₁` again signals interaction.

Parameters that sit near zero across all three methods are candidates to **fix**
when reducing the model / the inverse-problem unknown set. `synthesize.py`
applies exactly this rule: a param is a **FIX candidate** only when each method's
max-normalized, output-aggregated score is below 5 % of that method's maximum.

### Reduction verdict (run `runs` of 2026-06-29: Morris N=24, Sobol M=32)

The consensus split was **22 KEEP / 8 FIX**:

* **FIX (negligible in all three):** `v17, k15, k22, kp4, km1, kp9, km4, km3`
* **Top KEEP (most influential):** `MCsynth, v16, kp5, gamma1, kp6, k14, k8`

These 8 FIX parameters are the recommended candidates to hold constant when
reducing the inverse-problem unknown set. (`Sₜ ≈ 0` for several mid-rank params
that Morris still flags weakly — `synthesize.py` keeps those, since FIX requires
*all three* methods to agree they are negligible.)
