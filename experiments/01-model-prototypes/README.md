# 01 · Model prototypes: the 27-state MATLAB model and first notebooks

Where the code base started. Before any neural network, the WNT–retinoid–HOX
biology existed as a 27-state MATLAB model (13 retinoid, 8 WNT and 6 HOX
species, solved with `ode15s` to t = 30000). This folder keeps that model and
its driver, the SciPy ports that made it usable from Python, the scratch
notebooks that first tried a physics-informed neural network (PINN) on it, and
a few early figures. It answers no research question on its own; it records
provenance. The PINN and identifiability experiments from [04](../04-forward-pinn/) onward use the
reduced seven-state model specified in [docs/model.md](../../docs/model.md).

| | |
|---|---|
| **Status** | historical (prototypes, not maintained) |
| **Period** | 2026-05 to 2026-07 |
| **Builds on** | — |

## Result

No quantitative result is claimed from this folder. What it produced:

- A SciPy port of the MATLAB dynamic-K16 model
  (`matlab/modelsys_2025_k16dynamic_converted.py`, Radau, rtol 1e-6,
  atol 1e-9). It is the parent of `model_big.py` in
  [03](../03-global-sensitivity/) and of the physics module in
  [02](../02-original-pinn/).
- Constant- vs dynamic-K16 comparison figures for the 27-state model
  (`figures/*.png`).
- Normal vs Severe APC Loss trajectories of the reduced seven-state model,
  with and without ATRA, for four state pairs (`figures/atra-superimpose/`,
  8 PNGs).

## Method

- `matlab/modelsys_2025_k16dynamic.m`: the 27-state nondimensional system, in
  which the β-catenin/TCF constant K16 depends on a HOX-complex state
  (`gamma1` scales the dynamic part). `matlab/normasysl_call.m` sets the
  27-element rate vector and plots constant vs dynamic K16.
- Python ports: a line-by-line SciPy translation of the MATLAB function, a
  second port with a `__main__` driver (BDF), and a smaller APC /
  β-catenin–TCF submodel (`notebooks/modelsys_smaller.py`).
- Notebooks: a two-state APC/β-catenin toy PINN (`wnt_pinn_starter.ipynb`), a
  first time-windowed forward PINN on the 27-state model (`PINN_forward.ipynb`),
  and a first SciPy simulation of the four regimes of the seven-state model
  (`Untitled.ipynb`).

## Contents

| path | purpose |
|---|---|
| `matlab/modelsys_2025_k16dynamic.m` | 27-state MATLAB model (dynamic K16) |
| `matlab/normasysl_call.m` | MATLAB driver: parameters, constant vs dynamic K16 plots |
| `matlab/modelsys_2025_k16dynamic_converted.py` | SciPy port of the MATLAB function (module) |
| `matlab/practice/practice_scripts.m` | personal MATLAB tutorial exercise; unrelated to the model |
| `notebooks/modelsys_2025_k16dynamic_converted.{ipynb,py}` | second SciPy port; the `.py` has a runnable driver |
| `notebooks/modelsys_smaller.py` | APC / β-catenin–TCF submodel |
| `notebooks/wnt_pinn_starter.ipynb` | two-state toy PINN (supervised stage, then residual) |
| `notebooks/PINN_forward.ipynb`, `PINN_forward-Copy1.ipynb` | first 27-state windowed PINN; the copy differs by one line |
| `notebooks/Untitled.ipynb` | first seven-state, four-regime SciPy simulation |
| `figures/hox_comparison.png`, `hox_components.png`, `wnt_comparison.png`, `wnt_dynamics.png` | early constant- vs dynamic-K16 figures (27-state model) |
| `figures/atra-superimpose/` | ATRA vs no-ATRA state-pair plots (seven-state model) |

## Reproduce

```bash
cd experiments/01-model-prototypes
python3 notebooks/modelsys_2025_k16dynamic_converted.py   # 27-state solve, prints the solution shape
jupyter lab notebooks/                                     # the PINN notebooks need torch; long GPU jobs
```

The MATLAB driver does not run as committed: it calls `modelsys_2025.m`
(constant K16), which is not in the repository. The constant-K16 variant
survives only in Python, in `wnt_model_physics.py` of
[02](../02-original-pinn/).

## Notes

- Several notebooks depend on kernel state that is not in the file:
  `modelsys_2025_k16dynamic_converted.ipynb` uses `plt` without importing it,
  and the stored output of `Untitled.ipynb` is a `ModuleNotFoundError`, so
  neither runs top to bottom as committed.
- `modelsys_smaller.py` saves its figure to a hard-coded absolute path; change
  it before running.
- The `figures/` PNGs share file names with the outputs of `PINN_solution.py`
  in [02](../02-original-pinn/) but are earlier versions; the run that
  produced them is not recorded. No generating script for
  `figures/atra-superimpose/` is in the repository.
- The 27-state model is screened for parameter sensitivity in
  [03](../03-global-sensitivity/). The seven-state model takes over from
  [04](../04-forward-pinn/) onward.
