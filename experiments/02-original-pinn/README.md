# 02 · Original PINN for the 27-state model, and the seven-state sensitivity analysis

The first PINN in the project tried to solve the full 27-state
WNT–retinoid–HOX system (23 explicit equations plus 4 implicit WNT
mass-matrix equations) by marching in time windows over t ∈ [0, 30000]. It ran
end to end but was never scored against its reference, and the project moved
to the reduced seven-state model. The folder also holds the first global
sensitivity analysis (SA) of that seven-state model: local elasticity, Morris
and Sobol over 14 parameters. Its original ranking ("θP dominates") was later
found to be contaminated by an inadmissible parameter box. The corrected tables
are saved here.

| | |
|---|---|
| **Status** | historical (27-state PINN abandoned); SA tables complete, corrected 2026-08 |
| **Period** | 2026-06 (SA tables regenerated 2026-08) |
| **Builds on** | [01 · model prototypes](../01-model-prototypes/) |

## Result

**27-state PINN** (`train.log`): the constant-K16 model trained 3 windows,
its full horizon of t = 1500. The dynamic-K16 model
trained all 60 windows to t = 30000, with final window losses between
3.88e-01 and 1.53e+00. No error table was saved, so the figures are
qualitative only.

**Seven-state SA** (stemness output, Normal regime, ±30% box). With
δP1 = 3.5, the APC decay rate δP = 1 + δP1(1 − θP) turns negative for
θP > 1.2857, and the ±30% box reaches θP = 1.30. APC then diverges, and
stemness (which has APC in its denominator) is contaminated too. Clipping θP
at 1.0 changes the global ranking:

| measure | original box (contaminated) | θP clipped at 1.0 |
|---|---|---|
| Morris μ\* rank 1 | `thetaP` 1.212 (next `eta13` 0.337) | `eta13` 0.533, then `thetaP` 0.448, `etaR` 0.442 |
| Sobol S1(θP) | 0.750 | 0.154 (below `eta13` 0.232 and `etaR` 0.167) |
| Morris nonlinearity flags (σ > μ\*/2) | 13 of 14 | 0 of 14 |
| APC Σ S1 | 1.145 (impossible, > 1) | 0.807 |

θP stays dominant in local elasticity (−3.61 on stemness, against 1.04 for
the runner-up), which does not depend on the box.

## Method

- Physics in PyTorch (`wnt_model_physics.py`) and a SciPy BDF reference
  (`wnt_reference_solver.py`, rtol 1e-8, atol 1e-10), for constant (K16 = 30)
  and dynamic K16.
- Network (`wnt_pinn_trainer.py`): tanh MLP, 5 layers × 128 units, fixed
  sin/cos features at the forcing frequency and twice it, hard initial-condition
  constraint, learnable per-state output scale.
- Training (`PINN_solution.py`): windows of length 500 pinned at the WNT
  on/off times (3000, 25000); adaptive per-equation residual weights; per
  window, 4000 Adam iterations (lr 2e-3, 4096 collocation points) then 80
  L-BFGS iterations; per-window checkpoints with resume.
- SA (`sensitivity_analysis.ipynb`, reproduced by `run_sa_7ode.py`): 14
  parameters; outputs are the time-averaged AUC over τ ∈ [0, 150] of the seven
  states plus stemness. Local central differences, Morris (N = 40, 4 levels,
  600 solves), Sobol (N = 256, 4096 solves), seed 42, SALib 1.5.2, LSODA
  rtol 1e-7.

## Contents

| path | purpose |
|---|---|
| `PINN_solution.py`, `PINN_solution.ipynb` | 27-state PINN driver (writes `pinn_results.npz` and four PNGs) |
| `wnt_model_physics.py`, `wnt_reference_solver.py`, `wnt_pinn_trainer.py` | physics, reference solver, windowed trainer |
| `plot_from_checkpoints.py` | figures from saved window checkpoints, mid-run or after |
| `train.log`, `{hox,wnt}_comparison.png`, `wnt_dynamics.png` | the saved 27-state run (PINN solid, reference dashed) |
| `run_sa_7ode.py`, `sa_results/{original_box,clipped_thetap}/` | SA as a script; CSV tables, `sa_config.json`, `run.log` |
| `sensitivity_analysis.ipynb`, `sa_plots/` | the original SA notebook and its figures (original box) |
| `sa.ipynb`, `Untitled.ipynb`, `Untitled-Copy1.ipynb`, `morris_*.png` | earlier Morris-only screens |

`pinn_checkpoints/` and `pinn_results.npz` are git-ignored local outputs.

## Reproduce

```bash
cd experiments/02-original-pinn
pip install SALib==1.5.2     # pinned in requirements-lock.txt
python3 run_sa_7ode.py       # both boxes, CPU, 512 s in the saved run; writes sa_results/
python3 PINN_solution.py     # 27-state PINN; GPU, many hours; launch inside tmux
```

## Notes

- Compare SA rankings across the two boxes, never magnitudes: clipping also
  halves θP's sampled range (0.6 to 0.3) while the other 13 parameters keep
  ±30%.
- The SA uses stemness S = b(1 + h13)/(1 + p + h5) (`sa_config.json`). The
  model specification and the forward PINN in [04](../04-forward-pinn/) use
  b(1 + h13)/((1 + p)(1 + h5)).
- The SA baseline is the Normal regime with the ATRA pulse ending at τ = 80;
  later experiments use τ = 88.
- `Untitled*.ipynb` reads `param_values` above the cell that defines it, so it
  fails on a fresh run-all.
- The corrected SA is reported in the historical manuscript
  [publications/papers/2026-08-pinn-identifiability/](../../publications/papers/2026-08-pinn-identifiability/).
  The 27-state counterpart is [03](../03-global-sensitivity/).
