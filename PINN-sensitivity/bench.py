"""Shared benchmark protocol for the sensitivity-PINN study.

One definition of the inverse problem used by EVERY method compared here
(classical MLE, integral PINN, sensitivity PINN), so differences are
attributable to the estimator, not to the data:

* model, 36 unknowns, 10 conditions: ``wnt_pinn.model.parameters`` (current
  config, tau2=88);
* reference: Radau rtol 1e-10 on a 5000-point grid over [0, 150];
* data: per condition, the t=0 anchor + 149 random grid points
  (rng seed ``data_seed + ci``), all 7 states, iid N(0, 0.002^2) noise;
* weights: 1/max(max|y_ref|, 0.05) per state (the PINN's relative weighting).
"""
import numpy as np
from scipy.integrate import solve_ivp

from wnt_pinn.model.parameters import (BASELINE, REGIMES, CONDITIONS,
                                       INITIAL_STATE, UNKNOWN)
from wnt_pinn.model.numpy_rhs import _ode_rhs

T = 150.0
GRID = 5000
N_DATA = 150
NOISE = 0.002
Y0 = np.array(INITIAL_STATE)

INIT_GUESS = {k: 1.5 * BASELINE[k] for k in UNKNOWN}
INIT_GUESS["W"] = 1.20
INIT_GUESS["thetaP"] = 0.60


def true_params(regime):
    return {**BASELINE, **REGIMES[regime]}


def solve(p, t_eval, rtol=1e-10, atol=1e-12, method="Radau"):
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0,
                    t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        return None
    return sol.y.T


# Added-condition designs (2026-10-17), identical to PINN-hybrid-ude/config.py:
# two information-matched conditions that move every regulator UP, and two
# depletion conditions that drive WNT signalling and RA towards zero.
EXTRA = {
    "base": [],
    "info": [{"name": "midATRA", "forcing": {"DR": 1.20, "tau1": 60.0, "tau2": 100.0}},
             {"name": "wntPulse2", "forcing": {"DW": 0.50, "tauW1": 60.0, "tauW2": 110.0}}],
    "dep": [{"name": "wntKO", "forcing": {"kW": 0.02}},
            {"name": "raKO", "forcing": {"mu0": 0.0, "AR": 0.0, "DR": 0.0}}],
    "depb": [{"name": "bcatKO", "forcing": {"kW": 0.0, "k13": 0.0}},
             {"name": "raKO", "forcing": {"mu0": 0.0, "AR": 0.0, "DR": 0.0}}],
}


def make_data(regime, data_seed=42, design="base"):
    """List of dicts (one per condition): forcing, t_d, y_d, w, t_ref, y_ref.
    The ten base conditions are byte-identical for every design."""
    p = true_params(regime)
    t_grid = np.linspace(0, T, GRID)
    out = []
    for ci, c in enumerate(list(CONDITIONS) + EXTRA[design]):
        y_ref = solve({**p, **c["forcing"]}, t_grid)
        rng = np.random.default_rng(data_seed + ci)
        idx = rng.choice(np.arange(1, GRID), size=N_DATA - 1, replace=False)
        idx = np.sort(np.concatenate([[0], idx]))
        y_d = y_ref[idx] + rng.normal(0.0, NOISE, size=(len(idx), 7))
        w = 1.0 / np.maximum(np.abs(y_ref).max(axis=0), 0.05)
        out.append(dict(name=c["name"], forcing=c["forcing"], t_d=t_grid[idx],
                        y_d=y_d, w=w, t_ref=t_grid, y_ref=y_ref))
    return out


# --- unconstrained coordinates: x = log(p / nominal) for all 36 -----------
# thetaP is log-parameterised too (not logit on (0,1)): Normal's true
# thetaP = 1.0 sits on the logit boundary, which would make it unreachable.
def to_x(vals):
    return np.array([np.log(vals[k] / BASELINE[k]) for k in UNKNOWN])


def from_x(x):
    return {k: BASELINE[k] * np.exp(xi) for k, xi in zip(UNKNOWN, x)}


def rel_errors(rec, regime):
    tv = true_params(regime)
    return {k: abs(rec[k] - tv[k]) / abs(tv[k]) for k in UNKNOWN}


def count_under(rec, regime, tol=0.10):
    return sum(e < tol for e in rel_errors(rec, regime).values())
