"""Direct ODE-integration least-squares parameter recovery (Peifer-Timmer
style), bypassing the PINN state net entirely. We integrate the ODE with the
current 36 params under every condition, stack the residuals against the sparse
noisy observations, and let scipy's trust-region least-squares (which handles
the parameter correlations / ill-conditioning far better than gradient descent)
recover the params. No net-derivative noise -> the cleanest shot at the
identifiable params (W, thetaP, ...).

Param transform: positive params s=log(p/nominal); thetaP logit-bounded (0,1)."""
import sys, time, numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from config import (BASELINE, REGIMES, CONDITIONS, UNKNOWN, INIT_GUESS,
                    NOMINAL, PARAM_RANGE, Y0)
from odes import _ode_rhs

T = 150.0
N_DATA = 150
NOISE = 0.002
rng = np.random.default_rng(42)
regime = sys.argv[1] if len(sys.argv) > 1 else "Normal"
true_p = {**BASELINE, **REGIMES[regime]}
true_vals = {k: true_p[k] for k in UNKNOWN}

def solve(p, t_eval):
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0, t_eval=t_eval,
                    method="LSODA", rtol=1e-7, atol=1e-9)
    return sol.y.T if sol.success else np.full((len(t_eval), 7), 1e3)

# build sparse noisy data per condition (same recipe as the PINN folder)
t_grid = np.linspace(0, T, 5000)
data = []
for ci, c in enumerate(CONDITIONS):
    p = {**true_p, **c["forcing"]}
    y_full = solve(p, t_grid)
    r = np.random.default_rng(42 + ci)
    idx = np.sort(np.concatenate([[0], r.choice(np.arange(1, 5000),
                                                 N_DATA - 1, replace=False)]))
    td = t_grid[idx]
    yd = y_full[idx] + r.normal(0, NOISE, (len(idx), 7))
    data.append((c["forcing"], td, yd))

# pack/unpack the 36 params <-> unconstrained vector
def unpack(x):
    p = dict(BASELINE)
    for k, xi in zip(UNKNOWN, x):
        lo, hi = PARAM_RANGE[k]
        if hi is None:
            p[k] = NOMINAL[k] * np.exp(xi)
        else:
            p[k] = lo + (hi - lo) / (1 + np.exp(-xi))
    return p
x0 = []
for k in UNKNOWN:
    lo, hi = PARAM_RANGE[k]
    if hi is None:
        x0.append(np.log(INIT_GUESS[k] / NOMINAL[k]))
    else:
        z = (INIT_GUESS[k] - lo) / (hi - lo)
        x0.append(np.log(z / (1 - z)))
x0 = np.array(x0)

neval = [0]
def resid(x):
    neval[0] += 1
    p = unpack(x)
    out = []
    for forcing, td, yd in data:
        ys = solve({**p, **forcing}, td)
        out.append((ys - yd).ravel())
    return np.concatenate(out)

print(f"=== ODE-fit {regime}: {len(CONDITIONS)} conditions, {len(UNKNOWN)} params ===")
t0 = time.time()
res = least_squares(resid, x0, method="trf", x_scale="jac",
                    ftol=1e-10, xtol=1e-10, max_nfev=200, verbose=2)
p = unpack(res.x)
errs = {k: abs(p[k] - true_vals[k]) / abs(true_vals[k]) * 100 for k in UNKNOWN}
ok = sum(e <= 10 for e in errs.values())
print(f"\n=== ODE-fit {regime}: {ok}/{len(UNKNOWN)} under 10%  "
      f"(W={errs['W']:.1f}%  thetaP={errs['thetaP']:.1f}%)  "
      f"{neval[0]} solves, {time.time()-t0:.0f}s ===")
for k in sorted(UNKNOWN, key=lambda k: errs[k]):
    print(f"  {k:10s} rec={p[k]:8.3f} true={true_vals[k]:7.3f} err={errs[k]:6.1f}%"
          + ("  OK" if errs[k] <= 10 else ""))
