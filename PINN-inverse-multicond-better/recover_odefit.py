"""Direct ODE-integration parameter recovery (Peifer & Timmer 2007 multiple-
condition shooting; in the project references) for the 36-parameter WNT-RA-HOX
inverse problem.

WHY THIS EXISTS (see Session Log 2026-06-30): the PINN inverse solver recovers
the state trajectory beautifully but recovers the PARAMETERS poorly — even the
high-sensitivity W froze at ~25% error. Controlled experiments showed the
bottleneck is the state-net's autodiff derivative: minimising the physics
residual on a frozen, data-fit net (gradient matching) still floored at the
net-derivative noise and gave biased params (W -> 0.63, vs joint's 0.98, truth
0.80). Removing the net entirely and integrating the ODE directly with a
trust-region least-squares solver (which handles the parameter correlations)
recovers ~18/36 params under 10% INCLUDING W (2.4%) — far past the PINN ceiling.

This is the parameter-recovery counterpart to the PINN: same model, same 6
conditions, same sparse-noisy data, same log/sigmoid parametrisation. It is a
classical estimator, not a neural net — use it when the GOAL is parameter
accuracy; use main.py (the PINN) when the goal is the continuous state field.
"""
import os, sys, json, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from config import (BASELINE, REGIMES, CONDITIONS, UNKNOWN, INIT_GUESS,
                    NOMINAL, PARAM_RANGE, Y0)
from odes import _ode_rhs

T = 150.0
N_DATA = 150
NOISE = 0.002
SEED = 42
GRID = 5000


def solve(p, t_eval):
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0, t_eval=t_eval,
                    method="LSODA", rtol=1e-7, atol=1e-9)
    return sol.y.T if sol.success else np.full((len(t_eval), 7), 1e3)


def pack_x0():
    x0 = []
    for k in UNKNOWN:
        lo, hi = PARAM_RANGE[k]
        if hi is None:
            x0.append(np.log(INIT_GUESS[k] / NOMINAL[k]))
        else:
            z = (INIT_GUESS[k] - lo) / (hi - lo)
            x0.append(np.log(z / (1 - z)))
    return np.array(x0)


def unpack(x):
    p = dict(BASELINE)
    for k, xi in zip(UNKNOWN, x):
        lo, hi = PARAM_RANGE[k]
        p[k] = NOMINAL[k] * np.exp(xi) if hi is None else lo + (hi - lo) / (1 + np.exp(-xi))
    return p


def recover(regime, out_dir="."):
    true_p = {**BASELINE, **REGIMES[regime]}
    true_vals = {k: true_p[k] for k in UNKNOWN}

    # sparse, noisy observations per condition (matches the PINN data recipe)
    t_grid = np.linspace(0, T, GRID)
    data = []
    for ci, c in enumerate(CONDITIONS):
        y_full = solve({**true_p, **c["forcing"]}, t_grid)
        r = np.random.default_rng(SEED + ci)
        idx = np.sort(np.concatenate(
            [[0], r.choice(np.arange(1, GRID), N_DATA - 1, replace=False)]))
        yd = y_full[idx] + r.normal(0, NOISE, (len(idx), 7))
        data.append((c["forcing"], t_grid[idx], yd))

    def resid(x):
        p = unpack(x)
        return np.concatenate([(solve({**p, **f}, td) - yd).ravel()
                               for f, td, yd in data])

    t0 = time.time()
    res = least_squares(resid, pack_x0(), method="trf", x_scale="jac",
                        ftol=1e-8, xtol=1e-8, gtol=1e-6, max_nfev=40)
    p = unpack(res.x)
    rec = {k: float(p[k]) for k in UNKNOWN}
    errs = {k: abs(rec[k] - true_vals[k]) / abs(true_vals[k]) * 100 for k in UNKNOWN}

    safe = regime.replace(" ", "_").replace("/", "_")
    with open(os.path.join(out_dir, f"{safe}_odefit_recovered.json"), "w") as fh:
        json.dump({"recovered": rec, "true": true_vals, "init": INIT_GUESS,
                   "errors_pct": errs}, fh, indent=2)
    ok = sum(e <= 10 for e in errs.values())
    print(f"  {regime:18s}  {ok:2d}/{len(UNKNOWN)} under 10%  "
          f"(W={errs['W']:.1f}%  thetaP={errs['thetaP']:.1f}%)  {time.time()-t0:.0f}s")
    return rec, true_vals, errs


def main(out_dir="."):
    print("=" * 64)
    print(f"Direct ODE-fit recovery — {len(UNKNOWN)} params, "
          f"{len(CONDITIONS)} conditions: {[c['name'] for c in CONDITIONS]}")
    print("=" * 64)
    all_err = {}
    for regime in REGIMES:
        _, _, errs = recover(regime, out_dir)
        all_err[regime] = errs

    # param-major summary, sorted by mean error (worst last) -> the ranking the
    # downstream sensitivity-based reduction consumes
    regimes = list(REGIMES)
    rows = sorted(UNKNOWN,
                  key=lambda k: np.mean([all_err[r][k] for r in regimes]))
    print("\n" + "=" * 64)
    print("Per-parameter recovery error (%), sorted best -> worst")
    print(f"{'param':10s} {'mean':>6s} | " + "  ".join(f"{r[:8]:>8s}" for r in regimes))
    print("-" * 64)
    csv = ["param,mean," + ",".join(regimes)]
    n_ok = 0
    for k in rows:
        e = [all_err[r][k] for r in regimes]
        m = float(np.mean(e))
        if m <= 10:
            n_ok += 1
        flag = "OK" if m <= 10 else ("~" if m <= 20 else "X")
        print(f"{k:10s} {m:6.1f} | " + "  ".join(f"{ei:8.1f}" for ei in e) + f"  {flag}")
        csv.append(f"{k},{m:.2f}," + ",".join(f"{ei:.2f}" for ei in e))
    with open(os.path.join(out_dir, "odefit_summary.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")
    print("-" * 64)
    print(f"{n_ok}/{len(UNKNOWN)} parameters under 10% (mean over regimes)")
    print("odefit_summary.csv written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
