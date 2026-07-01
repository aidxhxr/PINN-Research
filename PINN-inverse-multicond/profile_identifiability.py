"""Profile-likelihood practical-identifiability analysis of the 36-parameter
WNT-RA-HOX inverse problem, using the `identifiability` PyPI package (v0.5.0,
Raue-style profile likelihood built on lmfit).

WHY: the existing identifiability.csv grades params by their *empirical recovery
error* (how close the ODE-fit landed). This script instead computes the
statistically rigorous verdict: profile each parameter's likelihood, holding it
fixed at a grid of values while re-optimising all the others, and read off the
95% confidence interval. A FINITE CI => practically identifiable; a CI that runs
into the scan boundary (NaN) => practically NON-identifiable (the data do not
constrain it). Same ODE model / 6 conditions / sparse-noisy data recipe as
recover_odefit.py, started from that fit's MLE.

Run:  python3 profile_identifiability.py <out_dir> [regime]
      (default regime: Normal)
"""
import sys, os, json, time

# --- resolve the name clash: this folder also has a local identifiability.py.
# Drop this script's dir from sys.path so `import identifiability` picks the pip
# package, then re-add it so the local config/odes/... still import.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import identifiability as IDENT            # pip package (profile likelihood)
sys.path.insert(0, HERE)

import numpy as np
import lmfit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from config import BASELINE, REGIMES, CONDITIONS, UNKNOWN, INIT_GUESS, Y0

# data / solver recipe — identical to recover_odefit.py
T, N_DATA, NOISE, SEED, GRID = 150.0, 150, 0.002, 42, 5000
POINTS = 7          # profile scan points per parameter
LIMITS = 0.2        # log scan spans [p*0.2, p/0.2] = factor-25 window
PROB = 0.95         # 95% confidence interval


def _rhs_import():
    from odes import _ode_rhs
    return _ode_rhs


def solve(p, t_eval, rhs):
    sol = solve_ivp(lambda t, y: rhs(t, y, p), (0, T), Y0, t_eval=t_eval,
                    method="LSODA", rtol=1e-7, atol=1e-9)
    return sol.y.T if sol.success else np.full((len(t_eval), 7), 1e3)


# module-level state so the objective is a top-level (picklable) function that
# multiprocessing workers can call (fork inherits these globals).
_DATA = None
_RHS = None


def objective(pars):
    p = dict(BASELINE)
    for k in UNKNOWN:
        p[k] = pars[k].value
    return np.concatenate([(solve({**p, **f}, td, _RHS) - yd).ravel()
                           for f, td, yd in _DATA])


def build_data(true_p, rhs):
    t_grid = np.linspace(0, T, GRID)
    data = []
    for ci, c in enumerate(CONDITIONS):
        y_full = solve({**true_p, **c["forcing"]}, t_grid, rhs)
        r = np.random.default_rng(SEED + ci)
        idx = np.sort(np.concatenate(
            [[0], r.choice(np.arange(1, GRID), N_DATA - 1, replace=False)]))
        yd = y_full[idx] + r.normal(0, NOISE, (len(idx), 7))
        data.append((c["forcing"], t_grid[idx], yd))
    return data


def main(out_dir=".", regime="Normal"):
    os.makedirs(out_dir, exist_ok=True)
    rhs = _rhs_import()
    print("=" * 68, flush=True)
    print(f"Profile-likelihood identifiability — regime={regime}, "
          f"{len(UNKNOWN)} params, {len(CONDITIONS)} conditions, "
          f"points={POINTS}, prob={PROB}", flush=True)
    print(f"identifiability pkg: {IDENT.__file__}", flush=True)
    print("=" * 68, flush=True)

    true_p = {**BASELINE, **REGIMES[regime]}
    true_vals = {k: float(true_p[k]) for k in UNKNOWN}
    global _DATA, _RHS
    _RHS = rhs
    _DATA = build_data(true_p, rhs)

    # MLE start from the ODE-fit recovery if available
    safe = regime.replace(" ", "_").replace("/", "_")
    start = {k: INIT_GUESS[k] for k in UNKNOWN}
    for root in (out_dir, os.path.join(HERE, "runs", "20260630_032142_odefit")):
        cand = os.path.join(root, f"{safe}_odefit_recovered.json")
        if os.path.exists(cand):
            start = json.load(open(cand))["recovered"]
            print(f"MLE start loaded from {cand}", flush=True)
            break

    # lmfit params: natural units, positive (min>0), no upper bound so the
    # profile is free to reveal an unbounded (non-identifiable) direction.
    params = lmfit.Parameters()
    for k in UNKNOWN:
        params.add(k, value=max(float(start[k]), 1e-5), min=1e-6)

    mini = lmfit.Minimizer(objective, params)
    t0 = time.time()
    result = mini.minimize(method="least_squares")
    print(f"[{time.time()-t0:.0f}s] MLE fit: nfev={result.nfev}, "
          f"chisqr={result.chisqr:.4e}, redchi={result.redchi:.4e}, "
          f"nfree={result.nfree}", flush=True)

    ci = IDENT.ConfidenceInterval(mini, result, p_names=list(UNKNOWN))
    print("Profiling all parameters (parallel across cores)...", flush=True)
    t0 = time.time()
    ci_values = ci.calc_all_ci(limits=LIMITS, points=POINTS, prob=PROB,
                               method="least_squares", log=True, mp=True)
    print(f"[{time.time()-t0:.0f}s] profiling done", flush=True)

    # verdict + tables
    rows = []
    for k in UNKNOWN:
        lo, hi = ci_values[k]
        mle = float(result.params[k].value)
        finite = np.isfinite(lo) and np.isfinite(hi)
        rel_w = (hi - lo) / abs(mle) if finite and mle != 0 else float("nan")
        if not finite:
            verdict = "NON-IDENT"          # CI hit the scan boundary
        elif rel_w <= 0.5:
            verdict = "IDENT"              # tight, well-constrained
        else:
            verdict = "WEAK"              # bounded but wide
        rows.append(dict(param=k, mle=mle, true=true_vals[k],
                         ci_low=lo, ci_high=hi, rel_width=rel_w,
                         verdict=verdict))

    order = {"IDENT": 0, "WEAK": 1, "NON-IDENT": 2}
    rows.sort(key=lambda r: (order[r["verdict"]],
                             r["rel_width"] if np.isfinite(r["rel_width"]) else 1e9))

    csv = ["param,mle,true,ci_low,ci_high,rel_width,verdict"]
    print(f"\n{'param':10s} {'mle':>9s} {'true':>9s} {'ci_low':>9s} "
          f"{'ci_high':>9s} {'rel_w':>7s}  verdict", flush=True)
    print("-" * 68, flush=True)
    for r in rows:
        lo_s = "bound" if not np.isfinite(r["ci_low"]) else f"{r['ci_low']:.3g}"
        hi_s = "bound" if not np.isfinite(r["ci_high"]) else f"{r['ci_high']:.3g}"
        rw_s = "  inf" if not np.isfinite(r["rel_width"]) else f"{r['rel_width']:.2f}"
        print(f"{r['param']:10s} {r['mle']:9.3g} {r['true']:9.3g} {lo_s:>9s} "
              f"{hi_s:>9s} {rw_s:>7s}  {r['verdict']}", flush=True)
        csv.append(f"{r['param']},{r['mle']:.6g},{r['true']:.6g},"
                   f"{r['ci_low']:.6g},{r['ci_high']:.6g},{r['rel_width']:.6g},"
                   f"{r['verdict']}")

    n_ident = sum(r["verdict"] == "IDENT" for r in rows)
    n_weak = sum(r["verdict"] == "WEAK" for r in rows)
    n_non = sum(r["verdict"] == "NON-IDENT" for r in rows)
    print("-" * 68, flush=True)
    print(f"IDENT (tight CI): {n_ident}   WEAK (wide CI): {n_weak}   "
          f"NON-IDENT (unbounded): {n_non}   / {len(UNKNOWN)}", flush=True)

    with open(os.path.join(out_dir, f"{safe}_profile_ci.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")
    with open(os.path.join(out_dir, f"{safe}_profile_ci.json"), "w") as fh:
        json.dump({"regime": regime, "prob": PROB, "points": POINTS,
                   "ci": {r["param"]: {kk: r[kk] for kk in
                          ("mle", "true", "ci_low", "ci_high", "rel_width", "verdict")}
                          for r in rows}}, fh, indent=2, default=str)

    try:
        ci.plot_all_ci()
        plt.savefig(os.path.join(out_dir, f"{safe}_profile_likelihood.png"), dpi=110)
        print("profile_likelihood.png written", flush=True)
    except Exception as e:
        print(f"(plot skipped: {e})", flush=True)

    print(f"\nWrote {safe}_profile_ci.csv / .json into {out_dir}", flush=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    reg = sys.argv[2] if len(sys.argv) > 2 else "Normal"
    main(out, reg)
