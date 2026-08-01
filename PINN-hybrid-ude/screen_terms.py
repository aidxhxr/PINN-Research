"""Equation-local screen: how learnable is EVERY regulatory edge in the model?

The full hybrid pipeline (`run_hybrid.sh`) costs ~2 GPU-hours per variant
because it trains 10 state networks per regime on top of the mechanism. That is
the right instrument for the headline claim, but it cannot be run 13 times x 3
parameterisations. This screen is the cheap one.

For one edge it fits ONLY the equation that edge lives in:

    eps_x * dx/dt = ... + f_NN(u) + ...          (trapezoidal, derivative-free)

using the reference trajectories directly, with every parameter OUTSIDE that
equation held at its true value and every parameter INSIDE it (except the ones
the network absorbed) started from the same deliberately-wrong 1.5x guess the
inverse PINN uses. It therefore measures two things per edge, cleanly separated
(Loman & Baker, arXiv:2510.14140):

  * FUNCTIONAL identifiability -- NRMSE of the learned f against the closed
    form it replaced, over the support the trajectories actually visit;
  * LOCAL PARAMETRIC damage -- how the surviving mechanistic parameters of that
    same equation are recovered, versus a mechanistic control fit of the same
    equation with no network in it at all.

HONESTY: exact reference states go in, so state-estimation error is absent and
these numbers are an UPPER BOUND on what the full pipeline can achieve. The
control column is fit under exactly the same conditions, so the hybrid-minus-
control delta -- which is the actual quantity of interest -- stays fair.

usage:
    python3 screen_terms.py --out runs/<ts>_screen [--terms a,b] [--params gated,sc]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from config import (BASELINE, REGIMES, CONDITIONS, DEVICE, FIXED,
                    HYBRID_TERMS, INIT_GUESS, NOMINAL, PARAM_RANGE)
from hybrid import build_one_term, observed_points, true_term
from model import InverseParams
from residual import physics_rhs

# state row each equation writes, and every BASELINE parameter that appears in
# that equation (the screen holds everything else at truth).
EQ_ROW = {"db": 0, "dapc": 1, "dh5": 2, "dh13": 3, "dm": 4, "dr": 5, "dc": 6}
EQ_PARAMS = {
    "db":   ["W", "eta13", "kappa13", "lambdaP", "lambda5", "kappa5"],
    "dapc": ["epsP", "rho5", "rhoB", "rho13", "deltaP1", "thetaP"],
    "dh5":  ["eps5", "a5", "etaR", "kappaR", "etaM", "kappaM"],
    "dh13": ["eps13", "a13", "etaB13", "kappaB13", "etaM13", "kappaM13"],
    "dm":   ["epsM", "aM", "etaBM", "kappaBM"],
    "dr":   ["epsR", "lambdaC"],
    "dc":   ["epsC", "aC", "etaRC", "kappaRC", "etaBC", "kappaBC"],
}


def load_refs(path, T=150.0):
    if path and os.path.exists(path):
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)
    from reference import generate_references
    return generate_references(T=T)


def make_tensors(refs_for_regime, stride, noise_std, seed):
    """Reference states on a fixed grid, per condition, as torch tensors."""
    rng = np.random.default_rng(seed)
    out = []
    for cond in CONDITIONS:
        t_ref, y_ref = refs_for_regime[cond["name"]]
        t = t_ref[::stride].copy()
        y = y_ref[::stride].copy()
        if noise_std > 0:
            y = y + rng.normal(0.0, noise_std, size=y.shape)
        scale = np.maximum(np.abs(y_ref).max(axis=0), 0.05)
        out.append(dict(
            name=cond["name"], forcing=cond["forcing"],
            t=torch.tensor(t, device=DEVICE).reshape(-1, 1),
            y=torch.tensor(y, device=DEVICE),
            scale=torch.tensor(scale, device=DEVICE).reshape(1, 7)))
    return out


def eq_residual(conds, row, params_fn, nn_terms):
    """Mean squared normalised trapezoidal residual of ONE equation."""
    total = 0.0
    for c in conds:
        p = params_fn(c["forcing"])
        f = physics_rhs(c["t"], c["y"], p, nn_terms)[:, row:row + 1]
        z = c["y"][:, row:row + 1]
        dt = c["t"][1:] - c["t"][:-1]
        r = ((z[1:] - z[:-1]) - 0.5 * dt * (f[1:] + f[:-1])) / c["scale"][0, row]
        total = total + (r ** 2).mean()
    return total / len(conds)


def fit_equation(eq, term, regime, refs_for_regime, conds, *, param, seed,
                 adam=1500, lbfgs=200, wd=1e-8, lr=5e-3):
    """Fit one equation: its own mechanistic parameters + (optionally) the net.

    `term=None` is the mechanistic control for that same equation -- the full
    closed form, all of its parameters unknown.
    """
    spec = HYBRID_TERMS[term] if term else None
    row = EQ_ROW[eq]
    replaced = set(spec["replaces"]) if term else set()

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    keys = [k for k in EQ_PARAMS[eq] if k not in FIXED and k not in replaced]
    init = {}
    for k in keys:
        j = float(np.exp(rng.normal(0.0, 0.05)))
        init[k] = (float(np.clip(INIT_GUESS[k] * j, 0.05, 0.95))
                   if k == "thetaP" else INIT_GUESS[k] * j)
    params = InverseParams(init, {k: PARAM_RANGE[k] for k in keys},
                           {k: NOMINAL[k] for k in keys}).to(DEVICE)

    # net built AFTER the parameters so the control and the hybrid draw the
    # same parameter jitter from the same seed (the RNG-desync trap).
    nn_terms = ({term: build_one_term(term, refs_for_regime, DEVICE,
                                      param=param)}
                if term else {})
    net_params = [p for n in nn_terms.values() for p in n.parameters()]
    true_vals = {**BASELINE, **REGIMES[regime]}

    def params_fn(forcing):
        return {**true_vals, **params.dict(), **forcing}

    def loss_fn():
        L = eq_residual(conds, row, params_fn, nn_terms)
        if net_params:
            L = L + wd * sum(n.l2() for n in nn_terms.values())
        return L

    opt = torch.optim.Adam([{"params": params.parameters(), "lr": lr}]
                           + ([{"params": net_params, "lr": lr}]
                              if net_params else []))
    for ep in range(adam):
        opt.zero_grad(set_to_none=True)
        L = loss_fn()
        L.backward()
        opt.step()

    lb = torch.optim.LBFGS(list(params.parameters()) + net_params, lr=0.5,
                           max_iter=lbfgs, tolerance_grad=1e-14,
                           tolerance_change=1e-16,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad(set_to_none=True)
        L = loss_fn()
        L.backward()
        return L

    lb.step(closure)
    final = float(eq_residual(conds, row, params_fn, nn_terms).detach())

    est = params.values()
    perr = {k: abs(est[k] - true_vals[k]) / abs(true_vals[k]) for k in keys}
    result = dict(regime=regime, term=term, param=(param if term else "none"),
                  eq=eq, seed=seed, physics=final,
                  n_eq_params=len(keys),
                  eq_under10=sum(1 for v in perr.values() if v < 0.10),
                  param_err={k: float(v) for k, v in perr.items()},
                  recovered={k: float(v) for k, v in est.items()})

    if term:
        x = observed_points(term, refs_for_regime)
        learned = nn_terms[term].curve(x, device=DEVICE)
        truth = true_term(term, x if x.shape[1] > 1 else x[:, 0], true_vals)
        rmse = float(np.sqrt(np.mean((learned - truth) ** 2)))
        denom = float(np.sqrt(np.mean(truth ** 2))) or 1.0
        result.update(term_rmse=rmse, term_nrmse=rmse / denom,
                      x_lo=float(x.min()), x_hi=float(x.max()))
        basal = spec.get("basal")
        if basal and basal in perr:
            result["basal_param"] = basal
            result["basal_err"] = float(perr[basal])
        result["net"] = nn_terms[term]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", default=None)
    ap.add_argument("--terms", default="all")
    ap.add_argument("--params", default="gated,sc,sc_bounded")
    ap.add_argument("--regimes", default="all")
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--adam", type=int, default=1500)
    ap.add_argument("--lbfgs", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    terms = ([t for t in HYBRID_TERMS
              if HYBRID_TERMS[t].get("input_kind") != "parameter"]
             if args.terms == "all" else
             [t.strip() for t in args.terms.split(",") if t.strip()])
    params_list = [p.strip() for p in args.params.split(",") if p.strip()]
    regimes = (list(REGIMES) if args.regimes == "all" else
               [r.strip() for r in args.regimes.split(",")])

    print(f"screen: {len(terms)} terms x {len(regimes)} regimes x "
          f"{len(params_list)} parameterisations, {args.starts} starts "
          f"(noise={args.noise})", flush=True)
    refs = load_refs(args.refs)

    rows = []
    t0 = time.time()
    for regime in regimes:
        conds = make_tensors(refs[regime], args.stride, args.noise, seed=7)

        # mechanistic control for every equation that hosts a screened term:
        # the same fit with the closed form left in place and ALL of that
        # equation's parameters unknown.
        controls = {}
        for eq in sorted({HYBRID_TERMS[t]["eq"] for t in terms}):
            best = None
            for s in range(args.starts):
                r = fit_equation(eq, None, regime, refs[regime], conds,
                                 param="none", seed=100 + 17 * s,
                                 adam=args.adam, lbfgs=args.lbfgs)
                if best is None or r["physics"] < best["physics"]:
                    best = r
            controls[eq] = best
            print(f"  {regime:<18s} {eq:<10s} {'mechanistic':<11s} "
                  f"{' ':>16s}  eq params "
                  f"{best['eq_under10']}/{best['n_eq_params']}"
                  f"  phys={best['physics']:.2e}  [{time.time()-t0:.0f}s]",
                  flush=True)

        for term in terms:
            eq = HYBRID_TERMS[term]["eq"]
            for param in params_list:
                best = None
                for s in range(args.starts):
                    r = fit_equation(eq, term, regime, refs[regime], conds,
                                     param=param, seed=100 + 17 * s,
                                     adam=args.adam, lbfgs=args.lbfgs)
                    if best is None or r["physics"] < best["physics"]:
                        best = r
                ctrl = controls[eq]
                row = {k: v for k, v in best.items()
                       if k not in ("net", "params_obj")}
                row["ctrl_eq_under10"] = ctrl["eq_under10"]
                row["ctrl_n_eq_params"] = ctrl["n_eq_params"]
                row["ctrl_physics"] = ctrl["physics"]
                row["ctrl_param_err"] = ctrl["param_err"]
                if "basal_param" in best:
                    row["ctrl_basal_err"] = ctrl["param_err"].get(
                        best["basal_param"])
                rows.append(row)
                # written after every fit, not just at the end: a 4-regime
                # sweep runs for hours and a partial atlas is still readable
                with open(os.path.join(args.out, "screen.json"), "w") as fh:
                    json.dump(rows, fh, indent=2)
                torch.save(best["net"].state_dict(),
                           os.path.join(args.out,
                                        f"{regime.replace(' ', '_')}_"
                                        f"{term}_{param}.pt"))
                print(f"  {regime:<18s} {term:<10s} {param:<11s} "
                      f"fNRMSE={row['term_nrmse']:6.1%}  "
                      f"eq params {row['eq_under10']}/{row['n_eq_params']} "
                      f"(ctrl {ctrl['eq_under10']}/{ctrl['n_eq_params']})  "
                      f"basal_err="
                      f"{row.get('basal_err', float('nan')):.1%}"
                      f" (ctrl {row.get('ctrl_basal_err', float('nan')):.1%})"
                      f"  [{time.time()-t0:.0f}s]", flush=True)

    with open(os.path.join(args.out, "screen.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'screen.json')}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
