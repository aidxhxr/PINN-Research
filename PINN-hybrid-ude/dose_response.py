"""Dose-response: is the UDE's basal-parameter bias a function of anchor visitation?

The 2026-08-01 A/B compared two condition sets and found that a learned term's
`f(0) = 0` anchor only protects the hosting equation's basal-production
parameter when the data actually reach the anchor. Two points cannot establish
that the anchor is the governing variable, and the information-matched arm only
excludes the "two more experiments help" alternative.

This runs the graded version. At every dose the condition count is FIXED at
eleven -- ten base conditions plus exactly one depletion whose depth varies --
so no dose is better informed than any other by construction, and `k = 1.0` is
a near-null dose that duplicates an existing condition. Three arms
(`anchor_doses.ARMS`):

    ra     graded retinoid restriction; r reaches its anchor exactly at k = 0
    wnt    graded WNT knockdown alone; b PLATEAUS at 0.013-0.019 because the
           HOXA13 -> beta-catenin feedback keeps producing it
    bcat   the same knockdown on BOTH arms of b's production; the plateau
           breaks and b reaches its anchor exactly

The reported quantity is the EXCESS basal error -- hybrid minus a mechanistic
control fit of the same equation on the same eleven conditions with the same
seeds and budget. Information that helps the equation generally cancels in that
difference; what is left is what the network costs.

Predictions were registered in `runs/20260802_dose_response/PREREGISTERED.md`
before any fit here was run.

HONESTY: like `screen_terms.py` this is the equation-local screen -- exact
reference states go in, so state-estimation error is absent and the numbers are
an upper bound on the full pipeline. Both arms of every comparison are fit
under identical conditions, so the difference stays fair.

usage:
    python3 dose_response.py --arm ra   --edge ra_h5  --out runs/<dir>
    python3 dose_response.py --arm wnt  --edge bm_myc --out runs/<dir>
    python3 dose_response.py --arm bcat --edge bm_myc --out runs/<dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from anchor_doses import (ARM_REGULATOR, ARMS, DOSES, anchor_ratio,
                          dose_conditions, solve_conditions)
from config import DEVICE, HYBRID_TERMS, REGIMES
from screen_terms import EQ_ROW, fit_equation


def make_tensors(refs_for_regime, conds, stride, noise_std, seed):
    """Reference states on a fixed grid, per condition, as torch tensors.

    Mirrors `screen_terms.make_tensors` but takes the condition list explicitly
    -- that module reads the global `config.CONDITIONS`, which is exactly what
    varies here.
    """
    rng = np.random.default_rng(seed)
    out = []
    for cond in conds:
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


def best_of(n_starts, fit):
    best = None
    for s in range(n_starts):
        r = fit(seed=100 + 17 * s)
        if best is None or r["physics"] < best["physics"]:
            best = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--edge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regimes", default="Normal,Severe APC Loss")
    ap.add_argument("--param", default="gated")
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--adam", type=int, default=3000)
    ap.add_argument("--lbfgs", type=int, default=300)
    args = ap.parse_args()

    # `config.HYBRID_TERM` defaults to a real edge, which REMOVES that edge's
    # parameters from UNKNOWN -- and INIT_GUESS/PARAM_RANGE/NOMINAL with it. The
    # screen picks its term per fit instead, so it needs the full parameter set;
    # with the default in place a control fit of the affected equation dies on a
    # KeyError, and a fit of any OTHER equation silently succeeds against a
    # different UNKNOWN than the one the atlas used. Fail loudly rather than let
    # that difference into a comparison.
    if os.environ.get("HYBRID_TERM", "").strip().lower() != "none":
        raise SystemExit(
            "set HYBRID_TERM=none before running the screen: the term is "
            "chosen per fit by --edge, and any other value shrinks UNKNOWN "
            "(see config.py HYBRID_REPLACED).")

    os.makedirs(args.out, exist_ok=True)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    eq = HYBRID_TERMS[args.edge]["eq"]
    basal = HYBRID_TERMS[args.edge].get("basal")
    var = ARM_REGULATOR[args.arm]

    print(f"dose-response  arm={args.arm}  edge={args.edge} (eq {eq}, "
          f"basal {basal}, regulator {var})", flush=True)
    print(f"  {len(DOSES)} doses x {len(regimes)} regimes, "
          f"{args.starts} starts, 11 conditions at every dose "
          f"(noise={args.noise})", flush=True)

    rows = []
    t0 = time.time()
    out_json = os.path.join(args.out, f"dose_{args.arm}_{args.edge}.json")

    for regime in regimes:
        for k in DOSES:
            conds_spec = dose_conditions(args.arm, k)
            refs = solve_conditions(regime, conds_spec)
            ratio, floor = anchor_ratio(refs, var)
            conds = make_tensors(refs, conds_spec, args.stride, args.noise,
                                 seed=7)

            ctrl = best_of(args.starts, lambda seed: fit_equation(
                eq, None, regime, refs, conds, param="none", seed=seed,
                adam=args.adam, lbfgs=args.lbfgs))
            hyb = best_of(args.starts, lambda seed: fit_equation(
                eq, args.edge, regime, refs, conds, param=args.param,
                seed=seed, adam=args.adam, lbfgs=args.lbfgs))

            h_basal = hyb.get("basal_err")
            c_basal = ctrl["param_err"].get(basal) if basal else None
            excess = (None if h_basal is None or c_basal is None
                      else h_basal - c_basal)

            row = dict(arm=args.arm, edge=args.edge, eq=eq, regime=regime,
                       dose=k, anchor_ratio=ratio, anchor_floor=floor,
                       param=args.param, starts=args.starts, noise=args.noise,
                       basal_param=basal,
                       hybrid_basal_err=h_basal, ctrl_basal_err=c_basal,
                       excess_basal_err=excess,
                       term_nrmse=hyb.get("term_nrmse"),
                       hybrid_eq_under10=hyb["eq_under10"],
                       hybrid_n_eq_params=hyb["n_eq_params"],
                       ctrl_eq_under10=ctrl["eq_under10"],
                       ctrl_n_eq_params=ctrl["n_eq_params"],
                       hybrid_physics=hyb["physics"],
                       ctrl_physics=ctrl["physics"],
                       hybrid_param_err=hyb["param_err"],
                       ctrl_param_err=ctrl["param_err"])
            rows.append(row)
            # written after every dose: the sweep runs for hours and a partial
            # curve is already interpretable
            with open(out_json, "w") as fh:
                json.dump(rows, fh, indent=2)
            torch.save(hyb["net"].state_dict(), os.path.join(
                args.out, f"{regime.replace(' ', '_')}_{args.arm}_"
                          f"{args.edge}_k{k:g}.pt"))

            print(f"  {regime:<18s} k={k:<5g} ratio={ratio:7.4f}  "
                  f"fNRMSE={hyb.get('term_nrmse', float('nan')):6.1%}  "
                  f"basal={h_basal:6.1%} (ctrl {c_basal:6.1%})  "
                  f"EXCESS={excess:7.1%}  "
                  f"eq {hyb['eq_under10']}/{hyb['n_eq_params']} "
                  f"(ctrl {ctrl['eq_under10']}/{ctrl['n_eq_params']})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\nwrote {out_json}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
