"""Prospective test: does the training-free reachability table actually work?

`anchor_reach.py` names, for every edge and without training anything, the
cheapest knockdown protocol that puts that edge's `f(0)=0` anchor into the data.
That is a prediction, so it should be tested prospectively on edges whose
outcome is not already known.

The two chosen are the WORST production edges in the atlas -- `m_h13`
(57.2% functional / 12.8% basal in Normal, 11.6% / 95.6% in Severe, 2/8
equation parameters) and `h13_b` (19.3% / 16.2%, 171.4% / 57.5%). Neither has
ever been run under any depletion condition, and the table says both are
fixable.

Each edge is run under THREE protocols, which makes the test self-controlled:

    none          the existing 10 conditions
    <near>        a knockdown that is a LARGER perturbation but does NOT reach
                  this edge's anchor  -- the information-matched control
    <reaching>    the protocol the table prescribes, which does reach it

The middle arm is the point. If the anchor is what matters, the near-miss arm
should behave like `none` however large its perturbation, and only the
prescribed arm should collapse the basal error.

usage:
    python3 protocol_test.py --edge m_h13 --protocols none,bcatKO,mycKO \\
        --out runs/<dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from anchor_doses import anchor_ratio, solve_conditions
from anchor_reach import PROTOCOLS
from config import CONDITIONS, DEVICE, HYBRID_TERMS
from dose_response import best_of, make_tensors
from screen_terms import fit_equation

PROTOCOL_FORCING = dict(PROTOCOLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", required=True)
    ap.add_argument("--protocols", required=True,
                    help="comma-separated, e.g. none,bcatKO,mycKO")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regimes", default="Normal,Severe APC Loss")
    ap.add_argument("--param", default="gated")
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--adam", type=int, default=3000)
    ap.add_argument("--lbfgs", type=int, default=300)
    args = ap.parse_args()

    if os.environ.get("HYBRID_TERM", "").strip().lower() != "none":
        raise SystemExit("set HYBRID_TERM=none (see dose_response.py)")

    os.makedirs(args.out, exist_ok=True)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    protos = [p.strip() for p in args.protocols.split(",") if p.strip()]
    for p in protos:
        if p not in PROTOCOL_FORCING:
            raise SystemExit(f"unknown protocol {p!r}; "
                             f"have {sorted(PROTOCOL_FORCING)}")

    spec = HYBRID_TERMS[args.edge]
    eq, basal, var = spec["eq"], spec.get("basal"), spec["inputs"][0]
    print(f"protocol test  edge={args.edge} (eq {eq}, basal {basal}, "
          f"regulator {var})", flush=True)
    print(f"  protocols: {', '.join(protos)}   "
          f"{len(regimes)} regimes, {args.starts} starts", flush=True)

    rows, t0 = [], time.time()
    out_json = os.path.join(args.out, f"protocol_{args.edge}.json")

    for regime in regimes:
        for proto in protos:
            forcing = PROTOCOL_FORCING[proto]
            conds_spec = (CONDITIONS if forcing is None else
                          CONDITIONS + [{"name": proto, "forcing": forcing}])
            refs = solve_conditions(regime, conds_spec)
            ratio, floor = anchor_ratio(refs, var)
            conds = make_tensors(refs, conds_spec, args.stride, 0.0, seed=7)

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
            row = dict(edge=args.edge, eq=eq, regime=regime, protocol=proto,
                       n_conditions=len(conds_spec), anchor_ratio=ratio,
                       anchor_floor=floor, basal_param=basal,
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
            with open(out_json, "w") as fh:
                json.dump(rows, fh, indent=2)
            torch.save(hyb["net"].state_dict(), os.path.join(
                args.out, f"{regime.replace(' ', '_')}_{args.edge}_"
                          f"{proto}.pt"))

            bs = "  n/a " if h_basal is None else f"{h_basal:6.1%}"
            cs = "  n/a " if c_basal is None else f"{c_basal:6.1%}"
            print(f"  {regime:<18s} {proto:<9s} ratio={ratio:7.4f}  "
                  f"fNRMSE={hyb.get('term_nrmse', float('nan')):7.1%}  "
                  f"basal={bs} (ctrl {cs})  "
                  f"eq {hyb['eq_under10']}/{hyb['n_eq_params']} "
                  f"(ctrl {ctrl['eq_under10']}/{ctrl['n_eq_params']})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\nwrote {out_json}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
