"""A/B two edge screens -- e.g. with and without the depletion conditions.

The screens must have been run over the same edges, regimes and
parameterisation; only the condition set (or whatever else is under test)
differs. Everything is reported per edge because the prediction is
DIFFERENTIATED, not global: `raKO` takes r to 0 exactly, so the RA-driven edges
should gain a lot, while the HOXA13->beta-catenin feedback holds b off zero no
matter how hard WNT is knocked down, so the b-driven edges should gain less.

usage: python3 compare_screens.py <A_label>=<dirA> <B_label>=<dirB>
"""
import json
import os
import sys

import numpy as np

ORDER = ["Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss"]


def load(spec):
    label, path = spec.split("=", 1)
    with open(os.path.join(path, "screen.json")) as fh:
        return label, {(r["term"], r["param"], r["regime"]): r
                       for r in json.load(fh)}


def main(specs):
    arms = [load(s) for s in specs]
    keys = sorted(set(arms[0][1]) & set(arms[1][1]))
    if not keys:
        print("no shared (term, param, regime) rows between the two screens")
        return
    (la, A), (lb, B) = arms

    print(f"\nA = {la}   B = {lb}\n")
    hdr = (f"{'edge':<9s} {'regime':<17s} {'param':<7s} "
           f"{'fNRMSE A':>9s} {'B':>8s} {'delta':>8s}   "
           f"{'basal A':>8s} {'B':>8s} {'delta':>8s}   {'eq A':>5s} {'B':>4s}")
    print(hdr)
    print("-" * len(hdr))
    agg = {}
    for term, param, regime in keys:
        a, b = A[(term, param, regime)], B[(term, param, regime)]
        dn = b["term_nrmse"] - a["term_nrmse"]
        ba, bb = a.get("basal_err"), b.get("basal_err")
        db = (bb - ba) if (ba is not None and bb is not None) else None
        fmt = lambda v: f"{100*v:7.1f}%" if v is not None else f"{'-':>8s}"
        print(f"{term:<9s} {regime:<17s} {param:<7s} "
              f"{100*a['term_nrmse']:8.1f}% {100*b['term_nrmse']:7.1f}% "
              f"{100*dn:+7.1f}%   {fmt(ba)} {fmt(bb)} "
              f"{(f'{100*db:+7.1f}%' if db is not None else '       -'):>8s}   "
              f"{a['eq_under10']:>2d}/{a['n_eq_params']:<2d} "
              f"{b['eq_under10']:>2d}/{b['n_eq_params']:<2d}")
        agg.setdefault(term, []).append((a, b))

    print(f"\nper-edge means over the regimes screened:")
    print(f"{'edge':<9s} {'fNRMSE A':>9s} {'B':>8s} {'delta':>8s}   "
          f"{'basal A':>8s} {'B':>8s} {'delta':>8s}   "
          f"{'eq params A':>12s} {'B':>5s}")
    for term, pairs in agg.items():
        na = np.mean([p[0]["term_nrmse"] for p in pairs])
        nb = np.mean([p[1]["term_nrmse"] for p in pairs])
        ba = [p[0].get("basal_err") for p in pairs if p[0].get("basal_err")]
        bb = [p[1].get("basal_err") for p in pairs if p[1].get("basal_err")]
        ea = sum(p[0]["eq_under10"] for p in pairs)
        eb = sum(p[1]["eq_under10"] for p in pairs)
        tot = sum(p[0]["n_eq_params"] for p in pairs)
        bam, bbm = (np.mean(ba) if ba else None), (np.mean(bb) if bb else None)
        fmt = lambda v: f"{100*v:7.1f}%" if v is not None else f"{'-':>8s}"
        d = f"{100*(bbm-bam):+7.1f}%" if (bam is not None
                                          and bbm is not None) else "       -"
        print(f"{term:<9s} {100*na:8.1f}% {100*nb:7.1f}% {100*(nb-na):+7.1f}%   "
              f"{fmt(bam)} {fmt(bbm)} {d:>8s}   "
              f"{ea:>9d}/{tot:<2d} {eb:>2d}/{tot:<2d}")


if __name__ == "__main__":
    main(sys.argv[1:])
