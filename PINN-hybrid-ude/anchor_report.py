"""Pre-flight check: is each learnable edge's f(0)=0 anchor actually OBSERVED?

Every mechanism network in `hybrid.py` is built with f(anchor)=0, and that is
the constraint meant to stop it absorbing a constant out of the equation's
basal-production parameter. The A/B run on 2026-08-01 showed the constraint
only does that job where the data reach the anchor:

    r -> 0 exactly (raKO)   =>  RA edges: functional 0.1%, basal 0.1-1.1%
    b -> 0.02     (wntKO)   =>  b edges:  functional 0.6-3.1%, basal 2.9-25.5%
    nothing near 0          =>  basal 10-79%

This script reports the number that predicts which case an edge is in, and it
needs no training -- only reference trajectories. Run it before choosing what
to hybridise, and again after adding a condition, to check the condition
actually moved the anchor.

usage:
    python3 anchor_report.py [refs.pkl]
    HYBRID_DEPLETION=1 python3 anchor_report.py     # with the KO conditions
"""
import os
import pickle
import sys

import numpy as np

from config import HYBRID_TERMS, REGIMES
from hybrid import VAR_INDEX

ORDER = list(REGIMES)


def main(refs_path=None):
    if refs_path and os.path.exists(refs_path):
        with open(refs_path, "rb") as fh:
            refs = pickle.load(fh)
    else:
        from reference import generate_references
        refs = generate_references()

    terms = [t for t, s in HYBRID_TERMS.items()
             if s.get("input_kind") != "parameter"]
    n_cond = len(next(iter(refs[ORDER[0]].values())) and refs[ORDER[0]])
    print(f"\nanchor visitation over {n_cond} conditions "
          f"(x_lo / x_hi per regime; 0.00 = the anchor is observed)\n")
    print(f"{'edge':<10s} {'regulator':<10s} " +
          " ".join(f"{r[:12]:>13s}" for r in ORDER) + "   verdict")

    for term in terms:
        spec = HYBRID_TERMS[term]
        if spec.get("anchor") is None:
            continue
        for k, var in enumerate(spec["inputs"]):
            idx = VAR_INDEX[var]
            ratios = []
            for regime in ORDER:
                lo = min(float(y[:, idx].min()) for _t, y in refs[regime].values())
                hi = max(float(y[:, idx].max()) for _t, y in refs[regime].values())
                ratios.append(max(lo, 0.0) / max(hi, 1e-12))
            worst = max(ratios)
            verdict = ("anchor OBSERVED" if worst < 0.01 else
                       "partial" if worst < 0.05 else
                       "anchor NEVER visited")
            name = term if k == 0 else ""
            print(f"{name:<10s} {var:<10s} " +
                  " ".join(f"{v:13.3f}" for v in ratios) + f"   {verdict}")

    print("\n  < 0.01  the basal-production parameter of that equation is "
          "pinned by data\n"
          "  < 0.05  partial: expect a few percent of basal error\n"
          "  >= 0.05 f(0)=0 is asserted where no data live; expect the "
          "network to\n"
          "          absorb a constant out of the basal parameter (10-80%)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
