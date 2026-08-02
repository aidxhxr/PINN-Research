"""What anchor floor does each depletion DOSE actually achieve? (no training)

The 2026-08-01 A/B compared two points -- 10 conditions versus 10 + two
knockouts -- and found the basal-production error of a UDE collapses when the
regulator reaches its f(0)=0 anchor and does not when it only approaches it.
Two points cannot separate "the anchor matters" from "these two experiments
happen to help", and the information-matched arm only rules out one alternative.

A DOSE-RESPONSE does separate them, and it removes the confound structurally
rather than by a control arm: hold the number of conditions FIXED at eleven and
vary only the depth of a single graded depletion. Every dose sees the same
amount of data; k = 1.0 is a near-null dose (it duplicates an existing
condition), so the series has its own internal zero.

This module defines the dose grids and reports the anchor ratio x_lo/x_hi each
one reaches -- computable from reference trajectories alone, which is what makes
the prediction registrable BEFORE the fits are run.

usage:  python3 anchor_doses.py
"""
from __future__ import annotations

import numpy as np

from config import BASELINE, CONDITIONS, REGIMES, Y0

# The graded depletion arms. Each entry maps a dose k in [0, 1] to the forcing
# of ONE extra condition appended to the base ten.
#
#   ra      graded retinoid restriction. r has no production floor once the
#           dietary/circadian/ATRA inputs are scaled away, so r -> 0 as k -> 0.
#           This is the arm the A/B showed working; the prediction is a smooth
#           monotone curve, of which the A/B saw only the endpoints.
#   wnt     graded WNT knockdown, `kW` alone -- exactly the existing `wntKO`
#           at k = 0.02. The HOXA13 -> beta-catenin feedback keeps producing b,
#           so the b floor should PLATEAU well above zero however hard WNT is
#           knocked down. Prediction: the basal error plateaus with it.
#   bcat    the same knockdown applied to BOTH arms of b's production, WNT and
#           the HOXA13 feedback. This is the experiment the A/B write-up named
#           as the next test: if the plateau in `wnt` is caused by the feedback,
#           removing the feedback must break it.
ARMS = {
    "ra":   lambda k: {"mu0": BASELINE["mu0"] * k,
                       "AR": BASELINE["AR"] * k, "DR": 0.0},
    "wnt":  lambda k: {"kW": k},
    "bcat": lambda k: {"kW": k, "k13": k},
}
DOSES = [1.0, 0.3, 0.1, 0.03, 0.01, 0.0]

# which regulator each arm is meant to deplete, and the edges scored on it
ARM_REGULATOR = {"ra": "r", "wnt": "b", "bcat": "b"}


def dose_conditions(arm, k):
    """The base ten conditions plus ONE graded depletion. Count is constant
    across doses, so no dose has more data than any other."""
    return CONDITIONS + [{"name": f"{arm}Dep", "forcing": ARMS[arm](k)}]


def solve_conditions(regime, conds, T=150.0, n_pts=5000):
    from scipy.integrate import solve_ivp
    from odes import _ode_rhs
    out = {}
    for c in conds:
        p = {**BASELINE, **REGIMES[regime], **c["forcing"]}
        sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0,
                        t_eval=np.linspace(0, T, n_pts),
                        method="Radau", rtol=1e-10, atol=1e-12)
        assert sol.success, f"{regime}/{c['name']}: {sol.message}"
        out[c["name"]] = (sol.t, sol.y.T)
    return out


def anchor_ratio(refs_for_regime, var):
    """x_lo / x_hi for one state over a condition set: the fraction of the
    regulator's observed range that separates the data from the anchor.

    Uses the registry's own state naming (`hybrid.VAR_INDEX`, where the APC
    state is `apc`) so an edge's `inputs` entry indexes correctly.
    """
    from hybrid import VAR_INDEX
    idx = VAR_INDEX[var]
    lo = min(float(y[:, idx].min()) for _t, y in refs_for_regime.values())
    hi = max(float(y[:, idx].max()) for _t, y in refs_for_regime.values())
    return max(lo, 0.0) / max(hi, 1e-12), max(lo, 0.0)


def main():
    regimes = ["Normal", "Severe APC Loss"]
    print("\nachieved anchor ratio x_lo/x_hi per dose "
          "(11 conditions at every dose)\n")
    for arm, var in ARM_REGULATOR.items():
        print(f"  arm {arm!r}  regulator {var!r}")
        print(f"    {'dose k':>8s} " +
              " ".join(f"{r[:16]:>26s}" for r in regimes))
        for k in DOSES:
            cells = []
            for regime in regimes:
                refs = solve_conditions(regime, dose_conditions(arm, k))
                ratio, floor = anchor_ratio(refs, var)
                cells.append(f"{ratio:12.4f} (floor {floor:.4f})")
            print(f"    {k:8.3f} " + " ".join(f"{c:>26s}" for c in cells))
        print()


if __name__ == "__main__":
    main()
