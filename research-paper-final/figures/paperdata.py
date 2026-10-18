"""Loaders for the estimator comparison (Figs 4-6, Table 2).

Every estimator is scored on the same benchmark data (bench.py; data seeds 42
and 7). Returned per (estimator, seed, regime): dict param -> recovered value,
or None if the run does not exist. Conventions:
  * PINN rungs are single-start: autodiff (`base`), weighted (`plumb`) from the
    2026-10-17 ablation, integral = start 0 of the 2026-10-02 integral runs;
  * SGN = schedule chosen by the truth-free shooting misfit (reselect.json);
  * MLE = relative-weighted trust-region fit from the common +50% start;
    MLE-ms = lowest final cost over that start and three jittered starts.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import norm

import style as st

SENS = os.path.join(st.REPO, "experiments/17-sensitivity-pinn")
RUN = os.path.join(SENS, "runs", "20261017_paper")
sys.path.insert(0, os.path.join(st.REPO, "src"))
sys.path.insert(0, SENS)
import bench  # noqa: E402
from wnt_pinn.model.parameters import UNKNOWN  # noqa: E402

REGULAR = [k for k in UNKNOWN if k not in ("thetaP", "deltaP1")]
SAFE = {r: r.replace(" ", "_") for r in st.REGIMES}
ESTIMATORS = ["PINN (autodiff)", "PINN (weighted)", "PINN (integral)",
              "SGN-PINN", "MLE", "MLE (4 starts)"]
EST_COLOR = {"PINN (autodiff)": "#c6dbef", "PINN (weighted)": "#9ecae1",
             "PINN (integral)": "#6baed6", "SGN-PINN": "#2171b5",
             "MLE": "#08519c", "MLE (4 starts)": "#08306b"}


def _json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def pinn_rung(rung, seed, regime):
    if rung == "integral":
        p = os.path.join(SENS, "runs", f"20261002_integral_seed{seed}", f"{SAFE[regime]}_starts.json")
        s = _json(p)
        if s is None:
            return None
        return next(x["recovered"] for x in s["starts"] if x["start"] == 0)
    p = os.path.join(RUN, f"ablation_seed{seed}", rung, f"{SAFE[regime]}_recovered.json")
    s = _json(p)
    return s["recovered"] if s else None


def sgn(seed, regime):
    d = os.path.join(RUN, f"sgn_seed{seed}")
    sel = _json(os.path.join(d, "reselect.json"))
    if sel and regime in sel:
        tag = sel[regime]["chosen"]
    else:  # fall back to the lower shooting cost is not available: S1
        tag = "_S1"
    r = _json(os.path.join(d, f"{SAFE[regime]}_pinn{tag}.json"))
    return r


def mle(seed, regime, multistart=False, weights="rel"):
    """Prefer the tight-tolerance fits (ms_tight: LSODA rtol 1e-11, FD step
    1e-5) when the full set exists; the default-tolerance fits (ms) stop short
    along the flattest directions in the high-WNT regimes."""
    tag = f"_s{seed}_{weights}"
    for d in ("ms_tight", "ms"):
        fs = [os.path.join(RUN, d, f"{SAFE[regime]}_init{tag}.json")]
        if multistart:
            fs += [os.path.join(RUN, d, f"{SAFE[regime]}_init_j{j}{tag}.json") for j in (1, 2, 3)]
        rs = [r for r in (_json(f) for f in fs) if r is not None]
        if rs and len(rs) == len(fs):
            return min(rs, key=lambda r: r["cost"])
    return None


def recovered(est, seed, regime):
    if est == "PINN (autodiff)":
        return pinn_rung("base", seed, regime)
    if est == "PINN (weighted)":
        return pinn_rung("plumb", seed, regime)
    if est == "PINN (integral)":
        return pinn_rung("integral", seed, regime)
    if est == "SGN-PINN":
        r = sgn(seed, regime)
        return r["recovered"] if r else None
    if est == "MLE":
        r = mle(seed, regime)
        return r["recovered"] if r else None
    if est == "MLE (4 starts)":
        r = mle(seed, regime, multistart=True)
        return r["recovered"] if r else None
    raise KeyError(est)


def log_errors(rec, regime):
    """signed log-error per parameter (all 36)."""
    tv = bench.true_params(regime)
    return {k: float(np.log(rec[k] / tv[k])) for k in UNKNOWN}


def count(rec, regime, keys=None, tol=0.10):
    tv = bench.true_params(regime)
    keys = keys or UNKNOWN
    return int(sum(abs(rec[k] - tv[k]) / abs(tv[k]) < tol for k in keys))


def crlb(regime):
    f = json.load(open(os.path.join(RUN, "fim_seed42", "fim_summary.json")))
    return f[regime]["sd35_none"]


def crlb_expected(regime):
    sd = np.array([crlb(regime)[k] for k in REGULAR])
    return float((norm.cdf(np.log(1.1) / sd) - norm.cdf(np.log(0.9) / sd)).sum())
