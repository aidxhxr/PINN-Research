"""
SA-big — shared core for the sensitivity analysis of the *big* (27-state)
Wnt / Retinoid / HOX model (`model_big.modelsys_2025_k16dynamic`).

This is the large-system counterpart of the 7-ODE screening in
`PINN/sensitivity_analysis.ipynb`.  It defines:

  * the baseline parameter set (the 27-element reaction-rate vector `p`
    + the cross-pathway coupling knobs),
  * the list of parameters that are varied for SA (+/-30% box),
  * the model outputs we score (biologically meaningful species + a stemness
    surrogate), evaluated as the time-average over a settled window,
  * a parallel `evaluate_matrix` used by Morris / Sobol / local sensitivity.

The state vector returned by the model (column index -> species):

    0 Ro   1 Ra   2 A(ALDH)  3 R(RA)   4 B(CRABP) 5 Br   6 N(RAR) 7 Nr
    8 C(CYP26A1)  9 Cr  10 Dc  11 Dn  12 Bc
    13 V  14 Di  15 Db  16 Bp  17 Da  18 P(APC)  19 Ba(beta-catenin)  20 X
    21 H5(HOXA5)  22 M(MYC)  23 Mi(MIZ1)  24 Ca  25 H13(HOXA13)  26 Ci
    27 Bcat/TCF (derived readout)
"""
import os
import numpy as np

from model_big import modelsys_2025_k16dynamic

# ----------------------------------------------------------------------------
# Solver / horizon settings (see model_big for why these are loosened vs the
# original reference run).  ~35 s per solve.
# ----------------------------------------------------------------------------
SOLVER   = dict(method="BDF", rtol=1e-3, atol=1e-6, t_N=12000.0, N_t=1200)
AVG_LO   = 6000.0     # average outputs over [AVG_LO, t_N]; system has settled
AVG_HI   = SOLVER["t_N"]

# ----------------------------------------------------------------------------
# Baseline parameters.
# The 27-element reaction-rate vector `p` (retinoid/RAS + coupling reactions),
# names taken from the MATLAB driver `normasysl_call.m` and the code's own
# unpacking in model_big (rkp1=p[0] ... MCsynth=p[26]).
# ----------------------------------------------------------------------------
P_NAMES = [
    "kp1", "km1", "k2", "kp3", "km3", "kp4", "km4", "kp5", "km5",
    "kp6", "km6", "k7", "k8", "kp9", "km9", "k10", "k14", "k15",
    "v16", "v17", "v18", "v19", "k20", "k21", "k22", "k23", "MCsynth",
]
P_BASELINE = {
    "kp1": 0.5, "km1": 0.1, "k2": 0.5, "kp3": 2.13e2, "km3": 1.32e1,
    "kp4": 6.00e9, "km4": 3.60e1, "kp5": 1.2e2, "km5": 4.00e1,
    "kp6": 1.01e1, "km6": 1.01e1, "k7": 1.02e1, "k8": 2.70e1,
    "kp9": 3.60e2, "km9": 3.00e1, "k10": 1.00e-1, "k14": 3.85e-2,
    "k15": 1.73e-3, "v16": 2.00e-1, "v17": 2.00e-1, "v18": 2.00e-2,
    "v19": 2.50e-2, "k20": 3.85e-2, "k21": 1.73e-1, "k22": 8.35e-3,
    "k23": 1.00e-2, "MCsynth": 1.00e-1,
}

# Cross-pathway coupling knobs (passed as scalar arguments to the model).
COUPLING_BASELINE = {
    "gamma1": 1.0,      # scales the dynamic K16 (Bcat/TCF availability)
    "retef":  0.0015,   # RA -> Wnt effect (APC degradation modulation)
    "wntef":  100.0,    # Wnt -> RA effect (CYP synthesis modulation)
}
# Fixed model arguments (NOT varied): characteristic concentrations, the APC
# functionality knob (kept healthy=1 to avoid the discontinuous APCdeg branch),
# and the boolean K8/K17 flags.
FIXED_ARGS = dict(CCret=1000.0, CCwnt=1000.0, funcpercent=1.0,
                  k8log=False, k17log=False)

BASELINE = {**P_BASELINE, **COUPLING_BASELINE}

# Parameters varied for SA: all 27 reaction rates + the 3 coupling knobs = 30.
PARAMS = P_NAMES + list(COUPLING_BASELINE.keys())

# +/-30% box around baseline (matches the 7-ODE screening convention).
VARY_FRAC = 0.30
problem = {
    "num_vars": len(PARAMS),
    "names": PARAMS,
    "bounds": [[BASELINE[k] * (1.0 - VARY_FRAC), BASELINE[k] * (1.0 + VARY_FRAC)]
               for k in PARAMS],
}

# ----------------------------------------------------------------------------
# Outputs: (display name, state column index or None for the derived stemness).
# ----------------------------------------------------------------------------
OUTPUTS = [
    ("RA",        3),
    ("CYP26A1",   8),
    ("APC",      18),
    ("b-catenin", 19),
    ("HOXA5",    21),
    ("MYC",      22),
    ("HOXA13",   25),
    ("Bcat-TCF", 27),
    ("Stemness", None),
]
OUTPUT_NAMES = [n for n, _ in OUTPUTS]


def _build_args(overrides):
    """Assemble (p_vector, kwargs) for the model from a {param: value} dict."""
    p = np.array([overrides.get(k, P_BASELINE[k]) for k in P_NAMES], dtype=float)
    coupling = {k: overrides.get(k, COUPLING_BASELINE[k]) for k in COUPLING_BASELINE}
    return p, coupling


def run_model(overrides):
    """Solve the big model with the given parameter overrides. Returns (t, Sol)
    or None on failure."""
    p, coupling = _build_args(overrides)
    try:
        t, Sol, _ = modelsys_2025_k16dynamic(
            p, coupling["retef"], coupling["wntef"],
            FIXED_ARGS["CCret"], FIXED_ARGS["CCwnt"], FIXED_ARGS["funcpercent"],
            FIXED_ARGS["k8log"], FIXED_ARGS["k17log"], coupling["gamma1"],
            **SOLVER,
        )
        if not np.all(np.isfinite(Sol)):
            return None
        return t, Sol
    except Exception:
        return None


def extract_outputs(t, Sol):
    """Time-average of each scored output over the settled window."""
    mask = (t >= AVG_LO) & (t <= AVG_HI)
    if mask.sum() < 2:
        return np.full(len(OUTPUTS), np.nan)
    tw = t[mask]
    span = tw[-1] - tw[0]
    means = {}
    out = np.empty(len(OUTPUTS))
    for k, (name, idx) in enumerate(OUTPUTS):
        if idx is not None:
            out[k] = np.trapezoid(Sol[mask, idx], tw) / span
            means[name] = out[k]
    # stemness surrogate (mirrors the 7-ODE S = b(1+a13*h13)/(1+apc+a5*h5),
    # alpha13 = alpha5 = 1), computed pointwise then time-averaged.
    Ba, P, H5, H13 = Sol[mask, 19], Sol[mask, 18], Sol[mask, 21], Sol[mask, 25]
    S = Ba * (1.0 + H13) / (1.0 + P + H5)
    out[-1] = np.trapezoid(S, tw) / span
    return out


def baseline_outputs():
    res = run_model({})
    if res is None:
        raise RuntimeError("baseline solve failed")
    return extract_outputs(*res)


def _eval_row(x):
    """Worker: evaluate one SALib sample row -> output vector (top-level so it
    is picklable for multiprocessing)."""
    overrides = {PARAMS[k]: x[k] for k in range(len(PARAMS))}
    res = run_model(overrides)
    if res is None:
        return np.full(len(OUTPUTS), np.nan)
    return extract_outputs(*res)


# SA drivers may override this to cap the worker pool.
DEFAULT_WORKERS = None


def evaluate_matrix(X, n_workers=None, chunk_report=64):
    """Run the model for every row of a SALib sample matrix in parallel.
    Returns Y (n_samples, n_outputs) with failed solves imputed by column mean."""
    import multiprocessing as mp
    n = X.shape[0]
    if n_workers is None:
        n_workers = DEFAULT_WORKERS or min(len(os.sched_getaffinity(0)), 60)
    print(f"  evaluating {n} solves on {n_workers} workers...", flush=True)
    rows = [X[i] for i in range(n)]
    Y = np.full((n, len(OUTPUTS)), np.nan)
    with mp.Pool(n_workers) as pool:
        for i, y in enumerate(pool.imap(_eval_row, rows, chunksize=1)):
            Y[i] = y
            if (i + 1) % chunk_report == 0 or i + 1 == n:
                print(f"    {i + 1}/{n} done", flush=True)
    n_fail = np.isnan(Y).any(axis=1).sum()
    if n_fail:
        print(f"  note: {n_fail}/{n} solves failed -> imputed with column mean",
              flush=True)
        col_mean = np.nanmean(Y, axis=0)
        nan_mask = np.isnan(Y)
        Y[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return Y


def slug(name):
    return name.lower().replace("-", "").replace(" ", "_")
