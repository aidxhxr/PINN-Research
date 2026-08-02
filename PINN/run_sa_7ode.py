#!/usr/bin/env python3
"""
Standalone reproduction of PINN/sensitivity_analysis.ipynb -> CSV tables.

The notebook stores only PNG figures; this script runs the identical model and
analysis (verbatim model / QoI / sampler settings) and writes numeric tables to
PINN/sa_results/ so the numbers can be cited in a paper.

Two parameter boxes are run by default:

  original_box/   -- the notebook's +/-30% box on all 14 parameters.
  clipped_thetap/ -- identical, except the thetaP upper bound is clipped
                     (default 1.0).

Why the clip: deltaP = 1 + deltaP1*(1 - thetaP) with deltaP1 = 3.5 goes NEGATIVE
for thetaP > 1/3.5 + 1 = 1.285714..., so the APC decay term reverses sign and
the APC trajectory diverges (AUC ~ 6.4e3 at thetaP = 1.30 vs 0.47 at baseline).
The notebook's box reaches thetaP = 1.30, so the Morris and Sobol results for
the APC output are corrupted there (mu* becomes O(1e3); sum of S1 exceeds 1).
The clipped box removes the artifact; both are written so the paper can report
the corrected numbers and document the artifact.

Usage
-----
    python3 run_sa_7ode.py                    # both boxes, Morris + Sobol
    python3 run_sa_7ode.py --skip-sobol       # Morris + local only
    python3 run_sa_7ode.py --box original     # one box only
    python3 run_sa_7ode.py --thetap-max 1.2   # different clip

The notebook evaluated the sample matrices serially; this script keeps that
(≈0.05 s/solve, ≈4 min per box) and prints progress.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
from scipy.integrate import solve_ivp

from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze


# ============================================================
# Model (7-variable nondimensional Wnt / RA system) + stemness
# -- copied verbatim from PINN/sensitivity_analysis.ipynb, cell 1
# ============================================================
def hill(x, K, n=1):
    x = max(x, 0.0)
    return x ** n / (K ** n + x ** n)


def ra_input(tau, p):
    dietary = p["AR"] * (1.0 + np.cos((2.0 * np.pi * tau / p["TR"]) - p["phi"]))
    treatment = 0.5 * p["DR"] * (
        np.tanh(p["q"] * (tau - p["tau1"])) - np.tanh(p["q"] * (tau - p["tau2"]))
    )
    return p["mu0"] + dietary + treatment


def model(tau, y, p):
    b, apc, h5, h13, m, r, c = np.maximum(y, 0.0)
    deltaP = 1.0 + p["deltaP1"] * (1.0 - p["thetaP"])
    muR = ra_input(tau, p)
    db = (p["W"] + p["eta13"] * hill(h13, p["kappa13"], p["nH"]) - b
          - p["lambdaP"] * apc * b - p["lambda5"] * h5 * b / (p["kappa5"] + b))
    dapc = (1.0 / p["epsP"]) * ((1.0 + p["rho5"] * h5)
            / (1.0 + p["rhoB"] * b + p["rho13"] * h13) - deltaP * apc)
    dh5 = (1.0 / p["eps5"]) * (p["a5"] + p["etaR"] * hill(r, p["kappaR"], 1)
            - h5 - p["etaM"] * m * h5 / (p["kappaM"] + m))
    dh13 = (1.0 / p["eps13"]) * (p["a13"] + p["etaB13"] * hill(b, p["kappaB13"], p["nB"])
            + p["etaM13"] * hill(m, p["kappaM13"], p["nM"]) - h13)
    dm = (1.0 / p["epsM"]) * (p["aM"] + p["etaBM"] * hill(b, p["kappaBM"], p["nB"]) - m)
    dr = (1.0 / p["epsR"]) * (muR - r - p["lambdaC"] * c * r)
    dc = (1.0 / p["epsC"]) * (p["aC"] + p["etaRC"] * hill(r, p["kappaRC"], 1)
            + p["etaBC"] * hill(b, p["kappaBC"], p["nB"]) - c)
    return [db, dapc, dh5, dh13, dm, dr, dc]


BASELINE = {
    "W": 0.80, "thetaP": 1.00, "nB": 2, "nM": 2, "nH": 2,
    "eta13": 0.75, "kappa13": 0.55, "lambdaP": 1.60, "lambda5": 1.30, "kappa5": 0.50,
    "epsP": 1.00, "rho5": 1.10, "rhoB": 1.10, "rho13": 1.30, "deltaP1": 3.50,
    "eps5": 1.20, "a5": 0.15, "etaR": 2.50, "kappaR": 0.40, "etaM": 2.50, "kappaM": 0.50,
    "eps13": 1.00, "a13": 0.18, "etaB13": 0.95, "kappaB13": 0.50, "etaM13": 0.55, "kappaM13": 0.50,
    "epsM": 0.60, "aM": 0.18, "etaBM": 1.35, "kappaBM": 0.50,
    "epsR": 0.40, "lambdaC": 0.85,
    "epsC": 0.80, "aC": 0.08, "etaRC": 1.50, "kappaRC": 0.50, "etaBC": 1.50, "kappaBC": 0.50,
    "mu0": 0.35, "AR": 0.04, "TR": 24.0, "phi": 0.0,
    "DR": 1.50, "q": 0.30, "tau1": 40.0, "tau2": 80.0,
    "alpha13": 1.00, "alpha5": 1.00,
}

# Curated, biologically meaningful knobs (same set used for the approved Sobol plot)
PARAMS = ["thetaP", "DR", "AR", "etaR", "etaM", "eta13", "etaB13",
          "etaBM", "etaBC", "lambdaC", "lambdaP", "lambda5", "rhoB", "rho13"]

OUTPUT_NAMES = ["b-catenin", "APC", "HOXA5", "HOXA13", "MYC", "RA", "CYP26A1", "Stemness"]

Y0 = [0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40]
TAU_SPAN = (0.0, 150.0)

N_EVAL = 600
SOLVER = dict(method="LSODA", rtol=1e-7, atol=1e-9)

# analysis settings (notebook cells 2-4)
REL_STEP = 0.05
N_MORRIS = 40
NUM_LEVELS = 4
N_SOBOL = 256
CALC_SECOND_ORDER = False
SEED = 42


def auc_outputs(p, n_eval=N_EVAL):
    """Solve the ODE system and return the AUC (time-average) of every output."""
    tau = np.linspace(*TAU_SPAN, n_eval)
    try:
        sol = solve_ivp(lambda t, y: model(t, y, p), TAU_SPAN, Y0, t_eval=tau,
                        method=SOLVER["method"], rtol=SOLVER["rtol"], atol=SOLVER["atol"])
        if not sol.success:
            return np.full(len(OUTPUT_NAMES), np.nan)
    except Exception:
        return np.full(len(OUTPUT_NAMES), np.nan)
    b, apc, h5, h13 = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    S = b * (1.0 + p["alpha13"] * h13) / (1.0 + apc + p["alpha5"] * h5)
    series = list(sol.y) + [S]
    span = TAU_SPAN[1] - TAU_SPAN[0]
    return np.array([np.trapezoid(s, sol.t) / span for s in series])


def evaluate_matrix(X, tag=""):
    """Run the model for every row of a SALib sample matrix -> Y (n_samples, n_outputs)."""
    n = X.shape[0]
    Y = np.full((n, len(OUTPUT_NAMES)), np.nan)
    t0 = time.time()
    for i in range(n):
        if i % 500 == 0:
            print(f"  {tag}{i}/{n}  ({time.time() - t0:.0f}s)", flush=True)
        p = BASELINE.copy()
        for k, key in enumerate(PARAMS):
            p[key] = X[i, k]
        Y[i] = auc_outputs(p)
    # impute any failed solves with the column mean so SALib gets finite input
    n_fail = int(np.isnan(Y).any(axis=1).sum())
    if n_fail:
        print(f"  note: {n_fail} failed solves imputed with column mean", flush=True)
    col_mean = np.nanmean(Y, axis=0)
    nan_mask = np.isnan(Y)
    Y[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    print(f"  {tag}done {n}/{n} in {time.time() - t0:.0f}s "
          f"({n_fail} failed solves)", flush=True)
    return Y, n_fail


# ------------------------------------------------------------------
# analyses
# ------------------------------------------------------------------
def local_sensitivity(rel_step=REL_STEP):
    """Central-difference normalized elasticity (notebook cell 2)."""
    base = auc_outputs(BASELINE)
    S = np.zeros((len(OUTPUT_NAMES), len(PARAMS)))
    for j, key in enumerate(PARAMS):
        d = BASELINE[key] * rel_step
        p_hi, p_lo = BASELINE.copy(), BASELINE.copy()
        p_hi[key], p_lo[key] = BASELINE[key] + d, BASELINE[key] - d
        dydp = (auc_outputs(p_hi) - auc_outputs(p_lo)) / (2.0 * d)
        S[:, j] = dydp * BASELINE[key] / base          # normalized elasticity
    return base, S


def make_problem(thetap_max=None):
    bounds = [[BASELINE[k] * 0.7, BASELINE[k] * 1.3] for k in PARAMS]
    if thetap_max is not None:
        j = PARAMS.index("thetaP")
        bounds[j][1] = min(bounds[j][1], float(thetap_max))
    return {"num_vars": len(PARAMS), "names": list(PARAMS), "bounds": bounds}


# ------------------------------------------------------------------
# CSV writers (no pandas dependency; plain, quoted-free numeric CSV)
# ------------------------------------------------------------------
def _w(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"  wrote {path}", flush=True)


def fmt(x):
    return f"{float(x):.10g}"


def run_box(outdir, thetap_max, skip_sobol):
    os.makedirs(outdir, exist_ok=True)
    problem = make_problem(thetap_max)
    print(f"\n{'=' * 70}\nBOX -> {outdir}", flush=True)
    print(f"  thetaP bounds: {problem['bounds'][0]}", flush=True)

    cfg = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_notebook": "PINN/sensitivity_analysis.ipynb",
        "output_dir": outdir,
        "thetap_max_clip": thetap_max,
        "model": {
            "n_states": 7,
            "state_order": ["b", "apc", "h5", "h13", "m", "r", "c"],
            "Y0": Y0,
            "TAU_SPAN": list(TAU_SPAN),
            "stemness": "S = b*(1+alpha13*h13)/(1+apc+alpha5*h5)",
            "alpha13": BASELINE["alpha13"], "alpha5": BASELINE["alpha5"],
        },
        "qoi": f"time-averaged AUC over {TAU_SPAN}: trapezoid(series, t) / (T1-T0)",
        "solver": {**SOLVER, "n_eval": N_EVAL},
        "outputs": OUTPUT_NAMES,
        "parameters": [
            {"name": k, "baseline": BASELINE[k],
             "lower": problem["bounds"][j][0], "upper": problem["bounds"][j][1],
             "box": "+/-30%" if thetap_max is None or k != "thetaP" else
                    f"+/-30% with upper clipped at {thetap_max}"}
            for j, k in enumerate(PARAMS)
        ],
        "local": {"method": "central difference", "rel_step": REL_STEP,
                  "quantity": "normalized elasticity dY/dp * p/Y"},
        "morris": {"N_MORRIS": N_MORRIS, "num_levels": NUM_LEVELS, "seed": SEED,
                   "n_solves": N_MORRIS * (len(PARAMS) + 1),
                   "nonlinear_flag": "sigma > mu_star / 2"},
        "sobol": {"N_SOBOL": N_SOBOL, "calc_second_order": CALC_SECOND_ORDER,
                  "seed": SEED, "n_solves": N_SOBOL * (len(PARAMS) + 2),
                  "skipped": bool(skip_sobol)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "SALib": _salib_version(),
        },
        "artifact_note": (
            "deltaP = 1 + deltaP1*(1-thetaP), deltaP1 = 3.5, so deltaP < 0 for "
            "thetaP > 1.2857143. In the unclipped +/-30% box thetaP reaches 1.30, "
            "APC decay reverses sign and the APC trajectory diverges (AUC ~6.4e3 "
            "at thetaP=1.30 vs 0.4734 at baseline), corrupting Morris/Sobol for "
            "the APC output. The clipped box removes this."
        ),
    }

    # ---- baseline + local ----
    print("[local] central-difference elasticity ...", flush=True)
    base, S_local = local_sensitivity()
    _w(os.path.join(outdir, "baseline_outputs.csv"),
       ["output", "baseline_auc"],
       [[n, fmt(v)] for n, v in zip(OUTPUT_NAMES, base)])
    _w(os.path.join(outdir, "local_elasticity.csv"),
       ["output"] + PARAMS,
       [[OUTPUT_NAMES[o]] + [fmt(S_local[o, j]) for j in range(len(PARAMS))]
        for o in range(len(OUTPUT_NAMES))])

    # ---- Morris ----
    X_m = morris_sample(problem, N_MORRIS, num_levels=NUM_LEVELS, seed=SEED)
    print(f"[Morris] {X_m.shape[0]} solves ...", flush=True)
    Y_m, n_fail_m = evaluate_matrix(X_m, tag="morris ")
    cfg["morris"]["n_failed_solves"] = n_fail_m

    rows, mu_star_mat = [], np.zeros((len(OUTPUT_NAMES), len(PARAMS)))
    for o, name in enumerate(OUTPUT_NAMES):
        Si = morris_analyze(problem, X_m, Y_m[:, o], num_levels=NUM_LEVELS,
                            seed=SEED, print_to_console=False)
        mu, mu_star, sigma = np.asarray(Si["mu"]), np.asarray(Si["mu_star"]), np.asarray(Si["sigma"])
        conf = np.asarray(Si["mu_star_conf"])
        mu_star_mat[o] = mu_star
        for j, pname in enumerate(PARAMS):
            rows.append([name, pname, fmt(mu[j]), fmt(mu_star[j]), fmt(sigma[j]),
                         fmt(conf[j]), str(bool(sigma[j] > mu_star[j] / 2.0))])
    _w(os.path.join(outdir, "morris_all.csv"),
       ["output", "param", "mu", "mu_star", "sigma", "mu_star_conf", "nonlinear_flag"],
       rows)

    # overall ranking: mean over outputs of the per-output max-normalized mu*
    denom = mu_star_mat.max(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    overall = (mu_star_mat / denom).mean(axis=0)
    order = np.argsort(-overall)
    _w(os.path.join(outdir, "morris_overall_ranking.csv"),
       ["param", "mean_normalized_mu_star"],
       [[PARAMS[j], fmt(overall[j])] for j in order])

    # ---- Sobol ----
    if not skip_sobol:
        X_s = sobol_sample.sample(problem, N_SOBOL,
                                  calc_second_order=CALC_SECOND_ORDER, seed=SEED)
        print(f"[Sobol] {X_s.shape[0]} solves ...", flush=True)
        Y_s, n_fail_s = evaluate_matrix(X_s, tag="sobol ")
        cfg["sobol"]["n_failed_solves"] = n_fail_s
        rows = []
        for o, name in enumerate(OUTPUT_NAMES):
            Si = sobol_analyze.analyze(problem, Y_s[:, o],
                                       calc_second_order=CALC_SECOND_ORDER,
                                       seed=SEED, print_to_console=False)
            S1, S1c = np.asarray(Si["S1"]), np.asarray(Si["S1_conf"])
            ST, STc = np.asarray(Si["ST"]), np.asarray(Si["ST_conf"])
            for j, pname in enumerate(PARAMS):
                rows.append([name, pname, fmt(S1[j]), fmt(S1c[j]),
                             fmt(ST[j]), fmt(STc[j])])
        _w(os.path.join(outdir, "sobol_all.csv"),
           ["output", "param", "S1", "S1_conf", "ST", "ST_conf"], rows)
    else:
        print("[Sobol] skipped (--skip-sobol)", flush=True)

    with open(os.path.join(outdir, "sa_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote {os.path.join(outdir, 'sa_config.json')}", flush=True)

    # ---- reproduction check against the stored figures ----
    # The reference values come from the notebook, i.e. the ORIGINAL box; a
    # mismatch in the clipped box is expected, not a failure.
    check_reproduction(outdir, skip_sobol, is_original=(thetap_max is None))


def _salib_version():
    try:
        from importlib.metadata import version
        return version("SALib")
    except Exception:
        return "unknown"


REF = {  # values read off PINN/sa_plots/{morris,sobol}_stemness.png
    "morris": {"mu_star": 1.2120, "sigma": 0.3596},
    "sobol": {"S1": 0.7496, "ST": 0.8082},
}


def _lookup(path, keycols, want, valcols):
    import csv
    with open(path) as f:
        for row in csv.DictReader(f):
            if all(row[k] == v for k, v in zip(keycols, want)):
                return {c: float(row[c]) for c in valcols}
    return None


def check_reproduction(outdir, skip_sobol, is_original=True):
    note = "" if is_original else "  [CLIPPED BOX: mismatch vs the notebook reference is EXPECTED]"
    print(f"\n--- reproduction check (Stemness / thetaP vs stored figures) ---{note}",
          flush=True)
    m = _lookup(os.path.join(outdir, "morris_all.csv"), ["output", "param"],
                ["Stemness", "thetaP"], ["mu_star", "sigma"])
    if m:
        print(f"  Morris  mu*={m['mu_star']:.4f} (ref {REF['morris']['mu_star']:.4f})  "
              f"sigma={m['sigma']:.4f} (ref {REF['morris']['sigma']:.4f})  "
              f"MATCH={abs(m['mu_star'] - REF['morris']['mu_star']) < 5e-4 and abs(m['sigma'] - REF['morris']['sigma']) < 5e-4}",
              flush=True)
    if not skip_sobol:
        s = _lookup(os.path.join(outdir, "sobol_all.csv"), ["output", "param"],
                    ["Stemness", "thetaP"], ["S1", "ST"])
        if s:
            print(f"  Sobol   S1={s['S1']:.4f} (ref {REF['sobol']['S1']:.4f})  "
                  f"ST={s['ST']:.4f} (ref {REF['sobol']['ST']:.4f})  "
                  f"MATCH={abs(s['S1'] - REF['sobol']['S1']) < 5e-4 and abs(s['ST'] - REF['sobol']['ST']) < 5e-4}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outroot", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "sa_results"))
    ap.add_argument("--skip-sobol", action="store_true",
                    help="run local + Morris only")
    ap.add_argument("--thetap-max", type=float, default=1.0,
                    help="upper clip on the thetaP box for the clipped run (default 1.0)")
    ap.add_argument("--box", choices=["original", "clipped", "both"], default="both",
                    help="which parameter box(es) to run (default both)")
    args = ap.parse_args()

    os.makedirs(args.outroot, exist_ok=True)
    t0 = time.time()
    print(f"7-ODE sensitivity analysis -> {args.outroot}", flush=True)
    print(f"  params  ({len(PARAMS)}): {PARAMS}", flush=True)
    print(f"  outputs ({len(OUTPUT_NAMES)}): {OUTPUT_NAMES}", flush=True)
    print(f"  Morris N={N_MORRIS} levels={NUM_LEVELS} seed={SEED}; "
          f"Sobol N={N_SOBOL} second_order={CALC_SECOND_ORDER} seed={SEED}", flush=True)

    if args.box in ("original", "both"):
        run_box(os.path.join(args.outroot, "original_box"), None, args.skip_sobol)
    if args.box in ("clipped", "both"):
        run_box(os.path.join(args.outroot, "clipped_thetap"), args.thetap_max,
                args.skip_sobol)

    print(f"\nALL DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
