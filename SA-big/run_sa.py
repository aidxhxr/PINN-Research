"""
SA-big — global + local sensitivity analysis of the 27-state Wnt/Retinoid/HOX
model.  Large-system counterpart of `PINN/sensitivity_analysis.ipynb`.

Produces, for every scored output:
  * Local sensitivity (normalized elasticity): per-output signed bar charts +
    an all-outputs heatmap.
  * Morris elementary-effects screening: mu* vs sigma scatter + influence
    ranking bar; plus an all-outputs mu* heatmap and an aggregate ranking.
  * (optional) Sobol variance decomposition: first-order S1 + total-order ST
    bars; plus an all-outputs ST heatmap.

All figures land in ./sa_plots, all numbers (CSV + ranking tables) in ./results.

Usage:
    python3 run_sa.py                       # Morris + local (+ Sobol if not skipped)
    python3 run_sa.py --morris-N 24 --skip-sobol
    python3 run_sa.py --sobol-N 32 --workers 60

Runtime note: each model solve is ~35 s.  Total solves =
    local: 2*P+1            (~61)
    Morris: N*(P+1)         (e.g. 24*31 = 744)
    Sobol:  M*(P+2)         (e.g. 32*32 = 1024, second-order off)
At ~60 parallel workers a solve-batch is ~40-60 s.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

import sa_core as sc

BLUE, ORANGE = "#1f77b4", "#ff7f0e"
PLOT_DIR = os.path.join(os.path.dirname(__file__), "sa_plots")
RES_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

PARAMS = sc.PARAMS
OUTPUT_NAMES = sc.OUTPUT_NAMES


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#444444",
        "axes.grid": False, "font.size": 11, "axes.titlesize": 12,
        "legend.frameon": False,
    })


def save_csv(path, header, rows, fmt="%.6g"):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join((c if isinstance(c, str) else (fmt % c)) for c in r) + "\n")
    print(f"  wrote {path}")


# ============================================================================
# Baseline
# ============================================================================
def do_baseline():
    print("\n[baseline] solving nominal model ...", flush=True)
    base = sc.baseline_outputs()
    rows = [[n, v] for n, v in zip(OUTPUT_NAMES, base)]
    print("  baseline settled outputs (time-avg over [%.0f, %.0f]):"
          % (sc.AVG_LO, sc.AVG_HI))
    for n, v in zip(OUTPUT_NAMES, base):
        print(f"    {n:<10} {v: .5f}")
    save_csv(os.path.join(RES_DIR, "baseline_outputs.csv"),
             ["output", "baseline_value"], rows)
    return base


# ============================================================================
# Local sensitivity (normalized elasticity via central differences)
# ============================================================================
def do_local(base, rel_step=0.05):
    print("\n[local] central-difference elasticities ...", flush=True)
    # build the +/- perturbation design matrix, evaluate in parallel
    X, meta = [], []
    for j, key in enumerate(PARAMS):
        for sign in (+1, -1):
            ov = dict(zip(PARAMS, [sc.BASELINE[k] for k in PARAMS]))
            ov[key] = sc.BASELINE[key] * (1.0 + sign * rel_step)
            X.append([ov[k] for k in PARAMS])
            meta.append((j, sign))
    Y = sc.evaluate_matrix(np.array(X))
    S = np.zeros((len(OUTPUT_NAMES), len(PARAMS)))   # elasticity[output, param]
    yj = {}
    for row, (j, sign) in zip(Y, meta):
        yj[(j, sign)] = row
    for j, key in enumerate(PARAMS):
        d = sc.BASELINE[key] * rel_step
        dydp = (yj[(j, +1)] - yj[(j, -1)]) / (2.0 * d)
        with np.errstate(divide="ignore", invalid="ignore"):
            S[:, j] = dydp * sc.BASELINE[key] / np.where(base == 0, np.nan, base)
    S = np.nan_to_num(S)

    # per-output signed bar charts
    for o, name in enumerate(OUTPUT_NAMES):
        s = S[o]
        order = np.argsort(np.abs(s))
        colors = [BLUE if s[i] >= 0 else ORANGE for i in order]
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.barh([PARAMS[i] for i in order], s[order], color=colors)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_xlabel("normalized local sensitivity (elasticity)")
        ax.set_title(f"{name} — local sensitivity")
        ax.legend(handles=[Patch(color=BLUE, label="positive"),
                           Patch(color=ORANGE, label="negative")],
                  loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"local_{sc.slug(name)}.png"), dpi=150)
        plt.close(fig)

    # combined heatmap (outputs x params)
    fig, ax = plt.subplots(figsize=(13, 5))
    vmax = np.abs(S).max() or 1.0
    im = ax.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(PARAMS))); ax.set_xticklabels(PARAMS, rotation=90, fontsize=8)
    ax.set_yticks(range(len(OUTPUT_NAMES))); ax.set_yticklabels(OUTPUT_NAMES)
    ax.set_title("Local sensitivity (elasticity) — all outputs")
    fig.colorbar(im, ax=ax, shrink=0.8, label="normalized sensitivity")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "local_heatmap_all.png"), dpi=150)
    plt.close(fig)

    # CSV: rows = params, cols = outputs
    rows = [[PARAMS[j]] + [S[o, j] for o in range(len(OUTPUT_NAMES))]
            for j in range(len(PARAMS))]
    save_csv(os.path.join(RES_DIR, "local_elasticity.csv"),
             ["param"] + OUTPUT_NAMES, rows)
    print("  saved local-sensitivity plots + CSV")
    return S


# ============================================================================
# Morris elementary-effects screening
# ============================================================================
def do_morris(N, num_levels=4):
    print(f"\n[morris] N={N} trajectories ...", flush=True)
    X = morris_sample(sc.problem, N, num_levels=num_levels, seed=42)
    Y = sc.evaluate_matrix(X)

    mu_star_mat = np.zeros((len(PARAMS), len(OUTPUT_NAMES)))
    all_rows = []
    for o, name in enumerate(OUTPUT_NAMES):
        Si = morris_analyze(sc.problem, X, Y[:, o], num_levels=num_levels,
                            seed=42, print_to_console=False)
        mu_star, mu, sigma = Si["mu_star"], Si["mu"], Si["sigma"]
        mu_star_mat[:, o] = mu_star
        order = np.argsort(mu_star)
        nonlinear = sigma > (mu_star / 2.0)

        # ---- ranking table to console + CSV ----
        print(f"\n  ── {name} ──")
        print(f"  {'Rank':<5}{'Parameter':<12}{'mu_star':>11}{'sigma':>11}{'mu':>11}")
        desc = np.argsort(-mu_star)
        for rank, idx in enumerate(desc, 1):
            flag = " *" if nonlinear[idx] else ""
            print(f"  {rank:<5}{PARAMS[idx]:<12}{mu_star[idx]:>11.4g}"
                  f"{sigma[idx]:>11.4g}{mu[idx]:>11.4g}{flag}")
            all_rows.append([name, PARAMS[idx], mu_star[idx], sigma[idx],
                             mu[idx], "yes" if nonlinear[idx] else "no"])

        # ---- per-output figure: mu*-sigma scatter + ranking bar ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax1.scatter(mu_star, sigma, s=55,
                    color=[ORANGE if n else BLUE for n in nonlinear], zorder=3)
        for i in range(len(PARAMS)):
            ax1.annotate(PARAMS[i], (mu_star[i], sigma[i]),
                         textcoords="offset points", xytext=(4, 3), fontsize=7.5)
        lim = max(mu_star.max(), sigma.max()) * 1.15 + 1e-12
        ax1.plot([0, lim], [0, lim / 2], "--", color="#888888", lw=0.9,
                 label=r"$\sigma=\mu^*/2$")
        ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)
        ax1.set_xlabel(r"$\mu^*$ (mean abs. elementary effect)")
        ax1.set_ylabel(r"$\sigma$ (std of elementary effect)")
        ax1.set_title(f"{name} — Morris screening")
        ax1.legend()

        ax2.barh([PARAMS[i] for i in order], mu_star[order],
                 color=[ORANGE if nonlinear[i] else BLUE for i in order])
        ax2.set_xlabel(r"$\mu^*$")
        ax2.set_title(f"{name} — influence ranking")
        ax2.legend(handles=[Patch(color=BLUE, label="near-linear"),
                            Patch(color=ORANGE, label=r"nonlinear/interacting ($\sigma>\mu^*/2$)")],
                   loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"morris_{sc.slug(name)}.png"), dpi=150)
        plt.close(fig)

    save_csv(os.path.join(RES_DIR, "morris_all.csv"),
             ["output", "param", "mu_star", "sigma", "mu", "nonlinear"], all_rows)

    # ---- all-outputs mu* heatmap (column-normalized for comparability) ----
    norm = mu_star_mat / (mu_star_mat.max(axis=0, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 11))
    im = ax.imshow(norm, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(OUTPUT_NAMES))); ax.set_xticklabels(OUTPUT_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(PARAMS))); ax.set_yticklabels(PARAMS, fontsize=8)
    ax.set_title("Morris $\\mu^*$ (column-normalized) — params × outputs")
    fig.colorbar(im, ax=ax, shrink=0.7, label=r"$\mu^*$ / max per output")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "morris_mustar_heatmap.png"), dpi=150)
    plt.close(fig)

    # ---- aggregate ranking across outputs (mean of column-normalized mu*) ----
    agg = norm.mean(axis=1)
    order = np.argsort(agg)
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.barh([PARAMS[i] for i in order], agg[order], color=BLUE)
    ax.set_xlabel("mean normalized $\\mu^*$ across outputs")
    ax.set_title("Overall parameter influence (Morris, aggregated)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "morris_overall_ranking.png"), dpi=150)
    plt.close(fig)
    save_csv(os.path.join(RES_DIR, "morris_overall_ranking.csv"),
             ["param", "mean_norm_mu_star"],
             [[PARAMS[i], agg[i]] for i in np.argsort(-agg)])
    print("\n  saved Morris plots + CSVs")
    return mu_star_mat


# ============================================================================
# Sobol variance decomposition
# ============================================================================
def do_sobol(M):
    print(f"\n[sobol] base sample M={M} (calc_second_order=False) ...", flush=True)
    X = sobol_sample.sample(sc.problem, M, calc_second_order=False, seed=42)
    Y = sc.evaluate_matrix(X)

    ST_mat = np.zeros((len(PARAMS), len(OUTPUT_NAMES)))
    all_rows = []
    x = np.arange(len(PARAMS))
    for o, name in enumerate(OUTPUT_NAMES):
        Si = sobol_analyze.analyze(sc.problem, Y[:, o], calc_second_order=False,
                                   seed=42, print_to_console=False)
        S1, ST = Si["S1"], Si["ST"]
        ST_mat[:, o] = ST
        for j in range(len(PARAMS)):
            all_rows.append([name, PARAMS[j], S1[j], ST[j],
                             Si["S1_conf"][j], Si["ST_conf"][j]])
        fig, ax = plt.subplots(figsize=(13, 5))
        w = 0.4
        ax.bar(x - w / 2, S1, w, color=BLUE, label="first-order $S_1$")
        ax.bar(x + w / 2, ST, w, color=ORANGE, label="total-order $S_T$")
        ax.axhline(0, color="#444444", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(PARAMS, rotation=90, fontsize=8)
        ax.set_ylabel("Sobol index")
        ax.set_title(f"Sobol sensitivity — {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"sobol_{sc.slug(name)}.png"), dpi=150)
        plt.close(fig)

    save_csv(os.path.join(RES_DIR, "sobol_all.csv"),
             ["output", "param", "S1", "ST", "S1_conf", "ST_conf"], all_rows)

    norm = ST_mat / (ST_mat.max(axis=0, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 11))
    im = ax.imshow(norm, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(OUTPUT_NAMES))); ax.set_xticklabels(OUTPUT_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(PARAMS))); ax.set_yticklabels(PARAMS, fontsize=8)
    ax.set_title("Sobol $S_T$ (column-normalized) — params × outputs")
    fig.colorbar(im, ax=ax, shrink=0.7, label=r"$S_T$ / max per output")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "sobol_ST_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  saved Sobol plots + CSV")
    return ST_mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morris-N", type=int, default=24)
    ap.add_argument("--sobol-N", type=int, default=32)
    ap.add_argument("--skip-local", action="store_true")
    ap.add_argument("--skip-morris", action="store_true")
    ap.add_argument("--skip-sobol", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.workers:
        sc.DEFAULT_WORKERS = args.workers

    set_style()
    print("=" * 70)
    print("SA-big: sensitivity analysis of the 27-state Wnt/Retinoid/HOX model")
    print(f"  parameters varied : {len(PARAMS)} (+/-{int(sc.VARY_FRAC*100)}% box)")
    print(f"  outputs scored    : {len(OUTPUT_NAMES)}")
    print(f"  solver            : {sc.SOLVER}")
    print("=" * 70)

    base = do_baseline()
    if not args.skip_local:
        do_local(base)
    if not args.skip_morris:
        do_morris(args.morris_N)
    if not args.skip_sobol:
        do_sobol(args.sobol_N)
    print("\nDONE. Figures -> sa_plots/ , numbers -> results/")


if __name__ == "__main__":
    main()
