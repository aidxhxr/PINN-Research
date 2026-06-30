"""
SA-big — cross-method synthesis / model-reduction recommendation.

`run_sa.py` produces the three sensitivity methods *separately* (local
elasticity, Morris mu*, Sobol S_T).  This script consolidates them into a
single per-parameter verdict: a consensus influence score and a FIX / KEEP
recommendation for reducing the model / the inverse-problem unknown set.

A parameter is a **FIX candidate** only when it is negligible in *all three*
methods (each method's column-normalized, output-aggregated score is below
`FIX_THRESH` of that method's maximum) — i.e. no method finds it influential.
This is the deliverable the README points at: "Parameters that sit near zero
across all three methods are candidates to fix."

Inputs  (written by run_sa.py):
    results/local_elasticity.csv
    results/morris_all.csv
    results/sobol_all.csv
Outputs:
    results/combined_ranking.csv
    sa_plots/combined_ranking.png
    + a verdict table to the console.

Usage:
    python3 synthesize.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sa_core as sc

HERE = os.path.dirname(__file__)
RES_DIR = os.path.join(HERE, "results")
PLOT_DIR = os.path.join(HERE, "sa_plots")
BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#bbbbbb"

# A param is a FIX candidate if every method's aggregate score is below this
# fraction of that method's maximum (i.e. <5% of the most influential param).
FIX_THRESH = 0.05

PARAMS = sc.PARAMS
OUTPUT_NAMES = sc.OUTPUT_NAMES


def _read_matrix(path, value_col):
    """Read a long-format CSV (output,param,...) into a (param x output) matrix
    of abs(value_col), aggregated by column-normalize then mean across outputs."""
    mat = {p: {} for p in PARAMS}
    with open(path) as f:
        for row in csv.DictReader(f):
            mat[row["param"]][row["output"]] = abs(float(row[value_col]))
    M = np.array([[mat[p].get(o, 0.0) for o in OUTPUT_NAMES] for p in PARAMS])
    return M  # (n_params, n_outputs)


def _read_local(path):
    """local_elasticity.csv is wide: param, then one column per output."""
    M = np.zeros((len(PARAMS), len(OUTPUT_NAMES)))
    idx = {p: i for i, p in enumerate(PARAMS)}
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c != "param"]
        for row in reader:
            i = idx[row["param"]]
            M[i] = [abs(float(row[c])) for c in cols]
    return M


def _aggregate(M):
    """Column-normalize (per output) then average across outputs -> one score
    per parameter in [0,1]. Mirrors run_sa.py's Morris/Sobol heatmap aggregation."""
    norm = M / (M.max(axis=0, keepdims=True) + 1e-12)
    return norm.mean(axis=1)


def main():
    local = _aggregate(_read_local(os.path.join(RES_DIR, "local_elasticity.csv")))
    morris = _aggregate(_read_matrix(os.path.join(RES_DIR, "morris_all.csv"), "mu_star"))
    sobol = _aggregate(_read_matrix(os.path.join(RES_DIR, "sobol_all.csv"), "ST"))

    # rescale each method to its own max so "% of most influential param" is
    # comparable across methods, then take the consensus (mean of the three).
    def rescale(v):
        return v / (v.max() + 1e-12)
    L, Mo, So = rescale(local), rescale(morris), rescale(sobol)
    consensus = (L + Mo + So) / 3.0

    fix = (L < FIX_THRESH) & (Mo < FIX_THRESH) & (So < FIX_THRESH)

    order = np.argsort(-consensus)

    # ---- console verdict table ----
    print("=" * 78)
    print("SA-big cross-method synthesis — model-reduction recommendation")
    print(f"  consensus = mean of per-method (max-normalized) aggregate scores")
    print(f"  FIX candidate = all three methods < {FIX_THRESH:.0%} of their max")
    print("=" * 78)
    print(f"  {'Rank':<5}{'Param':<10}{'Local':>9}{'Morris':>9}{'Sobol':>9}"
          f"{'Consensus':>11}  Verdict")
    for rank, i in enumerate(order, 1):
        verdict = "FIX" if fix[i] else "keep"
        print(f"  {rank:<5}{PARAMS[i]:<10}{L[i]:>9.3f}{Mo[i]:>9.3f}{So[i]:>9.3f}"
              f"{consensus[i]:>11.3f}  {verdict}")
    fix_list = [PARAMS[i] for i in order if fix[i]]
    keep_list = [PARAMS[i] for i in order if not fix[i]]
    print("-" * 78)
    print(f"  KEEP ({len(keep_list)}): {', '.join(keep_list)}")
    print(f"  FIX  ({len(fix_list)}): {', '.join(fix_list)}")
    print("=" * 78)

    # ---- CSV ----
    path = os.path.join(RES_DIR, "combined_ranking.csv")
    with open(path, "w") as f:
        f.write("param,local_norm,morris_norm,sobol_norm,consensus,verdict\n")
        for i in order:
            f.write(f"{PARAMS[i]},{L[i]:.6g},{Mo[i]:.6g},{So[i]:.6g},"
                    f"{consensus[i]:.6g},{'fix' if fix[i] else 'keep'}\n")
    print(f"  wrote {path}")

    # ---- plot: grouped bars (consensus order) + FIX shading ----
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    names = [PARAMS[i] for i in order][::-1]   # least influential at bottom
    Lp, Mp, Sp = L[order][::-1], Mo[order][::-1], So[order][::-1]
    fixp = fix[order][::-1]
    y = np.arange(len(names))
    h = 0.26
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(y + h, Lp, h, color=GREY, label="Local |elasticity|")
    ax.barh(y, Mp, h, color=BLUE, label=r"Morris $\mu^*$")
    ax.barh(y - h, Sp, h, color=ORANGE, label=r"Sobol $S_T$")
    for yi, isfix in zip(y, fixp):
        if isfix:
            ax.axhspan(yi - 0.45, yi + 0.45, color="#ffe9e0", zorder=0)
    ax.axvline(FIX_THRESH, color="#cc3333", ls="--", lw=0.9,
               label=f"FIX threshold ({FIX_THRESH:.0%})")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("aggregate influence (max-normalized per method)")
    ax.set_title("SA-big — cross-method parameter influence\n"
                 "(shaded = FIX candidate: negligible in all three methods)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "combined_ranking.png"), dpi=150)
    plt.close(fig)
    print(f"  wrote {os.path.join(PLOT_DIR, 'combined_ranking.png')}")


if __name__ == "__main__":
    main()
