"""Top-8 posterior marginals for the Bayesian inverse PINN.

Slices the 8 MOST-identifiable parameters (from the full-36 FIM screen, sister
folder PINN-fisher-matrix-top8) out of the SAME full-36 HMC posterior and plots
them as a 4-wide x 2-tall grid per regime. These are honest marginals of the
36-param posterior, restricted to the 8 columns — nothing is re-run.

Default regimes: Normal and Severe APC Loss (the low- and high-WNT extremes).
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the 8 most-identifiable params, worst-case across regimes (fisher-top8 FIM)
TOP8 = ["lambdaC", "etaBM", "kappaBM", "kappaBC",
        "etaBC", "etaRC", "kappaB13", "kappaR"]

REGIME_LABEL = {"Normal": "Normal", "Early_Adenoma": "Early Adenoma",
                "Advanced_Adenoma": "Advanced Adenoma",
                "Severe_APC_Loss": "Severe APC Loss"}

# SA-big palette: IDENT=blue, WEAK=orange, NON-IDENT=red
VERDICT_COL = {"IDENT": "#1f77b4", "WEAK": "#ff7f0e", "NON-IDENT": "#cc3333"}


def plot_regime(run_dir, safe, mode="top"):
    npz = os.path.join(run_dir, f"{safe}_posterior_samples.npz")
    js = os.path.join(run_dir, f"{safe}_posterior_summary.json")
    if not (os.path.exists(npz) and os.path.exists(js)):
        print(f"  [skip] {safe}: outputs missing in {run_dir}")
        return
    z = np.load(npz, allow_pickle=True)
    names = list(z["names"])
    values = z["values"]
    true = z["true"]
    with open(js) as fh:
        summary = json.load(fh)["params"]

    def covered(k):
        lo, hi = summary[k]["ci95"]
        return lo <= summary[k]["true"] <= hi

    if mode == "least":
        # 8 WORST-recovered = truth furthest outside the 95% CI (largest |z|).
        # This is the honest failure signal: shrink calls everything IDENT
        # because high-WNT posteriors are tight-but-BIASED, not wide.
        params = sorted(names, key=lambda k: -abs(summary[k]["z_truth"]))[:8]
        which = "8 worst-recovered"
    else:
        params = TOP8
        which = "top-8 most identifiable"

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, k in zip(axes.flat, params):
        i = names.index(k)
        vals = values[:, i]
        ax.hist(vals, bins=40, density=True, color="#1f77b4", alpha=0.75)
        ax.axvline(true[i], color="#444444", lw=2, label="true")
        ax.axvline(summary[k]["post_mean"], color="k", ls="--", lw=1.2,
                   label="post mean")
        ok = covered(k)
        col = "#1f77b4" if ok else "#cc3333"
        tag = "truth IN CI" if ok else "truth OUT of CI"
        ax.set_title(f"{k}  [{tag}]  z={summary[k]['z_truth']:.1f}",
                     fontsize=11, color=col)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8)
    ftag = "worst8" if mode == "least" else "top8"
    ncov = sum(covered(k) for k in names)
    fig.suptitle(f"{REGIME_LABEL.get(safe, safe)} — posterior marginals "
                 f"({which})   "
                 f"[blue = truth inside 95% CI, red = confidently wrong]   "
                 f"coverage {ncov}/36", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(run_dir, f"{safe}_{ftag}_marginals.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--regimes", nargs="+",
                    default=["Normal", "Severe_APC_Loss"])
    ap.add_argument("--mode", choices=["top", "least"], default="top")
    args = ap.parse_args()
    for safe in args.regimes:
        plot_regime(args.run, safe, mode=args.mode)


if __name__ == "__main__":
    main()
