"""The headline figure: learned f_NN(x) vs the true closed form it replaced.

This is the FUNCTIONAL identifiability panel (arXiv:2510.14140) -- separate from
whether the remaining mechanistic parameters were recovered. Two honesty rules
are baked in:

  * the curve is only scored/solid over the range the regulator ACTUALLY visits
    in the reference trajectories; outside it the data never constrained the
    net, so it is drawn dashed and excluded from the RMSE.
  * the true curve is per-regime (regimes differ in W/thetaP, which moves where
    the trajectories sit even though the term's own parameters are shared).

usage: python3 plot_learned_term.py <run_dir>
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss"]
# SA-big palette (standing preference across this repo's figures)
C_LEARNED = "#1f77b4"
C_TRUE = "#cc3333"
C_GREY = "#7f7f7f"

LABEL = {
    "ra_h5":  (r"$r$  (RA)", r"$\eta_R\,r/(\kappa_R+r)$", r"RA $\to$ HOXA5"),
    "bm_myc": (r"$b$  ($\beta$-catenin)", r"$\eta_{BM}\,h(b;\kappa_{BM})$",
               r"WNT $\to$ MYC"),
    "b_h13":  (r"$b$  ($\beta$-catenin)", r"$\eta_{B13}\,h(b;\kappa_{B13})$",
               r"WNT $\to$ HOXA13"),
    "bc_cyp": (r"$b$  ($\beta$-catenin)", r"$\eta_{BC}\,h(b;\kappa_{BC})$",
               r"WNT $\to$ CYP26A1"),
}


def main(run_dir):
    files = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "*_term.json"))):
        safe = os.path.basename(f).replace("_term.json", "")
        name = safe.replace("_", " ")
        files[name] = json.load(open(f))
    if not files:
        print("[plot_learned_term] no *_term.json -- mechanistic control run?")
        return

    term = next(iter(files.values()))["term"]
    xlab, ylab, title = LABEL.get(term, ("x", "f(x)", term))

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    print(f"\nlearned term '{term}' -- functional identifiability")
    print(f"{'regime':<20s} {'range':>18s} {'RMSE':>9s} {'NRMSE':>8s}")

    for ax, name in zip(axes, ORDER):
        key = next((k for k in files if k.replace(" ", "") == name.replace(" ", "")),
                   None)
        if key is None:
            ax.set_visible(False)
            continue
        d = files[key]
        g = np.array(d["grid"]); learned = np.array(d["learned"])
        truth = np.array(d["truth"])

        ax.plot(g, truth, color=C_TRUE, lw=2.4, label="true (Hill)")
        ax.plot(g, learned, color=C_LEARNED, lw=2.0, ls="--",
                label=r"learned $f_{NN}$")
        ax.axvspan(d["r_lo"], d["r_hi"], color=C_GREY, alpha=0.13, lw=0,
                   label="observed range")
        ax.set_title(f"{name}\nNRMSE {d['nrmse']:.1%}", fontsize=10)
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.25, lw=0.6)
        print(f"{name:<20s} [{d['r_lo']:6.3f},{d['r_hi']:6.3f}] "
              f"{d['rmse']:>9.4f} {d['nrmse']:>7.1%}")

    axes[0].set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Neural-mechanistic hybrid — learned term vs. truth  ({title})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(run_dir, f"learned_term_{term}.png")
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
