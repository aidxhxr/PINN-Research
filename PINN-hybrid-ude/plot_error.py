"""Standalone: parameter-recovery error vs iteration, built from the
*_history.json files a run writes continuously. Works on an in-progress run
(call it any time to see how far the error has fallen) or a finished one.

Usage:
    python3 plot_error.py                 # newest runs/<ts>/ dir
    python3 plot_error.py runs/2026..../  # a specific run dir
"""
import os
import sys
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import BASELINE, REGIMES, UNKNOWN

COLORS = dict(zip(REGIMES, ["tab:blue", "tab:green", "tab:purple", "tab:pink"]))


def _safe(name):
    return name.replace(" ", "_").replace("/", "_")


def main(run_dir):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    found = 0
    for name in REGIMES:
        path = os.path.join(run_dir, f"{_safe(name)}_history.json")
        if not os.path.exists(path):
            print(f"  (skip {name}: no history yet)")
            continue
        with open(path) as fh:
            h = json.load(fh)
        if not h.get("epoch"):
            continue
        found += 1
        c = COLORS[name]
        eps = np.asarray(h["epoch"], dtype=float)
        tv = {pk: ({**BASELINE, **REGIMES[name]})[pk] for pk in UNKNOWN}
        E = np.array([
            [abs(h[pk][i] - tv[pk]) / abs(tv[pk]) * 100.0
             for pk in UNKNOWN if tv[pk] != 0]
            for i in range(len(eps))
        ])
        axL.semilogy(eps, E.mean(axis=1), "-", lw=1.8, color=c,
                     label=f"{name} (mean)")
        axL.semilogy(eps, np.median(E, axis=1), "--", lw=1.0, color=c,
                     alpha=0.7, label=f"{name} (median)")
        axR.plot(eps, (E <= 5.0).sum(axis=1), "-", lw=1.8, color=c, label=name)
        print(f"  {name:<20s} iters {int(eps[0])}..{int(eps[-1])}  "
              f"mean.err {E.mean(axis=1)[-1]:.1f}%  "
              f"within5% {(E[-1] <= 5).sum()}/{len(UNKNOWN)}")

    if not found:
        print("No *_history.json with data found in", run_dir)
        return

    for ax in (axL, axR):
        ax.set_xlabel("Iteration (Adam + L-BFGS)")
        ax.grid(True, alpha=0.3)
    axL.axhline(5.0, ls=":", color="k", alpha=0.6, label="5% target")
    axL.set_ylabel("Relative error  (%)")
    axL.set_title("Parameter recovery error vs iteration")
    axL.legend(fontsize=7, ncol=2)
    axR.set_ylabel("# params within 5%")
    axR.set_ylim(0, len(UNKNOWN))
    axR.set_title(f"Params recovered <5%  (of {len(UNKNOWN)})")
    axR.legend(fontsize=8)
    out = os.path.join(run_dir, "inv_param_error.png")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rd = sys.argv[1]
    else:
        dirs = sorted(glob.glob("runs/*/"))
        rd = dirs[-1] if dirs else "."
    print("run dir:", rd)
    main(rd)
