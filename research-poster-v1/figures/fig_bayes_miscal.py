"""Bayesian inverse PINN, Severe APC Loss: the 4 worst-recovered posteriors.

Vector re-draw of the TOP ROW of research-paper/bayes_worst8_severe.png
(md5-identical to PINN-bayesian/runs/20260712_204532_bayes/
Severe_APC_Loss_worst8_marginals.png, produced by
PINN-bayesian/plot_top8_marginals.py --mode least). Data: that run's
Severe_APC_Loss_posterior_samples.npz (+ _posterior_summary.json for ci95 and
z_truth). Worst = truth furthest outside the 95% interval (largest |z_truth|);
the first four are epsR, W, eta13, etaBM (recomputed and asserted).
Miscalibration figure: tight posterior, truth far outside the 95% interval.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import style
from style import BLUE, RED, INK

RUN = ("/home/29/aidahxr/PINN-Research/PINN-bayesian/runs/"
       "20260712_204532_bayes")
SAFE = "Severe_APC_Loss"
TEX = {"epsR": r"$\epsilon_R$", "W": r"$W$", "eta13": r"$\eta_{13}$",
       "etaBM": r"$\eta_{BM}$"}

style.setup()
z = np.load(os.path.join(RUN, f"{SAFE}_posterior_samples.npz"),
            allow_pickle=True)
names = list(map(str, z["names"])); values = z["values"]; true = z["true"]
with open(os.path.join(RUN, f"{SAFE}_posterior_summary.json")) as fh:
    summ = json.load(fh)["params"]
params = sorted(names, key=lambda k: -abs(summ[k]["z_truth"]))[:4]
assert params == ["epsR", "W", "eta13", "etaBM"], params

fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.4))
for ax, k in zip(axes, params):
    i = names.index(k)
    v = values[:, i]; tv = float(true[i]); lo, hi = summ[k]["ci95"]
    n, _, _ = ax.hist(v, bins=40, density=True, color=BLUE, alpha=0.85, lw=0)
    ymax = n.max()
    ax.axvline(tv, color=RED, ls=":", lw=2.6)
    ax.plot([lo, hi], [ymax * 0.05] * 2, color=INK, lw=3.0,
            solid_capstyle="butt", zorder=5)
    ax.set_title(TEX[k], fontsize=26, pad=6)
    ax.set_yticks([])
    ax.set_ylim(0, ymax * 1.12)
    lo_x, hi_x = min(v.min(), tv), max(v.max(), tv)
    pad = 0.12 * (hi_x - lo_x)
    ax.set_xlim(lo_x - pad, hi_x + pad)
    ax.set_xticks([round(tv, 2), round(float(np.median(v)), 2)])
    ax.tick_params(axis="x", labelsize=20)
    style.clean(ax, x=False, y=True)
    ax.grid(False)

fig.legend(handles=[Patch(color=BLUE, label="posterior"),
                    Line2D([], [], color=INK, lw=3.0, label="95% interval"),
                    Line2D([], [], color=RED, ls=":", lw=2.6, label="truth")],
           loc="lower center", ncol=3, fontsize=20, borderaxespad=0.0,
           columnspacing=1.4, handlelength=1.4)
fig.get_layout_engine().set(rect=(0, 0.15, 1, 0.85), wspace=0.09)
style.save(fig, "fig_bayes_miscal")
pdf = os.path.join(style.ASSETS, "fig_bayes_miscal.pdf")
if not os.path.exists(pdf):
    fig.savefig(pdf)
