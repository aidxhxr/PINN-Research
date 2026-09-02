"""Posterior marginals of W and thetaP across the 4 regimes (Bayesian inverse PINN).

Vector re-draw of PINN-bayesian/runs/20260713_204442_bayes/bayes_W_thetaP.png
(produced by PINN-bayesian/aggregate_marginals.py::plot_W_thetaP) with poster
typography and no suptitle. Same data: the 4 per-regime
*_posterior_samples.npz files of that run (3000 HMC draws x 36 params); the
dotted vertical line is each regime's true value (`true` array in the npz,
identical to PINN-bayesian/config.py REGIMES).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style
from style import REGIME_COLORS

RUN = ("/home/29/aidahxr/PINN-Research/PINN-bayesian/runs/"
       "20260713_204442_bayes")
REGIMES = [("Normal", "Normal"), ("Early_Adenoma", "Early Adenoma"),
           ("Advanced_Adenoma", "Advanced Adenoma"),
           ("Severe_APC_Loss", "Severe APC Loss")]
PANELS = [("W", "posterior of $W$"), ("thetaP", r"posterior of $\theta_P$")]

style.setup()
data = {}
for safe, _ in REGIMES:
    z = np.load(os.path.join(RUN, f"{safe}_posterior_samples.npz"),
                allow_pickle=True)
    data[safe] = (list(map(str, z["names"])), z["values"], z["true"])

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
for ax, (pname, title) in zip(axes, PANELS):
    for safe, label in REGIMES:
        names, vals, true = data[safe]
        i = names.index(pname)
        c = REGIME_COLORS[label]
        ax.hist(vals[:, i], bins=50, density=True, histtype="stepfilled",
                alpha=0.35, color=c, lw=0)
        ax.hist(vals[:, i], bins=50, density=True, histtype="step",
                lw=2.0, color=c)
        ax.axvline(float(true[i]), color=c, ls=":", lw=2.2)
    ax.set_title(title, fontsize=24, pad=8)
    ax.set_yticks([])
    ax.set_ylabel("density", fontsize=21)
    ax.tick_params(axis="x", labelsize=20)
    style.clean(ax, x=False, y=True)
    ax.grid(False)
axes[0].set_xticks([0.5, 1.0, 1.5, 2.0])
axes[1].set_xticks([0.25, 0.5, 0.75, 1.0])

handles = [Line2D([], [], color=REGIME_COLORS[lab], lw=4, label=lab)
           for _, lab in REGIMES]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=19,
           handlelength=1.2, borderaxespad=0.0, columnspacing=1.0,
           handletextpad=0.5)
fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
style.save(fig, "fig_bayes_marginals")
pdf = os.path.join(style.ASSETS, "fig_bayes_marginals.pdf")
if not os.path.exists(pdf):
    fig.savefig(pdf)
