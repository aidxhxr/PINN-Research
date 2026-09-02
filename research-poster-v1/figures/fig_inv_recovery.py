"""Inverse PINN: true vs recovered, the 3 best-recovered parameters x 4 regimes.

Vector re-draw of the TOP ROW of research-paper/inv_recovery_bars_best8.png
(md5-identical to PINN-inverse-pinn-boost/runs/20260711_170848_replot_legendfix/
inv_recovery_bars_best8.png, produced by PINN-inverse-pinn-boost/regen_plots.py
-> plotting.plot_recovery_bars with params = plotting.best_params(k=8)).
Data: that run dir's <regime>_recovered.json ("true" / "recovered" dicts).
best_params ranks by mean relative error over regimes; the first three are
lambdaC, etaBC, kappaBC (recomputed here from the same json, and asserted).
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import style
from style import GREY, BLUE

RUN = ("/home/29/aidahxr/PINN-Research/PINN-inverse-pinn-boost/runs/"
       "20260711_170848_replot_legendfix")
REGIMES = [("Normal", "Normal"), ("Early_Adenoma", "Early"),
           ("Advanced_Adenoma", "Advanced"), ("Severe_APC_Loss", "Severe")]
TEX = {"lambdaC": r"$\lambda_C$", "etaBC": r"$\eta_{BC}$",
       "kappaBC": r"$\kappa_{BC}$"}

style.setup()
true, rec = {}, {}
for safe, _ in REGIMES:
    with open(os.path.join(RUN, f"{safe}_recovered.json")) as fh:
        j = json.load(fh)
    true[safe], rec[safe] = j["true"], j["recovered"]

# same ranking as plotting.best_params: mean relative error over regimes
scored = sorted(
    (np.mean([abs(rec[s][pk] - true[s][pk]) / abs(true[s][pk])
              for s, _ in REGIMES if true[s][pk] != 0]), pk)
    for pk in true["Normal"])
params = [pk for _, pk in scored[:3]]
assert params == ["lambdaC", "etaBC", "kappaBC"], params

fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
x = np.arange(len(REGIMES)); w = 0.38
for ax, pk in zip(axes, params):
    tv = [true[s][pk] for s, _ in REGIMES]
    rv = [rec[s][pk] for s, _ in REGIMES]
    ax.bar(x - w / 2, tv, w, color=GREY, label="true")
    ax.bar(x + w / 2, rv, w, color=BLUE, label="recovered")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in REGIMES], fontsize=20,
                       rotation=35, ha="right", rotation_mode="anchor")
    ax.set_title(TEX[pk], fontsize=26, pad=6)
    ax.tick_params(axis="y", labelsize=20)
    top = max(max(tv), max(rv))
    ax.set_ylim(0, top * 1.12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    style.clean(ax, x=False, y=True)

fig.legend(handles=[Patch(color=GREY, label="true"),
                    Patch(color=BLUE, label="recovered")],
           loc="lower center", ncol=2, fontsize=20, borderaxespad=0.0,
           columnspacing=1.5, handlelength=1.4)
fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
style.save(fig, "fig_inv_recovery")
pdf = os.path.join(style.ASSETS, "fig_inv_recovery.pdf")
if not os.path.exists(pdf):
    fig.savefig(pdf)
