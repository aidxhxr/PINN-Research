"""Compact dose-response for the poster's fourth column.

Anchor-ratio panel only (the panel that carries the claim), both regimes side
by side. The vs-dose comparison is stated in the caption instead of plotted.
Data: runs/20260802_dose_response/dose_{ra,wnt,bcat}_*.json
"""
import json, glob
import style
from style import BLUE, ORANGE, RED, INK, INK2, SURFACE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

style.setup()
RUN = "/home/29/aidahxr/PINN-Research/PINN-hybrid-ude/runs/20260802_dose_response/"
recs = []
for f in glob.glob(RUN + "dose_*_*.json"):
    recs += json.load(open(f))

ARMS = {
    "ra":   ("retinoid restriction", BLUE, "o"),
    "wnt":  ("WNT knockdown only", ORANGE, "s"),
    "bcat": ("WNT + 1 more siRNA", RED, "D"),
}
FLOOR = 5e-4

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True)
for ax, reg, short in zip(axes, ("Normal", "Severe APC Loss"),
                          ("Normal", "Severe APC Loss")):
    for arm, (lab, c, mk) in ARMS.items():
        pts = sorted([r for r in recs if r["arm"] == arm and r["regime"] == reg],
                     key=lambda r: r["anchor_ratio"])
        if not pts:
            continue
        x = [max(r["anchor_ratio"], FLOOR) for r in pts]
        y = [100 * r["hybrid_basal_err"] for r in pts]
        ax.plot(x, y, "-", color=c, lw=2.6, zorder=3, alpha=0.92,
                solid_capstyle="round")
        ax.scatter(x, y, s=110, color=c, marker=mk, zorder=4,
                   edgecolors=SURFACE, linewidths=1.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.5)
    ax.set_ylim(-0.05, 260)
    ax.set_yticks([0, 1, 10, 100])
    ax.set_yticklabels(["0", "1", "10", "100"])
    ax.set_xlim(3.2e-4, 0.35)
    ax.set_xticks([FLOOR, 1e-2, 1e-1])
    ax.set_xticklabels(["0", ".01", ".1"])
    ax.set_title(short, loc="left", fontsize=24, color=INK)
    ax.set_xlabel("anchor ratio")
    style.clean(ax)

axes[0].set_ylabel("basal-parameter\nerror (%)", fontsize=23)
axes[0].annotate("2.2% to 0.0%", (FLOOR, 0.0), xytext=(8.0e-4, 14.0),
                 fontsize=21, color=INK, fontweight=700,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))
axes[1].annotate("1.5% to 0.0%", (FLOOR, 0.0), xytext=(8.0e-4, 14.0),
                 fontsize=21, color=INK, fontweight=700,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))

h = [Line2D([], [], color=c, lw=3.0, marker=mk, ms=11, label=lab,
            markeredgecolor=SURFACE) for arm, (lab, c, mk) in ARMS.items()]
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=18.5,
           handlelength=1.6, borderaxespad=0.0, columnspacing=1.1)
fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
style.save(fig, "fig_dose_compact")
