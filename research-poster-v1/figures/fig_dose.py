"""THE HERO FIGURE -- the dose-response, with the condition count fixed at 11.

Rows = regime, columns = the same points plotted against DOSE and against
ANCHOR RATIO. In Normal the two edges collapse onto one curve against ratio
(median gap 0.4pp) but not against dose. In Severe the collapse is partial --
shown, not hidden: that pre-registered prediction failed.

Authored for poster scale: short labels; the poster caption carries the
conclusion.
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
    "ra":   ("retinoid restriction   ra_h5, a$_5$", BLUE, "o"),
    "wnt":  ("WNT knockdown only   bm_myc, a$_M$", ORANGE, "s"),
    "bcat": ("WNT + one more siRNA   same edge", RED, "D"),
}
FLOOR = 5e-4

fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.2), sharey=True)
for i, reg in enumerate(("Normal", "Severe APC Loss")):
    for j, xkey in enumerate(("dose", "anchor_ratio")):
        ax = axes[i, j]
        for arm, (lab, c, mk) in ARMS.items():
            pts = sorted([r for r in recs if r["arm"] == arm
                          and r["regime"] == reg], key=lambda r: r[xkey])
            if not pts:
                continue
            x = [max(r[xkey], FLOOR) for r in pts]
            y = [100 * r["hybrid_basal_err"] for r in pts]
            ax.plot(x, y, "-", color=c, lw=2.6, zorder=3, alpha=0.92,
                    solid_capstyle="round")
            ax.scatter(x, y, s=105, color=c, marker=mk, zorder=4,
                       edgecolors=SURFACE, linewidths=1.8)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.5)
        ax.set_ylim(-0.05, 260)
        ax.set_yticks([0, 1, 10, 100])
        ax.set_yticklabels(["0", "1", "10", "100"])
        ax.set_xlim(3.2e-4, 1.9)
        ax.set_xticks([FLOOR, 1e-2, 1e-1, 1])
        ax.set_xticklabels(["0", ".01", ".1", "1"])
        style.clean(ax)
        ax.tick_params(labelbottom=(i == 1))
        if j == 0:
            ax.set_ylabel(f"{reg.replace(' APC Loss', '')}\nbasal error (%)",
                          fontsize=23)

axes[1, 0].set_xlabel("dose  k")
axes[1, 1].set_xlabel("anchor ratio")
axes[0, 0].set_title("vs DOSE", loc="left", fontsize=26, color=INK)
axes[0, 1].set_title("vs ANCHOR RATIO", loc="left", fontsize=26, color=INK)

axes[0, 1].annotate("2.2% to 0.0%", (FLOOR, 0.0), xytext=(9.0e-4, 12.0),
                    fontsize=21, color=INK, fontweight=700,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
axes[1, 1].annotate("1.5% to 0.0%", (FLOOR, 0.0), xytext=(8.0e-4, 2.4),
                    fontsize=21, color=INK, fontweight=700,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
axes[1, 0].annotate("one bad\nrestart", (0.1, 23.4), xytext=(6.5e-3, 105),
                    fontsize=19, color=INK2,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
axes[1, 1].annotate("collapse only\npartial here", (0.016, 23.4),
                    xytext=(5.4e-4, 105), fontsize=19, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

h = [Line2D([], [], color=c, lw=3.0, marker=mk, ms=11, label=lab,
            markeredgecolor=SURFACE)
     for arm, (lab, c, mk) in ARMS.items()]
fig.legend(handles=h, loc="lower center", ncol=1, fontsize=21,
           handlelength=2.0, borderaxespad=0.6, labelspacing=0.30)
fig.get_layout_engine().set(rect=(0, 0.155, 1, 1.0))
style.save(fig, "fig_dose")
