"""The attribution test: information-matched control vs anchor-reaching depletion.

Two extra conditions that move every regulator UP (no anchor gets closer)
change nothing in 8 of 8 cells; the same-sized pair that REACHES the anchor
collapses the error. Data: runs/20260801_infoctl/comparison.txt.
"""
import style
from style import BLUE, ORANGE, GREY, INK, INK2, SURFACE
import matplotlib.pyplot as plt

style.setup()

# edge, regime, basal error: 10 cond / 12 info-matched / 12 depletion
rows = [
    ("ra_h5",  "Normal", 17.0, 12.9, 0.1),
    ("ra_h5",  "Severe", 33.5, 33.4, 0.7),
    ("rc_cyp", "Normal", 10.9, 10.9, 1.1),
    ("rc_cyp", "Severe", 20.3, 20.8, 0.3),
    ("bm_myc", "Normal", 19.2, 20.0, 2.9),
    ("bm_myc", "Severe", 78.9, 82.5, 2.5),
    ("bc_cyp", "Normal", 24.6, 26.9, 25.5),
    ("bc_cyp", "Severe", 30.0, 46.1, 64.8),
]

fig, ax = plt.subplots(figsize=(10.4, 5.4))
for i, (edge, reg, base, info, dep) in enumerate(rows):
    y = len(rows) - 1 - i
    # connector: baseline -> info (grey, "nothing happens")
    ax.plot([base, info], [y, y], color=GREY, lw=2.0, alpha=0.55, zorder=2,
            solid_capstyle="round")
    # connector: baseline -> depletion (blue, the gain)
    ax.plot([base, dep], [y, y], color=BLUE, lw=2.0, alpha=0.30, zorder=2,
            solid_capstyle="round")
    ax.scatter([base], [y], s=153.0, color=GREY, zorder=4, edgecolors=SURFACE,
               linewidths=2)
    ax.scatter([info], [y], s=153.0, color=ORANGE, zorder=5, edgecolors=SURFACE,
               linewidths=2)
    ax.scatter([dep], [y], s=187.0, color=BLUE, zorder=6, edgecolors=SURFACE,
               linewidths=2)

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{e} {r}" for e, r, *_ in reversed(rows)],
                   family="IBM Plex Mono", fontsize=22.0)
ax.set_xlim(-2, 90)
ax.set_xlabel("basal-parameter error (%)")
ax.set_ylim(-0.7, len(rows) - 0.40)
style.clean(ax)

# legend stacked in the empty upper-right region
for y0, lab, c in ((7.15, "10 conditions", GREY),
                   (6.45, "+2 that MISS the anchor", ORANGE),
                   (5.75, "+2 that REACH it", BLUE)):
    ax.scatter([47], [y0], s=187.0, color=c, edgecolors=SURFACE, linewidths=2,
               zorder=6)
    ax.text(50, y0, lab, fontsize=21.3, va="center", color=INK, fontweight=600)

# call out the one edge whose anchor neither arm reaches
ax.annotate("anchor never reached", (65.5, 0.22), xytext=(58, 1.05),
            fontsize=20.0, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
style.save(fig, "fig_attribution")
