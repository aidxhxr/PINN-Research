"""Five constraint parameterisations, all failing — architecture is not the fix.

Screened on bm_myc, Normal, one start, matched budget (round-2 note table).
Mechanistic control: 0.0%. The monotone+bounded construction the UDE
literature recommends (sc / sc_bounded) does not rescue the basal parameter.
"""
import style
from style import BLUE, ORANGE, GREY, RED, INK, INK2
import matplotlib.pyplot as plt

style.setup()

# (label, note, basal aM error %, functional NRMSE %)
rows = [
    ("gated",       "softplus · exponential gate",      14.1, 3.0),
    ("sc",          "monotone + exact anchor",          24.0, 5.1),
    ("sc_bounded",  "+ oracle upper bound",             13.8, 2.9),
    ("lin",         "anchor linear in x",               22.8, 4.8),
    ("lin_mono",    "monotone rate (cannot saturate)", 202.8, 35.1),
]

fig, ax = plt.subplots(figsize=(10.4, 5.8))
H = 0.40
for i, (lab, note, basal, fn) in enumerate(rows):
    y = len(rows) - 1 - i
    color = RED if basal > 100 else ORANGE
    ax.barh(y, basal, height=H, color=color, zorder=3)
    ax.text(basal + 3.5, y, f"{basal:.1f}%", va="center", fontsize=23.6,
            fontweight=700, color=INK)
    ax.text(3.0, y + 0.30, note, fontsize=20.0, color=INK2, va="bottom")

ax.axvline(0, color=INK, lw=1.4, zorder=4)
# mechanistic control reference
ax.annotate("mechanistic control: 0.0%", (0, -0.72), fontsize=21.3,
            color=BLUE, fontweight=600, va="center", annotation_clip=False)
ax.plot([0, 0], [-0.5, len(rows) - 0.45], color=BLUE, lw=2.5, zorder=5,
        solid_capstyle="butt")

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in reversed(rows)], family="IBM Plex Mono",
                   fontsize=23.6)
ax.set_xlim(0, 235)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_ylim(-0.95, len(rows) - 0.30)
ax.set_xlabel("basal-parameter a$_M$ error (%)")
ax.grid(axis="x")
style.clean(ax)
style.save(fig, "fig_param_fail")
