"""The anchor-support diagram — the poster's conceptual centerpiece.

Per regulator: the range the data actually observe (normalized x/x_hi, so the
observed support is [ratio, 1]) versus the anchor at 0 where f(0)=0 is
asserted. No regulator ever approaches zero across all 10 conditions and all
4 regimes. A protocol from the training-free design table drives the floor to
0 -- except APC, whose production is bounded below.

Authored for poster scale: short labels only; the conclusion is carried by the
poster caption, not by the figure.
Data: runs/20260802_anchor_reach/anchor_reach.txt (worst over 4 regimes).
"""
import style
from style import BLUE, ORANGE, RED, INK, INK2, SURFACE
import matplotlib.pyplot as plt

style.setup()

# regulator label, 10-condition anchor ratio, protocol, structurally unreachable?
rows = [
    ("r   RA",       0.1434, "raKO",    False),
    ("b   b-cat",    0.0812, "bcatKO",  False),
    ("c   CYP26A1",  0.1806, "cypKO",   False),
    ("m   MYC",      0.1155, "mycKO",   False),
    ("h13 HOXA13",   0.1926, "hox13KO", False),
    ("apc APC",      0.1785, "none",    True),
]

fig, ax = plt.subplots(figsize=(10.4, 6.4))
H = 0.40
for i, (name, r10, proto, structural) in enumerate(rows):
    y = len(rows) - 1 - i
    rko = 0.0946 if structural else 0.0
    ax.barh(y, r10, left=0, height=H,
            color=RED if structural else ORANGE, alpha=0.22, zorder=2)
    ax.barh(y, 1 - r10, left=r10, height=H, color=BLUE, alpha=0.88, zorder=3)
    ax.plot([r10], [y], marker="|", ms=26, mew=3.4, color=INK, zorder=5)
    ax.text(r10 + 0.016, y, f"{r10:.3f}", fontsize=21, color=SURFACE,
            va="center", fontweight=600, zorder=6)

    c = RED if structural else BLUE
    ax.annotate("", xy=(max(rko, 0.002), y - 0.40), xytext=(r10, y - 0.40),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4,
                                shrinkA=0, shrinkB=0), zorder=5)
    ax.text(0.215, y - 0.40,
            proto if not structural else "no protocol reaches it",
            va="center", fontsize=21, color=c,
            family="IBM Plex Mono" if not structural else "Inter",
            fontweight=500)

ax.axvline(0, color=INK, lw=2.4, zorder=6)
ax.annotate("anchor\nf(0)=0", (0.0, len(rows) - 0.34), ha="center",
            fontsize=22, color=INK, fontweight=700, va="bottom",
            annotation_clip=False, linespacing=1.3)
ax.annotate("never observed", (0.115, len(rows) - 0.30), ha="left",
            fontsize=21, color=ORANGE, fontweight=600, va="bottom",
            annotation_clip=False)

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in reversed(rows)], family="IBM Plex Mono",
                   fontsize=22)
ax.set_xlim(-0.006, 1.0)
ax.set_ylim(-0.95, len(rows) + 0.45)
ax.set_xlabel("regulator value / its observed maximum")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(False)
style.clean(ax)
style.save(fig, "fig_support")
