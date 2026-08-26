"""The anchor-support diagram — the poster's conceptual centerpiece.

Per regulator: the range the data actually observe (normalized x/x_hi, so the
observed support is [ratio, 1]) vs the anchor at 0 where f(0)=0 is asserted.
A knockout protocol (from the training-free design table anchor_reach.py)
drives the floor to 0 — except APC, whose production is bounded below.

Data: runs/20260802_anchor_reach/anchor_reach.txt (worst ratio over 4 regimes).
"""
import style
from style import BLUE, ORANGE, GREY, RED, INK, INK2, SURFACE
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

style.setup()

# regulator, 10-cond ratio, ratio under cheapest reaching protocol, protocol
rows = [
    ("r   (RA)",        0.1434, 0.0000, "raKO",    False),
    ("b   (β-cat)",     0.0812, 0.0000, "bcatKO",  False),
    ("c   (CYP26A1)",   0.1806, 0.0000, "cypKO",   False),
    ("m   (MYC)",       0.1155, 0.0000, "mycKO",   False),
    ("h13 (HOXA13)",    0.1926, 0.0000, "hox13KO", False),
    ("apc (APC)",       0.1785, 0.0946, "—",       True),   # structural floor
]

fig, ax = plt.subplots(figsize=(10.4, 4.6))
H = 0.30
for i, (name, r10, rko, proto, structural) in enumerate(rows):
    y = len(rows) - 1 - i
    # unobserved gap: 0 .. r10 (wash), observed support: r10 .. 1 (solid-ish)
    ax.barh(y, r10, left=0, height=H, color=RED if structural else ORANGE,
            alpha=0.15, zorder=2)
    ax.barh(y, 1 - r10, left=r10, height=H, color=BLUE, alpha=0.82, zorder=3)
    # data floor marker
    ax.plot([r10], [y], marker="|", ms=22, mew=3, color=INK, zorder=5)
    # where the protocol takes the floor
    color = RED if structural else BLUE
    ax.annotate("", xy=(rko + 0.002, y - 0.30), xytext=(r10, y - 0.30),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                shrinkA=0, shrinkB=0))
    label = ("floor 0.095 — structural" if structural
             else f"{proto} → 0")
    ax.annotate(label, (max(rko, 0.004) + 0.008, y - 0.30), va="center",
                fontsize=14.5, color=color, family="IBM Plex Mono",
                fontweight=500)

ax.axvline(0, color=INK, lw=1.6, zorder=4)
ax.annotate("anchor\nf(0) = 0", (0, len(rows) - 0.30), ha="center",
            fontsize=15, color=INK, fontweight=600, va="bottom",
            annotation_clip=False)
ax.annotate("observed support of the 10 base conditions", (0.60, len(rows) - 0.42),
            fontsize=14.5, color=SURFACE, ha="center", fontweight=600, zorder=6)
ax.annotate("no data ever here", (0.088, len(rows) - 0.42), fontsize=14.5,
            color=ORANGE, ha="center", fontweight=600, zorder=6)

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in reversed(rows)], family="IBM Plex Mono",
                   fontsize=15)
ax.set_xlim(-0.015, 1.0)
ax.set_ylim(-0.75, len(rows) + 0.15)
ax.set_xlabel("regulator value, normalized to its observed maximum  x / x_hi")
ax.grid(False)
style.clean(ax)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

style.save(fig, "fig_support")
