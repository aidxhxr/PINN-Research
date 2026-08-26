"""The prospective test of the training-free design table.

m_h13: the prescribed protocol fixes the parameter; a LARGER perturbation that
misses the anchor by a hair is WORSE than doing nothing.
h13_b: the informative failure — the prescribed protocol reaches the anchor by
deleting the very parameter the anchor exists to protect. Two-clause rule.

Data: runs/20260802_protocol_test/protocol_{m_h13,h13_b}.json
"""
import json
import numpy as np
import style
from style import BLUE, ORANGE, GREY, RED, INK, INK2, SURFACE
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

style.setup()
RUN = "/home/29/aidahxr/PINN-Research/PINN-hybrid-ude/runs/20260802_protocol_test/"

PANELS = [
    ("protocol_m_h13.json", "m_h13 — basal a$_{13}$",
     ["none", "bcatKO", "mycKO"],
     ["baseline", "near-miss\ndouble KO", "prescribed\nmycKO"]),
    ("protocol_h13_b.json", "h13_b — basal W",
     ["none", "mycKO", "hox13KO"],
     ["baseline", "near-miss\ntriple KO", "prescribed\nhox13KO"]),
]
REGS = [("Normal", "N"), ("Severe APC Loss", "S")]

fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.6), sharex=True)
H = 0.34
for ax, (fname, title, protos, labels) in zip(axes, PANELS):
    d = json.load(open(RUN + fname))
    for i, p in enumerate(protos):
        y0 = len(protos) - 1 - i  # baseline on top
        for j, (reg, tag) in enumerate(REGS):
            row = next(r for r in d if r["regime"] == reg
                       and r["protocol"] == p)
            v = 100 * row["hybrid_basal_err"]
            reaches = row["anchor_ratio"] < 0.01
            color = BLUE if reaches else (RED if p != "none" else GREY)
            y = y0 + (0.19 if j == 0 else -0.19)
            ax.barh(y, v, height=H, color=color, zorder=3)
            ax.text(v + 1.8, y, f"{tag}  {v:.1f}", va="center", fontsize=20.5,
                    fontweight=700 if v < 1 else 400, color=INK, zorder=4)
    ax.set_yticks(range(len(protos)))
    ax.set_yticklabels(reversed(labels), fontsize=21.3, linespacing=1.3)
    ax.set_ylim(-0.62, len(protos) - 0.38)
    ax.set_xlim(0, 118)
    ax.set_title(title, loc="left", fontsize=23.6, pad=8, color=INK)
    ax.grid(axis="x")
    style.clean(ax, x=True, y=False)

axes[1].set_xlabel("basal-parameter error (%)")

# short callouts — the poster caption carries the full story
axes[0].text(73, 1.19, "bigger is not better", fontsize=20.5, color=RED,
             fontweight=600, va="center")
axes[1].text(76, 0.19, "reaches the anchor\nby deleting W itself",
             fontsize=20.5, color=INK, va="center")

# regime key in the free upper-right of the top panel
axes[0].text(116, 2.19, "N = Normal    S = Severe APC Loss", fontsize=20.5,
             color=INK2, ha="right", va="center")

fig.legend(handles=[Patch(color=GREY, label="baseline"),
                    Patch(color=RED, label="misses the anchor"),
                    Patch(color=BLUE, label="reaches the anchor")],
           loc="outside upper center", ncol=3, fontsize=21.3)
style.save(fig, "fig_prospective")
