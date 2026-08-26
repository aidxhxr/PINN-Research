"""FIM identifiability verdict per regime — stacked composition of 36 params.

Data: PINN-fisher-matrix/runs/20260711_203325_fisher/*_fim_summary.json and
*_correlated_pairs.csv (authoritative post-rename run).
IDENT=blue / WEAK=orange / NON-IDENT=red (professor's mapping).
"""
import json, csv, os
import style
from style import BLUE, ORANGE, RED, INK, INK2, SURFACE
import matplotlib.pyplot as plt

style.setup()
RUN = "/home/29/aidahxr/PINN-Research/PINN-fisher-matrix/runs/20260711_203325_fisher/"
REGIMES = ["Normal", "Early_Adenoma", "Advanced_Adenoma", "Severe_APC_Loss"]
SHORT = ["Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss"]

rows = []
for r in REGIMES:
    d = json.load(open(RUN + r + "_fim_summary.json"))
    with open(RUN + r + "_correlated_pairs.csv") as f:
        npairs = max(0, sum(1 for _ in csv.reader(f)) - 1)
    rows.append((d["n_ident"], d["n_weak"], d["n_nonident"],
                 d["cond_number"], npairs, d["nonident_params"]))

fig, ax = plt.subplots(figsize=(10.4, 4.2))
H, GAP = 0.52, 0.35
for i, (ni, nw, nn, cond, npairs, bad) in enumerate(rows):
    y = len(rows) - 1 - i
    x = 0
    for val, c in ((ni, BLUE), (nw, ORANGE), (nn, RED)):
        if val:
            ax.barh(y, val - GAP, left=x + GAP / 2, height=H, color=c, zorder=3)
            if val >= 3:
                ax.text(x + val / 2, y, str(val), ha="center", va="center",
                        color="white", fontsize=22.8, fontweight=700, zorder=4)
        x += val
    # right-side annotation: condition number + correlated pairs
    ax.text(37.0, y + 0.16, f"cond {cond:.1e}".replace("e+", "e"),
            fontsize=20.5, family="IBM Plex Mono", color=INK2, va="center")
    ax.text(37.0, y - 0.22, f"{npairs} corr. pairs", fontsize=20.5,
            family="IBM Plex Mono", color=INK, va="center", fontweight=500)

ax.set_yticks(range(len(rows)))
ax.set_yticklabels(reversed(SHORT), fontsize=22.8)
ax.set_xlim(0, 46)
ax.set_xticks([0, 9, 18, 27, 36])
ax.set_ylim(-0.55, len(rows) - 0.45)
ax.set_xlabel("of 36 parameters")
ax.grid(False)
style.clean(ax)

# legend as colored keys in one row above
for x0, lab, c in ((0, "IDENT", BLUE), (9.5, "WEAK", ORANGE), (18.5, "NON-IDENT", RED)):
    ax.scatter([x0 + 0.6], [len(rows) - 0.10], s=238.0, color=c, marker="s",
               clip_on=False)
    ax.text(x0 + 2.0, len(rows) - 0.10, lab, fontsize=21.3, va="center",
            color=INK, fontweight=600)
ax.set_ylim(-0.55, len(rows) + 0.25)

style.save(fig, "fig_fim")
