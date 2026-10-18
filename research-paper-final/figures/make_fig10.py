"""Fig 9 (panels from the screens; the hybrid-Fisher panel is made separately).

Equation-local screen (exact states, gated net, 3 starts, same seeds in every
arm), four learned production edges x {Normal, Severe APC Loss}:
  base   10 base conditions               runs/20260801_depletion_ab/screen_nodep/screen.json
  info   10 + 2 information-matched       runs/20260801_infoctl/screen.json
  dep    10 + 2 depletion (wntKO, raKO)   runs/20260801_depletion_ab/screen_dep/screen.json
(A) basal-production error, base design -> + depletion pair, with the
    mechanistic control on the 12 depletion conditions; the achieved anchor
    ratio rho = x_lo/x_hi of the regulator is printed under each cell.
(B) attribution: the ratio arm / base of the basal error for the
    information-matched pair and for the depletion pair (same size, same kind
    of condition), plus the count of equation parameters within 10 %.
Cross-checked against runs/20260801_infoctl/comparison.txt and
runs/20260801_depletion_ab/AB_comparison.txt (1 dp).
"""
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style as st

st.setup()
R = st.REPO
RUNS = os.path.join(R, "experiments/15-hybrid-ude/runs")
SRC = {"base": os.path.join(RUNS, "20260801_depletion_ab/screen_nodep/screen.json"),
       "info": os.path.join(RUNS, "20260801_infoctl/screen.json"),
       "dep": os.path.join(RUNS, "20260801_depletion_ab/screen_dep/screen.json")}
TXT = [os.path.join(RUNS, "20260801_infoctl/comparison.txt"),
       os.path.join(RUNS, "20260801_depletion_ab/AB_comparison.txt")]
EDGES = ["ra_h5", "rc_cyp", "bm_myc", "bc_cyp"]
ELAB = {"ra_h5": "ra_h5 ($a_5$)\nRA$\\to$HOXA5", "rc_cyp": "rc_cyp ($a_C$)\nRA$\\to$CYP26",
        "bm_myc": "bm_myc ($a_M$)\n$\\beta$-cat$\\to$MYC", "bc_cyp": "bc_cyp ($a_C$)\n$\\beta$-cat$\\to$CYP26"}
REGS = ["Normal", "Severe APC Loss"]
RMK = {"Normal": "o", "Severe APC Loss": "D"}

data = {k: json.load(open(p)) for k, p in SRC.items()}


def cell(arm, e, r):
    rows = [x for x in data[arm] if x["term"] == e and x["regime"] == r]
    assert len(rows) == 1
    x = rows[0]
    return dict(basal=100 * x["basal_err"], ctrl=100 * x["ctrl_basal_err"], nrmse=100 * x["term_nrmse"],
                eq=x["eq_under10"], n_eq=x["n_eq_params"], ratio=x["x_lo"] / x["x_hi"], seed=x["seed"])


V = {(a, e, r): cell(a, e, r) for a in SRC for e in EDGES for r in REGS}
# cross-check against the comparison tables (basal %, 1 dp)
pairs = {"12dep": "dep", "12cond": "dep", "12info": "info", "10cond": "base"}
for p in TXT:
    A_ = B_ = None
    for line in open(p):
        m = re.match(r"^A = (\w+)\s+B = (\w+)", line)
        if m:
            A_, B_ = pairs[m.group(1)], pairs[m.group(2)]
            continue
        m = re.match(r"^(\w+)\s+(Normal|Severe APC Loss)\s+gated\s+([\d.]+)%\s+([\d.]+)%\s+\S+\s+([\d.]+)%\s+"
                     r"([\d.]+)%", line)
        if m and A_:
            e, r = m.group(1), m.group(2)
            assert abs(V[(A_, e, r)]["basal"] - float(m.group(5))) < 0.051, (p, line)
            assert abs(V[(B_, e, r)]["basal"] - float(m.group(6))) < 0.051, (p, line)

FLOOR = 0.05
fig = plt.figure(figsize=(st.COL2, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.28, left=0.075, right=0.99, top=0.90,
                      bottom=0.255)
axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])
DX = 0.19
xpos = {(e, r): i + (-DX if r == "Normal" else DX) for i, e in enumerate(EDGES) for r in REGS}

# ---- A
for e in EDGES:
    for r in REGS:
        x = xpos[(e, r)]
        b, d = V[("base", e, r)], V[("dep", e, r)]
        worse = d["basal"] > 1.5 * b["basal"]
        axA.annotate("", xy=(x, d["basal"]), xytext=(x, b["basal"]),
                     arrowprops=dict(arrowstyle="-|>", lw=0.7, mutation_scale=6,
                                     color=(st.RED if worse else ("#9ab8d6" if d["basal"] < b["basal"] / 1.5 else st.LIGHT)), shrinkA=2.5, shrinkB=2.5),
                     zorder=2)
        axA.scatter(x, b["basal"], s=20, marker=RMK[r], color=st.ORANGE, edgecolors="white", lw=0.4, zorder=4)
        axA.scatter(x, d["basal"], s=20, marker=RMK[r], color=st.BLUE, edgecolors="white",
                    lw=0.4, zorder=4)
        axA.scatter(x + 0.085, d["ctrl"], s=12, marker=RMK[r], facecolors="white",
                    edgecolors=st.GREY, lw=0.7, zorder=3)
        rb, rd = b["ratio"], d["ratio"]
        axA.text(x, 1.12e-3, f"{rb:.3f}\n$\\to${'0' if rd < 1e-9 else f'{rd:.3f}'[1:]}", ha="center",
                 va="bottom", fontsize=5.4, color=st.INK2, linespacing=0.9)
axA.text(-0.64, 1.9e-3, "$\\rho$", fontsize=7, color=st.INK2, ha="center", va="center")
axA.set_yscale("log"); axA.set_ylim(1e-3, 300)
axA.set_yticks([0.01, 0.1, 1, 10, 100]); axA.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
axA.axhline(10, color=st.INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
axA.set_xticks(range(len(EDGES))); axA.set_xticklabels([ELAB[e] for e in EDGES], fontsize=6.4, linespacing=1.0)
axA.tick_params(axis="x", length=0)
axA.set_xlim(-0.75, len(EDGES) - 0.45); axA.grid(axis="x", visible=False)
axA.set_ylabel("basal parameter error (%)")
axA.set_title("depletion pair visits the anchor", fontsize=7.5, loc="left")
axA.text(3.0, 6.0, "$\\beta$-catenin never\nreaches 0", fontsize=6.0, color=st.INK2, ha="center", va="top")
st.panel(axA, "A", x=-0.12, y=1.03)

# ---- B
ARMS = [("info", "+2 information-matched", st.ORANGE, "s", False),
        ("dep", "+2 depletion", st.BLUE, "D", True)]
for e in EDGES:
    for r in REGS:
        x0 = xpos[(e, r)]
        for k, (arm, lab, c, m, filled) in enumerate(ARMS):
            ratio = V[(arm, e, r)]["basal"] / V[("base", e, r)]["basal"]
            x = x0 + (k - 0.5) * 0.15
            axB.vlines(x, 1.0, ratio, color=c, lw=0.8, zorder=2)
            axB.scatter(x, ratio, s=18, marker=RMK[r], color=(c if filled else "white"), edgecolors=c,
                        lw=0.8, zorder=4)
axB.axhline(1.0, color=st.INK2, lw=0.6, zorder=1)
axB.set_yscale("log"); axB.set_ylim(2e-3, 40)
axB.set_yticks([0.01, 0.1, 1]); axB.set_yticklabels(["0.01", "0.1", "1"])
axB.set_xticks(range(len(EDGES))); axB.set_xticklabels([e for e in EDGES], fontsize=6.6)
axB.tick_params(axis="x", length=0)
axB.set_xlim(-0.55, len(EDGES) - 0.45); axB.grid(axis="x", visible=False)
axB.set_ylabel("basal error, arm / base design")
axB.set_title("attribution: same-sized pair that misses the anchor", fontsize=7.5, loc="left")
cnt = {a: (sum(V[(a, e, r)]["eq"] for e in EDGES for r in REGS),
           sum(V[(a, e, r)]["n_eq"] for e in EDGES for r in REGS)) for a in SRC}
axB.text(0.02, 0.98, "equation parameters within 10 %:\n"
         f"base {cnt['base'][0]}/{cnt['base'][1]}   info-matched {cnt['info'][0]}/{cnt['info'][1]}   "
         f"depletion {cnt['dep'][0]}/{cnt['dep'][1]}", transform=axB.transAxes, fontsize=6.2, color=st.INK,
         va="top", ha="left")
st.panel(axB, "B", x=-0.13, y=1.03)

h = [Line2D([], [], marker="o", color=st.ORANGE, lw=0, ms=4, label="hybrid, 10 base conditions"),
     Line2D([], [], marker="o", mfc="white", mec=st.ORANGE, lw=0, ms=4, label="+2 information-matched"),
     Line2D([], [], marker="o", color=st.BLUE, lw=0, ms=4, label="+2 depletion (wntKO, raKO)"),
     Line2D([], [], marker="o", mfc="white", mec=st.GREY, lw=0, ms=3.5, label="mechanistic control, 12 cond."),
     Line2D([], [], marker="o", color=st.INK2, lw=0, ms=3.5, label="Normal"),
     Line2D([], [], marker="D", color=st.INK2, lw=0, ms=3.2, label="Severe"),
     Line2D([], [], color=st.RED, lw=0.8, label="error increases")]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=6.4,
           handletextpad=0.25, columnspacing=1.1)

audit = {"figure": "fig10_depletion", "units": "percent", "sources": {k: os.path.relpath(p, R)
                                                                            for k, p in SRC.items()},
         "cross_check": [os.path.relpath(p, R) for p in TXT],
         "cells": {f"{a}/{e}/{r}": V[(a, e, r)] for a in SRC for e in EDGES for r in REGS},
         "ratio_arm_over_base": {f"{a}/{e}/{r}": V[(a, e, r)]["basal"] / V[("base", e, r)]["basal"]
                                 for a in ("info", "dep") for e in EDGES for r in REGS},
         "eq_params_within10": {a: f"{c[0]}/{c[1]}" for a, c in cnt.items()},
         "log_floor_pct": "none needed (min plotted value 0.007 %)",
         "note": "rho = x_lo/x_hi of the edge's regulator over the arm's conditions; the information-matched pair "
                 "leaves rho unchanged in every cell. bc_cyp gets worse under depletion (b reaches only "
                 "rho 0.013-0.015)."}
st.save(fig, "fig10_depletion", audit)
print(cnt)
