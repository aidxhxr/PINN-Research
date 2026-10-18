"""Fig 7 -- the thetaP information gap and its design remedy.

(A) Cramer-Rao relative s.d. of thetaP (deltaP1 held known, the other 34
    parameters free) per regime and per design. Training-free FIM at the truth,
    sigma = 0.002, all seven states on 200 points.
    Source: experiments/11-fisher-information/runs/20260921_thetaP_design/thetaP_design.json
(B) Converged maximum-likelihood fit (mle.py, 36 free, best of 4 starts) under three
    condition sets of 10 / 12 / 12 conditions: error of the identifiable product
    deltaP1*(1-thetaP) (Early/Advanced/Severe; in Normal the true product is 0)
    and the number of the 36 parameters recovered within 10 %.
    Source: experiments/15-hybrid-ude/runs/20260921_odefit_thetaP/{base10,info12,dep12}/
    Cross-check: runs/20260921_aggregations/odefit_thetaP.txt
(C) Why: the log-sensitivity theta_P * d b(t) / d theta_P of beta-catenin in the
    control condition, per regime, from the shared model (src/wnt_pinn/model,
    the same LSODA solve thetaP_design.py uses for the FIM).
"""
import json
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style as st

sys.path.insert(0, os.path.join(st.REPO, "experiments/11-fisher-information"))
import thetaP_design as td  # noqa: E402  (shared model: src/wnt_pinn/model)

st.setup()
R = st.REPO
SRC_A = os.path.join(R, "experiments/11-fisher-information/runs/20260921_thetaP_design/thetaP_design.json")
REG = st.REGIMES
MK = {"Normal": "o", "Early Adenoma": "s", "Advanced Adenoma": "^", "Severe APC Loss": "D"}

# ---------------------------------------------------------------- data (A)
rows = json.load(open(SRC_A))
DESIGNS = [("ra6", "RA only\n(6)"),
           ("base10-wntInhib", "base $-$\nwntInhib (9)"),
           ("base10", "base\n(10)"),
           ("base10+midATRA", "+mid\nATRA"),
           ("base10+wntPulse2", "+WNT\npulse"),
           ("base10+kW=0.5", "0.5"),
           ("base10+kW=0.25", "0.25"),
           ("base10+kW=0.1", "0.1"),
           ("base10+kW=0.02", "0.02"),
           ("base10+kW=0", "0"),
           ("base10+bcatKO", "+$\\beta$-cat\nKO")]
A = {r: {d: 100 * next(x["relsd_thetaP"] for x in rows if x["regime"] == r and x["design"] == d)
         for d, _ in DESIGNS} for r in REG}
LOO_OTHER = {r: {x["design"]: 100 * x["relsd_thetaP"] for x in rows if x["regime"] == r
                 and x["design"].startswith("base10-") and x["design"] != "base10-wntInhib"} for r in REG}

# ---------------------------------------------------------------- data (B)
# Converged true MLE (unit weights, 36 free parameters, data seed 42), best of
# four starts (the common +50% start and three log-jittered ones) by final cost;
# experiments/17-sensitivity-pinn/runs/20261017_paper/ms/<regime>_init[_jN]_s42_none[_design].json
MS = os.path.join(R, "experiments/17-sensitivity-pinn/runs/20261017_paper/ms_tight")
ARMS_B = [("base10", "base, 10 conditions", st.GREY, "o", True),
          ("info12", "+2 information-matched", st.GREY, "s", False),
          ("dep12", "+2 depletion (WNT KD, RA-free)", st.BLUE, "D", True)]
SUFFIX = {"base10": "", "info12": "_info", "dep12": "_dep"}
TRUE = {r: {**td.BASELINE, **td.REGIMES[r]} for r in REG}
B = {}
for arm, *_ in ARMS_B:
    for r in REG:
        safe = r.replace(" ", "_")
        fs = [os.path.join(MS, f"{safe}_init{j}_s42_none{SUFFIX[arm]}.json")
              for j in ("", "_j1", "_j2", "_j3")]
        runs = [json.load(open(f)) for f in fs if os.path.exists(f)]
        assert len(runs) == 4, (arm, r, len(runs))
        d = min(runs, key=lambda x: x["cost"])
        rec, tru = d["recovered"], {k: TRUE[r][k] for k in d["recovered"]}
        pr = rec["deltaP1"] * (1 - rec["thetaP"]); pt = tru["deltaP1"] * (1 - tru["thetaP"])
        n = int(sum(abs(rec[k] - v) / abs(v) < 0.10 for k, v in tru.items()))
        B[(arm, r)] = dict(product_rec=pr, product_true=pt,
                           product_err_pct=(100 * abs(pr - pt) / pt if pt else None),
                           n_within10=n, n_params=len(tru), cost=d["cost"],
                           costs=[x["cost"] for x in runs],
                           thetaP_err_pct=100 * abs(rec["thetaP"] - tru["thetaP"]) / tru["thetaP"],
                           W_err_pct=100 * abs(rec["W"] - tru["W"]) / tru["W"])

# ---------------------------------------------------------------- data (C)
t = np.linspace(0.0, td.T, 400)
C = {}
for r in REG:
    p = {**td.BASELINE, **td.REGIMES[r]}
    th = p["thetaP"]; h = 1e-3 * th
    y = td.solve(p, t); yp = td.solve({**p, "thetaP": th + h}, t); ym = td.solve({**p, "thetaP": th - h}, t)
    S = th * (yp - ym) / (2 * h)
    C[r] = dict(b=y[:, 0], Sb=S[:, 0], Sall=np.sqrt((S ** 2).sum(1)))

# ================================================================ figure
fig = plt.figure(figsize=(st.COL2, 4.75))
gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1.0], width_ratios=[1.0, 1.0, 1.08],
                      hspace=0.55, wspace=0.42, left=0.075, right=0.985, top=0.95, bottom=0.10)
axA = fig.add_subplot(gs[0, :])
axB1 = fig.add_subplot(gs[1, 0]); axB2 = fig.add_subplot(gs[1, 1]); axC = fig.add_subplot(gs[1, 2])

# ---- A
x = np.arange(len(DESIGNS))
spans = [(-0.5, 1.5, "fewer conditions", "#f3f3f3"), (2.5, 4.5, "+1 information-\nmatched", "#f3f3f3"),
         (4.5, 10.5, "+1 WNT knockdown, residual WNT $k_W$", "#e8f0f8")]
for lo, hi, lab, c in spans:
    axA.axvspan(lo, hi, color=c, zorder=0, lw=0)
    axA.text((lo + hi) / 2, 320, lab, ha="center", va="top", fontsize=6.6, color=st.INK2, linespacing=0.95)
for i, r in enumerate(REG):
    y = np.array([A[r][d] for d, _ in DESIGNS])
    # base line through the base / +1 designs; ra6 and LOO joined by a dotted lead-in
    axA.plot(x[2:], y[2:], color=st.REG_COLOR[r], lw=1.2, zorder=3)
    axA.plot(x[:3], y[:3], color=st.REG_COLOR[r], lw=0.8, ls=(0, (1.5, 1.5)), zorder=3)
    axA.scatter(x, y, s=16, marker=MK[r], color=st.REG_COLOR[r], edgecolors="white", linewidths=0.5,
                zorder=4, label=st.REG_SHORT[r])
axA.axhline(10, color=st.INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
axA.text(10.45, 10 * 1.12, "10 %", fontsize=6.5, color=st.INK2, ha="right", va="bottom")
axA.set_yscale("log"); axA.set_ylim(8e-3, 400)
axA.set_yticks([0.01, 0.1, 1, 10, 100]); axA.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
axA.set_xticks(x); axA.set_xticklabels([l for _, l in DESIGNS], fontsize=6.6, linespacing=0.95)
axA.set_xlim(-0.5, len(DESIGNS) - 0.5)
axA.grid(axis="x", visible=False)
axA.set_ylabel("Cramér–Rao rel. s.d. of $\\theta_P$ (%)")
# fold annotations for Severe: base -> kW=0.1
s0, s1 = A["Severe APC Loss"]["base10"], A["Severe APC Loss"]["base10+kW=0.1"]
axA.text(8.0, 3.6, f"Severe: {s0:.0f} % $\\to$ {s1:.1f} % at $k_W$ = 0.1 ($\\times${s1 / s0:.3f})",
         fontsize=6.6, color=st.REG_COLOR["Severe APC Loss"], ha="center", va="bottom")
axA.legend(loc="lower left", ncol=4, fontsize=6.6, handletextpad=0.2, columnspacing=0.9,
           bbox_to_anchor=(0.0, 0.0), markerscale=1.1)
st.panel(axA, "A", x=-0.06, y=1.0)

# ---- B1: product error (E/A/S)
regB = REG[1:]
W = 0.26
for j, (arm, lab, c, m, filled) in enumerate(ARMS_B):
    xs = np.arange(len(regB)) + (j - 1) * W
    ys = [B[(arm, r)]["product_err_pct"] for r in regB]
    axB1.vlines(xs, 1e-2, ys, color=c, lw=0.8, zorder=2)
    axB1.scatter(xs, ys, s=20, marker=m, color=c if filled else "white", edgecolors=c,
                 linewidths=0.9, zorder=3, label=lab)
    for xx, yy in zip(xs, ys):
        axB1.text(xx, yy * 1.35, f"{yy:.1f}" if yy >= 0.1 else f"{yy:.2f}", ha="center", va="bottom",
                  fontsize=5.8, color=st.INK2)
axB1.set_yscale("log"); axB1.set_ylim(1e-2, 900)
axB1.set_yticks([0.01, 0.1, 1, 10, 100]); axB1.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
axB1.axhline(10, color=st.INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
axB1.set_xticks(range(len(regB))); axB1.set_xticklabels([st.REG_SHORT[r] for r in regB])
for tl, r in zip(axB1.get_xticklabels(), regB):
    tl.set_color(st.REG_COLOR[r])
axB1.grid(axis="x", visible=False); axB1.set_xlim(-0.5, len(regB) - 0.5)
axB1.set_ylabel("error of $\\delta_{P1}(1-\\theta_P)$ (%)")
axB1.set_title("MLE: identifiable product", fontsize=7.5, loc="left")
st.panel(axB1, "B", x=-0.30, y=1.04)

# ---- B2: the bound on the product, per design (150-point benchmark design, fim2_*)
F2 = os.path.join(R, "experiments/17-sensitivity-pinn/runs/20261017_paper")
FIM2 = {arm: json.load(open(os.path.join(F2, f"fim2_{d}", "fim_summary.json")))
        for arm, d in (("base10", "base"), ("info12", "info"), ("dep12", "dep"))}
B2 = {}
for j, (arm, lab, c, m, filled) in enumerate(ARMS_B):
    xs = np.arange(len(regB)) + (j - 1) * W
    ys = []
    for r in regB:
        th = td.REGIMES[r]["thetaP"]
        ys.append(100 * FIM2[arm][r]["sd35_none"]["thetaP"] * th / (1 - th))  # sd(log kappa_P)
    B2[arm] = dict(zip(regB, ys))
    axB2.bar(xs, ys, W * 0.92, color=c if filled else "white", edgecolor=c, linewidth=0.8, zorder=2)
    errs = [B[(arm, r)]["product_err_pct"] for r in regB]
    axB2.scatter(xs, errs, s=10, marker="_", color=st.INK, linewidths=1.2, zorder=3)
axB2.set_yscale("log"); axB2.set_ylim(0.05, 90)
axB2.set_yticks([0.1, 1, 10]); axB2.set_yticklabels(["0.1", "1", "10"])
axB2.set_xticks(range(len(regB))); axB2.set_xticklabels([st.REG_SHORT[r] for r in regB], fontsize=6.6)
for tl, r in zip(axB2.get_xticklabels(), regB):
    tl.set_color(st.REG_COLOR[r])
axB2.grid(axis="x", visible=False); axB2.set_xlim(-0.55, len(regB) - 0.45)
axB2.set_ylabel(r"CRLB of $\delta_{P1}(1-\theta_P)$ (%)")
axB2.set_title("bound (bars), fit (ticks)", fontsize=7.5, loc="left")
hB = [Line2D([], [], marker=m, color=c, markerfacecolor=(c if f else "white"), markeredgecolor=c,
             lw=0, markersize=4.2, label=l) for _, l, c, m, f in ARMS_B]
fig.legend(handles=hB, loc="lower center", ncol=3, bbox_to_anchor=(0.365, -0.005), fontsize=6.6,
           handletextpad=0.25, columnspacing=1.0)

# ---- C
for r in REG:
    axC.plot(t, np.abs(C[r]["Sb"]), color=st.REG_COLOR[r], lw=1.1, label=st.REG_SHORT[r])
axC.set_yscale("log"); axC.set_ylim(1e-2, 5)
axC.set_xlim(0, 150); axC.set_xticks([0, 50, 100, 150])
axC.set_xlabel("time $t$")
axC.set_ylabel("$|\\theta_P\\,\\partial b/\\partial\\theta_P|$")
axC.set_title("why: $\\beta$-catenin stops seeing $\\theta_P$", fontsize=7.5, loc="left")
for r, yv in zip(REG, [C[r] for r in REG]):
    pk = float(np.median(np.abs(yv["Sb"])[t >= 10]))
    axC.text(152, pk, f"{pk:.2g}", fontsize=5.8, color=st.REG_COLOR[r], va="center", ha="left",
             clip_on=False)
axC.legend(loc="lower right", ncol=2, fontsize=6.2, handlelength=1.1, columnspacing=0.7,
           handletextpad=0.3)
st.panel(axC, "C", x=-0.30, y=1.04)

# ================================================================ audit
audit = {
    "figure": "fig07_thetaP",
    "panels": {
        "A": {"source": os.path.relpath(SRC_A, R), "units": "percent CRLB rel. sd of thetaP, deltaP1 known",
              "values": A, "leave_one_out_other_conditions": LOO_OTHER,
              "ratio_kW0.1_over_base10": {r: A[r]["base10+kW=0.1"] / A[r]["base10"] for r in REG},
              "ratio_info_over_base10": {r: {d: A[r][d] / A[r]["base10"] for d in ("base10+midATRA",
                                                                                     "base10+wntPulse2")}
                                         for r in REG}},
        "B": {"source": os.path.relpath(MS, R),
              "estimator": "unit-weight MLE, 36 free, best of 4 starts by cost, data seed 42",
              "cells": {f"{a}/{r}": v for (a, r), v in B.items()},
              "crlb_kappaP_pct_150pt": B2},
        "C": {"source": "src/wnt_pinn/model via experiments/11-fisher-information/thetaP_design.solve (LSODA rtol 1e-9), "
                        "ctrl condition, central difference h = 1e-3 thetaP",
              "peak_abs_log_sensitivity_b": {r: float(np.abs(C[r]["Sb"]).max()) for r in REG},
              "median_abs_log_sensitivity_b_t_ge_10 (printed)": {r: float(np.median(np.abs(C[r]["Sb"])[t >= 10]))
                                                                 for r in REG},
              "b_range": {r: [float(C[r]["b"].min()), float(C[r]["b"].max())] for r in REG}},
    },
}
st.save(fig, "fig07_thetaP", audit)
print(json.dumps(audit["panels"]["C"], default=float)[:600])
