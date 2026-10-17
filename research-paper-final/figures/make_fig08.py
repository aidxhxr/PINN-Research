"""Fig 8 -- a learned term whose anchor the data do not observe.

(A) the learned MYC activation f(b) (beta-catenin -> MYC, gated net) of
    the round-1 full-pipeline hybrid run vs the true Hill term, Early Adenoma,
    evaluated from the saved net on [0, b_max]; the data support [b_lo, b_hi]
    is shaded.  The net honours f(0)=0 exactly but lifts f by a near-constant
    over the support, and the basal production a_M drops by the same amount.
    Source: PINN-hybrid-ude/runs/20260727_012448_bm_myc/<regime>_{term_bm_myc.pt,term.json,recovered.json}
    (the curves cached in research-poster-latex-hybrid/data/hybrid_comparison.json
    are checked against the net on the support).
(B) the compensation identity  mean_support[f_hat - f] = a_M - a_M_hat
    for every bm_myc fit: the five-form screen (exact states; gated, sc,
    sc_bounded, lin; lin_mono excluded -- it fails through eps_M instead) and
    the round-1 full pipeline (estimated states).
    Source: PINN-hybrid-ude/runs/20260921_screen_5form/{screen.json,*.pt}
(C) The edge atlas: basal-production parameter error of the seven production
    edges, hybrid (gated, sc) vs the mechanistic control fit on the same 10
    conditions, 4 regimes.  Normal parsed from
    runs/20260801_204046_screen_atlas/screen.log (0.1 % precision; that run
    crashed before writing json), other regimes from
    runs/20260802_screen_atlas_rest/screen.json; cross-checked against
    runs/20260921_aggregations/atlas_4regime.txt.
(D) Five-form constraint screen for bm_myc (aM error), 4 regimes, 3 starts.
    Source: runs/20260921_screen_5form/screen.json (cross-check:
    runs/20260921_aggregations/screen_5form.txt).  The atlas's bm_myc cells in
    (C) come from a different run (different restarts), so they differ from (D)
    in Early/Advanced/Severe; each panel uses its own run.
"""
import json
import os
import re
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style as st

st.setup()
R = st.REPO
HY = os.path.join(R, "PINN-hybrid-ude")
sys.path.insert(0, HY)
from hybrid import MechanisticNN, ShapeConstrainedNN, AnchoredRateNN, true_term  # noqa: E402
from config import BASELINE, REGIMES as CREG  # noqa: E402

RUNS = os.path.join(HY, "runs")
R1 = os.path.join(RUNS, "20260727_012448_bm_myc")
CACHE = os.path.join(R, "research-poster-latex-hybrid/data/hybrid_comparison.json")
FIVE = os.path.join(RUNS, "20260921_screen_5form")
FIVE_TXT = os.path.join(RUNS, "20260921_aggregations/screen_5form.txt")
ATL_LOG = os.path.join(RUNS, "20260801_204046_screen_atlas/screen.log")
ATL_JS = os.path.join(RUNS, "20260802_screen_atlas_rest/screen.json")
ATL_AGG = os.path.join(RUNS, "20260921_aggregations/atlas_4regime.txt")
REG = st.REGIMES
MK = {"Normal": "o", "Early Adenoma": "s", "Advanced Adenoma": "^", "Severe APC Loss": "D"}
DT = torch.get_default_dtype()
SC_DARK = "#a34a00"   # second hybrid parameterisation (sc), darker orange


def curve(net, x):
    with torch.no_grad():
        return net(torch.tensor(np.asarray(x), dtype=DT).reshape(-1, 1)).numpy().ravel()


def make_net(param, sd):
    xs = float(sd["x_scale"].reshape(-1)[0])
    n = {"gated": lambda: MechanisticNN(x_scale=xs),
         "sc": lambda: ShapeConstrainedNN(x_scale=xs),
         "sc_bounded": lambda: ShapeConstrainedNN(x_scale=xs, u_max=1.35),
         "lin": lambda: AnchoredRateNN(x_scale=xs),
         "lin_mono": lambda: AnchoredRateNN(x_scale=xs, mono=True)}[param]()
    n.load_state_dict(sd)
    n.eval()
    return n


# ------------------------------------------------------------ (A) round-1
cache = {r["regime"]: r["term"] for r in json.load(open(CACHE))["bm_myc"]}
R1D = {}
for r in REG:
    s = r.replace(" ", "_")
    net = make_net("gated", torch.load(os.path.join(R1, s + "_term_bm_myc.pt"), map_location="cpu"))
    tj = json.load(open(os.path.join(R1, s + "_term.json")))
    rec = json.load(open(os.path.join(R1, s + "_recovered.json")))["recovered"]
    p = {**BASELINE, **CREG[r]}
    g = np.asarray(tj["grid"])
    fl = curve(net, g)
    assert np.allclose(fl, tj["learned"], atol=1e-6) and np.allclose(fl, cache[r]["learned"], atol=1e-6), r
    ft = true_term("bm_myc", g, p)
    d = fl - ft
    R1D[r] = dict(net=net, p=p, lo=tj["r_lo"], hi=tj["r_hi"], aM_hat=rec["aM"], aM=p["aM"],
                  epsM_hat=rec["epsM"], epsM=p["epsM"], dA=p["aM"] - rec["aM"],
                  off_mean=float(d.mean()), off_sd=float(d.std()), nrmse=tj["nrmse"])

# ------------------------------------------------------------ five-form
FORMS = ["gated", "sc", "sc_bounded", "lin", "lin_mono"]
five = json.load(open(FIVE + "/screen.json"))
F5 = {}
for row in five:
    r, prm = row["regime"], row["param"]
    net = make_net(prm, torch.load(os.path.join(FIVE, f"{r.replace(' ', '_')}_bm_myc_{prm}.pt"),
                                   map_location="cpu"))
    p = {**BASELINE, **CREG[r]}
    g = np.linspace(row["x_lo"], row["x_hi"], 400)
    d = curve(net, g) - true_term("bm_myc", g, p)
    F5[(r, prm)] = dict(basal=100 * row["basal_err"], ctrl=100 * row["ctrl_basal_err"],
                        nrmse=100 * row["term_nrmse"], aM_hat=row["recovered"]["aM"],
                        epsM_hat=row["recovered"]["epsM"], dA=p["aM"] - row["recovered"]["aM"],
                        off_mean=float(d.mean()), off_sd=float(d.std()), seed=row["seed"])
# cross-check vs the aggregation table (1 dp)
for line in open(FIVE_TXT):
    m = re.match(r"^\s+(Normal|Early Adenoma|Advanced Adenoma|Severe APC Loss)\s+bm_myc\s+(\w+)\s+fNRMSE=\s*"
                 r"([\d.]+)%.*basal_err=([\d.]+)% \(ctrl ([\d.]+)%\)", line)
    if m:
        v = F5[(m.group(1), m.group(2))]
        assert abs(v["basal"] - float(m.group(4))) < 0.051 and abs(v["ctrl"] - float(m.group(5))) < 0.051, line

# ------------------------------------------------------------ atlas
EDGES = ["ra_h5", "rc_cyp", "bc_cyp", "bm_myc", "b_h13", "m_h13", "h13_b"]
BASAL = {"ra_h5": "a_5", "rc_cyp": "a_C", "bm_myc": "a_M", "bc_cyp": "a_C", "b_h13": "a_{13}",
         "m_h13": "a_{13}", "h13_b": "W"}
pat = re.compile(r"^\s*Normal\s+(\w+)\s+(gated|sc)\s+fNRMSE=\s*([\d.]+)%\s+eq params (\d+)/(\d+) "
                 r"\(ctrl (\d+)/(\d+)\)\s+basal_err=([\d.nan]+)% \(ctrl ([\d.nan]+)%\)")
ATL = {}
for line in open(ATL_LOG):
    m = pat.match(line)
    if m and m.group(1) in EDGES:
        ATL[("Normal", m.group(1), m.group(2))] = dict(basal=float(m.group(8)), ctrl=float(m.group(9)),
                                                      fnrmse=float(m.group(3)), precision="0.1 % (log)")
for row in json.load(open(ATL_JS)):
    if row["term"] in EDGES:
        ATL[(row["regime"], row["term"], row["param"])] = dict(
            basal=100 * row["basal_err"], ctrl=100 * row["ctrl_basal_err"], fnrmse=100 * row["term_nrmse"],
            precision="full")
assert len(ATL) == 7 * 2 * 4
for line in open(ATL_AGG):
    m = re.match(r"^(\w+)\s+(\w+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s*$", line)
    if m and m.group(1) in EDGES:
        e = m.group(1)
        for k, prm, key in ((3, "gated", "basal"), (4, "sc", "basal"), (5, "gated", "ctrl")):
            mean = np.mean([ATL[(rg, e, prm)][key] for rg in REG])
            assert abs(mean - float(m.group(k))) <= 0.15, (e, prm, mean, m.group(k))

FLOOR = 0.05   # percent; the atlas log prints 0.1 % precision, so "0.0 %" means < 0.05 %

# ================================================================ figure
fig = plt.figure(figsize=(st.COL2, 5.0))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.08], hspace=0.42, left=0.075, right=0.985,
                         top=0.955, bottom=0.145)
top = outer[0].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.30)
bot = outer[1].subgridspec(1, 2, width_ratios=[4.0, 1.55], wspace=0.10)
axA = fig.add_subplot(top[0]); axI = fig.add_subplot(top[1])
bl = bot[0].subgridspec(1, 4, wspace=0.07)
axB = [fig.add_subplot(bl[i]) for i in range(4)]
axC = fig.add_subplot(bot[1], sharey=axB[0])
for a in axB[1:]:
    a.sharey(axB[0])

# ---- A left: curves (Early Adenoma)
rA = "Early Adenoma"
d = R1D[rA]
gg = np.linspace(0, d["hi"], 600)
fl, ft = curve(d["net"], gg), true_term("bm_myc", gg, d["p"])
axA.axvspan(d["lo"], d["hi"], color="#f1f1f1", zorder=0, lw=0)
axA.text((d["lo"] + d["hi"]) / 2, 1.47, "data support of $b$", ha="center", va="top", fontsize=6.6,
         color=st.INK2)
axA.plot(gg, ft, color=st.INK, lw=1.4, label="true Hill term $f(b)$")
axA.plot(gg, fl, color=st.ORANGE, lw=1.4, ls=(0, (4, 1.6)), label="learned $\\hat f(b)$ (gated net)")
axA.plot(gg, ft + d["dA"], color=st.ORANGE, lw=0.7, ls=":", alpha=0.9,
         label="$f(b) + (a_M - \\hat a_M)$")
axA.scatter([0], [0], s=14, color=st.INK, zorder=5, clip_on=False)
axA.annotate(f"anchor $f(0)=0$:\nno data below $b$ = {d['lo']:.2f}", xy=(0.012, 0.02), xytext=(0.03, 1.12),
             fontsize=6.4, color=st.INK2, va="bottom",
             arrowprops=dict(arrowstyle="->", lw=0.5, color=st.INK2, shrinkB=1))
xb = 1.45
yb0 = true_term("bm_myc", np.array([xb]), d["p"])[0]
yb1 = curve(d["net"], [xb])[0]
axA.annotate("", xy=(xb, yb1), xytext=(xb, yb0),
             arrowprops=dict(arrowstyle="<->", lw=0.6, color=st.INK2, shrinkA=0, shrinkB=0))
axA.text(1.0, 0.70,
         f"offset {d['off_mean']:.3f}\n$a_M - \\hat a_M$ = {d['dA']:.3f}\n"
         f"($\\hat a_M$ = {d['aM_hat']:.3f} vs {d['aM']:.2f}, "
         f"$-${100 * d['dA'] / d['aM']:.0f} %)", fontsize=6.4, color=st.INK, va="center", ha="left")
axA.set_xlim(0, d["hi"] * 1.01); axA.set_ylim(0, 1.5)
axA.set_xlabel("$\\beta$-catenin $b$"); axA.set_ylabel("MYC activation")
axA.set_title("Early Adenoma, full pipeline (round 1)", fontsize=7.5, loc="left")
axA.legend(loc="lower right", fontsize=6.4, handlelength=2.0, borderaxespad=0.2)
st.panel(axA, "A", x=-0.13, y=1.03)

# ---- A right: identity
lim = (-0.09, 0.20)
axI.plot(lim, lim, color=st.INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
for r in REG:
    for prm in FORMS[:-1]:
        v = F5[(r, prm)]
        axI.errorbar(v["dA"], v["off_mean"], yerr=v["off_sd"], fmt=MK[r], ms=3.6, color=st.REG_COLOR[r],
                     mec="white", mew=0.4, elinewidth=0.6, capsize=0, zorder=3)
    v = R1D[r]
    axI.errorbar(v["dA"], v["off_mean"], yerr=v["off_sd"], fmt=MK[r], ms=4.2, mfc="white",
                 mec=st.REG_COLOR[r], mew=0.9, color=st.REG_COLOR[r], elinewidth=0.6, capsize=0, zorder=4)
axI.set_xlim(*lim); axI.set_ylim(*lim)
axI.set_aspect("equal", adjustable="box")
axI.set_xlabel("basal deficit $a_M - \\hat a_M$")
axI.set_ylabel("mean offset of $\\hat f - f$ on support")
axI.set_title("compensation identity", fontsize=7.5, loc="left")
hI = [Line2D([], [], marker=MK[r], color=st.REG_COLOR[r], lw=0, ms=4, label=st.REG_SHORT[r]) for r in REG]
hI += [Line2D([], [], marker="o", color=st.INK2, lw=0, ms=4, label="screen, exact states (4 forms)"),
       Line2D([], [], marker="o", mfc="white", mec=st.INK2, lw=0, ms=4, label="full pipeline")]
axI.legend(handles=hI, loc="upper left", fontsize=6.2, ncol=1, handletextpad=0.2, borderaxespad=0.1,
           labelspacing=0.25, bbox_to_anchor=(1.0, 1.0))
st.panel(axI, "B", x=-0.42, y=1.03)

# ---- B: atlas
x = np.arange(len(EDGES))
for ax, r in zip(axB, REG):
    g = np.array([ATL[(r, e, "gated")]["basal"] for e in EDGES])
    s = np.array([ATL[(r, e, "sc")]["basal"] for e in EDGES])
    c = np.array([ATL[(r, e, "gated")]["ctrl"] for e in EDGES])
    for i in range(len(EDGES)):
        lo = max(min(g[i], s[i], c[i]), FLOOR); hi = max(g[i], s[i], c[i])
        ax.vlines(x[i], lo, hi, color=LIGHT if False else "#d9d9d9", lw=0.8, zorder=1)
    ax.scatter(x, np.maximum(c, FLOOR), s=17, facecolors="white", edgecolors=st.GREY, marker="o",
               linewidths=0.9, zorder=3)
    ax.scatter(x - 0.16, np.maximum(g, FLOOR), s=15, color=st.ORANGE, marker="o", edgecolors="white",
               linewidths=0.4, zorder=4)
    ax.scatter(x + 0.16, np.maximum(s, FLOOR), s=14, color=SC_DARK, marker="s", edgecolors="white",
               linewidths=0.4, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}" for e in EDGES], rotation=90, fontsize=6.4)
    ax.set_xlim(-0.6, len(EDGES) - 0.4)
    ax.axhline(10, color=st.INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.grid(axis="x", visible=False)
    ax.set_title(st.REG_SHORT[r], fontsize=7.5, color=st.REG_COLOR[r])
    if ax is not axB[0]:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", length=0)
axB[0].set_ylim(0.01, 1000)
axB[0].set_yticks([0.01, 0.1, 1, 10, 100, 1000])
axB[0].set_yticklabels(["0.01", "0.1", "1", "10", "100", "1000"])
axB[0].yaxis.set_minor_locator(plt.NullLocator())
axB[0].set_ylabel("basal parameter error (%)")
st.panel(axB[0], "C", x=-0.55, y=1.06)

# ---- C: five forms
FL = ["gated", "sc", "sc\nbounded", "lin", "lin\nmono", "mech.\ncontrol"]
xs = np.arange(6)
for j, r in enumerate(REG):
    off = (j - 1.5) * 0.15
    ys = [F5[(r, f)]["basal"] for f in FORMS] + [F5[(r, "gated")]["ctrl"]]
    for k, yv in enumerate(ys):
        ax_c = st.GREY if k == 5 else st.REG_COLOR[r]
        axC.scatter(xs[k] + off, yv, s=14, marker=MK[r], color=st.REG_COLOR[r],
                    edgecolors="white", linewidths=0.4, zorder=4)
axC.axvspan(4.5, 5.5, color="#f3f3f3", zorder=0, lw=0)
axC.axhline(10, color=st.INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
axC.set_xticks(xs); axC.set_xticklabels(FL, fontsize=6.4, rotation=90)
axC.set_xlim(-0.5, 5.5)
axC.grid(axis="x", visible=False)
plt.setp(axC.get_yticklabels(), visible=False); axC.tick_params(axis="y", length=0)
axC.set_title("bm_myc, five forms", fontsize=7.5)
hC = [Line2D([], [], marker=MK[r], color=st.REG_COLOR[r], lw=0, ms=3.6, label=st.REG_SHORT[r]) for r in REG]
st.panel(axC, "D", x=-0.12, y=1.06)

hB = [Line2D([], [], marker="o", mfc="white", mec=st.GREY, lw=0, ms=4, label="mechanistic control"),
      Line2D([], [], marker="o", color=st.ORANGE, lw=0, ms=4, label="hybrid, gated"),
      Line2D([], [], marker="s", color=SC_DARK, lw=0, ms=3.6, label="hybrid, sc (monotone + exact anchor)")]
fig.legend(handles=hB + hC, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.005), fontsize=6.4,
           handletextpad=0.2, columnspacing=0.9)

# ================================================================ audit
def strip(dd):
    return {k: v for k, v in dd.items() if k not in ("net", "p")}


audit = {
    "figure": "fig08_learned_term",
    "panels": {
        "A_curves": {"source": os.path.relpath(R1, R), "cache_checked": os.path.relpath(CACHE, R),
                     "regime_shown": rA, "per_regime": {r: strip(v) for r, v in R1D.items()}},
        "A_identity": {"source": os.path.relpath(FIVE, R),
                       "points_screen": {f"{r}/{p}": F5[(r, p)] for r in REG for p in FORMS[:-1]},
                       "points_full_pipeline": {r: strip(v) for r, v in R1D.items()},
                       "excluded_lin_mono": {r: {"dA": F5[(r, "lin_mono")]["dA"],
                                                 "off_mean": F5[(r, "lin_mono")]["off_mean"],
                                                 "epsM_hat": F5[(r, "lin_mono")]["epsM_hat"]} for r in REG},
                       "note": "lin_mono excluded: eps_M is inflated (1.2-18.3 vs 0.6), so the deficit is not "
                               "carried by a_M alone."},
        "B_atlas": {"sources": {"Normal": os.path.relpath(ATL_LOG, R), "others": os.path.relpath(ATL_JS, R),
                                "cross_check": os.path.relpath(ATL_AGG, R)},
                    "cells": {f"{r}/{e}/{p}": ATL[(r, e, p)] for r in REG for e in EDGES for p in ("gated", "sc")},
                    "floor_pct": FLOOR,
                    "cells_under_10pct": {p: int(sum(ATL[(r, e, p)]["basal"] < 10 for r in REG for e in EDGES))
                                          for p in ("gated", "sc")},
                    "control_cells_under_10pct": int(sum(ATL[(r, e, "gated")]["ctrl"] < 10
                                                         for r in REG for e in EDGES))},
        "C_five_form": {"source": os.path.relpath(FIVE + "/screen.json", R),
                        "cross_check": os.path.relpath(FIVE_TXT, R),
                        "basal_err_pct": {f"{r}/{p}": F5[(r, p)]["basal"] for r in REG for p in FORMS},
                        "control_basal_err_pct": {r: F5[(r, "gated")]["ctrl"] for r in REG}},
        "discrepancy_atlas_vs_five_form_bm_myc": {
            r: {"atlas_gated": ATL[(r, "bm_myc", "gated")]["basal"], "five_gated": F5[(r, "gated")]["basal"],
                "atlas_sc": ATL[(r, "bm_myc", "sc")]["basal"], "five_sc": F5[(r, "sc")]["basal"],
                "atlas_ctrl": ATL[(r, "bm_myc", "gated")]["ctrl"], "five_ctrl": F5[(r, "gated")]["ctrl"]}
            for r in REG},
    },
}
st.save(fig, "fig08_learned_term", audit)
print(json.dumps(audit["panels"]["discrepancy_atlas_vs_five_form_bm_myc"], indent=0))
print({r: (round(v["dA"], 4), round(v["off_mean"], 4)) for r, v in R1D.items()})
print(audit["panels"]["B_atlas"]["cells_under_10pct"], audit["panels"]["B_atlas"]["control_cells_under_10pct"])
