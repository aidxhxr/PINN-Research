"""Fig 10 -- the anchor dose-response and the prospective protocol test.

(A, B) Dose-response with the condition count FIXED at 11 (10 base + one graded
    depletion of residual fraction k; k = 1 duplicates a base condition).
    Hybrid basal-parameter error (filled, joined) and the mechanistic control on
    the same 11 conditions (small hollow) against (A) the achieved anchor ratio
    rho = x_lo/x_hi and (B) the dose k.  Arms: ra (raKO -> r, edge ra_h5, a5),
    wnt (wntKO -> b, bm_myc, aM), bcat (wntKO + k13 knockdown -> b, bm_myc, aM).
    Normal: runs/20260802_dose_response (2 starts).
    Severe: runs/20260802_dose_starts5 (5 starts; the 2-start values from
    runs/20260802_dose_response are drawn faint for comparison).
    Error bars at k = 0 (wnt, bcat): mean and range over three disjoint seed
    bases x 5 starts, runs/20260921_dose_seeds/seed{100,300,500}.
(C) Prospective protocol test, runs/20260802_protocol_test/protocol_{m_h13,h13_b}.json:
    m_h13 (basal a13): baseline / bcatKO (larger perturbation, misses the
    anchor) / mycKO (prescribed by anchor_reach.py); h13_b (basal W):
    baseline / mycKO (misses) / hox13KO (reaches the anchor but deletes W, the
    flagged failure).
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style as st

st.setup()
R = st.REPO
RUNS = os.path.join(R, "experiments/15-hybrid-ude/runs")
D2 = os.path.join(RUNS, "20260802_dose_response")
D5 = os.path.join(RUNS, "20260802_dose_starts5")
DS = os.path.join(RUNS, "20260921_dose_seeds")
DP = os.path.join(RUNS, "20260802_protocol_test")
FILES = {"ra": "dose_ra_ra_h5.json", "wnt": "dose_wnt_bm_myc.json", "bcat": "dose_bcat_bm_myc.json"}
NAVY = "#08306b"
ARMS = [("ra", "ra: raKO$\\to r$ (ra_h5, $a_5$)", st.BLUE, "o"),
        ("bcat", "bcat: wntKO + $k_{13}\\to b$ (bm_myc, $a_M$)", NAVY, "s"),
        ("wnt", "wnt: wntKO$\\to b$ (bm_myc, $a_M$)", st.ORANGE, "^")]
REGS = ["Normal", "Severe APC Loss"]
R0, K0, YF = 1.2e-4, 3e-3, 3e-4      # where ratio 0 / dose 0 are drawn; y floor (%)


def load(d, arm, regime):
    recs = [r for r in json.load(open(os.path.join(d, FILES[arm]))) if r["regime"] == regime]
    return sorted(recs, key=lambda r: r["dose"])


main = {("Normal", a): load(D2, a, "Normal") for a in FILES}
main.update({("Severe APC Loss", a): load(D5, a, "Severe APC Loss") for a in FILES})
faint = {a: load(D2, a, "Severe APC Loss") for a in FILES}
seeds = {}
for a in ("wnt", "bcat"):
    for r in REGS:
        v = []
        for s in (100, 300, 500):
            rec = [x for x in json.load(open(os.path.join(DS, f"seed{s}", FILES[a]))) if x["regime"] == r]
            assert len(rec) == 1 and rec[0]["dose"] == 0.0 and rec[0]["starts"] == 5
            v.append(100 * rec[0]["hybrid_basal_err"])
        seeds[(a, r)] = dict(values=v, mean=float(np.mean(v)), lo=min(v), hi=max(v),
                             ratio=rec[0]["anchor_ratio"])
# sanity: Severe 5-start k=0 run uses seed base 100
for a in ("wnt", "bcat"):
    k0 = [r for r in main[("Severe APC Loss", a)] if r["dose"] == 0][0]
    assert abs(100 * k0["hybrid_basal_err"] - seeds[(a, "Severe APC Loss")]["values"][0]) < 1e-9

fig = plt.figure(figsize=(st.COL2, 4.45))
outer = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.30, left=0.075, right=0.99, top=0.95,
                         bottom=0.215)
left = outer[0].subgridspec(2, 2, hspace=0.16, wspace=0.10)
right = outer[1].subgridspec(2, 1, hspace=0.62)
ax = [[fig.add_subplot(left[i, j]) for j in range(2)] for i in range(2)]
axP = [fig.add_subplot(right[i]) for i in range(2)]
for i in range(2):
    for j in range(2):
        if (i, j) != (0, 0):
            ax[i][j].sharey(ax[0][0])


def xval(r, key):
    v = r["anchor_ratio"] if key == "ratio" else r["dose"]
    return v if v > 0 else (R0 if key == "ratio" else K0)


audit_cells = []
for i, reg in enumerate(REGS):
    for j, key in enumerate(["ratio", "dose"]):
        a_ = ax[i][j]
        for arm, lab, c, m in ARMS:
            recs = main[(reg, arm)]
            xs = np.array([xval(r, key) for r in recs])
            yh = np.array([max(100 * r["hybrid_basal_err"], YF) for r in recs])
            yc = np.array([max(100 * r["ctrl_basal_err"], YF) for r in recs])
            o = np.argsort(xs)
            a_.plot(xs[o], yh[o], color=c, lw=1.0, zorder=3)
            a_.scatter(xs, yh, s=17, marker=m, color=c, edgecolors="white", linewidths=0.4, zorder=4)
            a_.scatter(xs, yc, s=8, marker=m, facecolors="white", edgecolors=c, linewidths=0.6, zorder=3,
                       alpha=0.8)
            if reg == "Severe APC Loss":
                fx = np.array([xval(r, key) for r in faint[arm]])
                fy = np.array([max(100 * r["hybrid_basal_err"], YF) for r in faint[arm]])
                a_.scatter(fx, fy, s=9, marker=m, color=c, alpha=0.3, linewidths=0, zorder=2)
            if arm in ("wnt", "bcat"):
                sd = seeds[(arm, reg)]
                x0 = (sd["ratio"] if sd["ratio"] > 0 else R0) if key == "ratio" else K0
                x0 *= 1.45
                a_.errorbar(x0, sd["mean"], yerr=[[sd["mean"] - sd["lo"]], [sd["hi"] - sd["mean"]]],
                            fmt=m, ms=3.6, mfc="white", mec=c, mew=0.8, color=c, elinewidth=0.8, capsize=1.5,
                            zorder=5)
            if j == 0:
                for r in recs:
                    audit_cells.append(dict(regime=reg, arm=arm, dose=r["dose"], anchor_ratio=r["anchor_ratio"],
                                            hybrid_basal_err_pct=100 * r["hybrid_basal_err"],
                                            ctrl_basal_err_pct=100 * r["ctrl_basal_err"],
                                            term_nrmse_pct=100 * r["term_nrmse"], starts=r["starts"],
                                            source=os.path.relpath(D2 if reg == "Normal" else D5, R)))
        a_.set_xscale("log"); a_.set_yscale("log")
        a_.axhline(10, color=st.INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
        a_.xaxis.set_minor_locator(plt.NullLocator())
        if key == "ratio":
            a_.set_xlim(R0 / 1.7, 0.3)
            a_.set_xticks([R0, 1e-3, 1e-2, 1e-1]); a_.set_xticklabels(["0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
            brk = R0 * 2.4
        else:
            a_.set_xlim(K0 / 1.7, 1.7)
            a_.set_xticks([K0, 0.01, 0.1, 1]); a_.set_xticklabels(["0", "0.01", "0.1", "1"])
            brk = K0 * 2.0
        a_.text(brk, YF / 1.6, "//", ha="center", va="center", fontsize=6.5, color=st.INK2, clip_on=False,
                zorder=7, bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
        if i == 0:
            plt.setp(a_.get_xticklabels(), visible=False)
        else:
            a_.set_xlabel("achieved anchor ratio $\\rho = x_{lo}/x_{hi}$" if key == "ratio"
                          else "knockdown dose $k$ (residual fraction)")
        if j == 1:
            plt.setp(a_.get_yticklabels(), visible=False)
        else:
            a_.set_ylabel("basal parameter error (%)")
        a_.text(0.03, 0.96, st.REG_SHORT[reg] + (" (5 starts)" if i == 1 else " (2 starts)"),
                transform=a_.transAxes, fontsize=7, color=st.REG_COLOR[reg], fontweight="bold",
                va="top", ha="left")
ax[0][0].set_ylim(YF / 1.6, 300)
ax[0][0].set_yticks([1e-3, 1e-2, 0.1, 1, 10, 100])
ax[0][0].set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
ax[0][0].set_title("against the anchor ratio", fontsize=7.5, loc="left")
ax[0][1].set_title("against the dose", fontsize=7.5, loc="left")
st.panel(ax[0][0], "A", x=-0.25, y=1.05)
st.panel(ax[0][1], "B", x=-0.08, y=1.05)

# ---- C protocol
PANELS = [("m_h13", "$a_{13}$", ["none", "bcatKO", "mycKO"],
           {"none": "base", "bcatKO": "bcatKO", "mycKO": "mycKO"}),
          ("h13_b", "$W$", ["none", "mycKO", "hox13KO"],
           {"none": "base", "mycKO": "mycKO", "hox13KO": "h13KO"})]
STYLE = {("m_h13", "none"): (st.ORANGE, True), ("m_h13", "bcatKO"): (st.ORANGE, False),
         ("m_h13", "mycKO"): (st.BLUE, True), ("h13_b", "none"): (st.ORANGE, True),
         ("h13_b", "mycKO"): (st.ORANGE, False), ("h13_b", "hox13KO"): (st.RED, True)}
prot_audit = []
for a_, (edge, bp, protos, plab) in zip(axP, PANELS):
    recs = json.load(open(os.path.join(DP, f"protocol_{edge}.json")))
    ticks, tl = [], []
    for gi, reg in enumerate(REGS):
        for pi, pr in enumerate(protos):
            r = [x for x in recs if x["regime"] == reg and x["protocol"] == pr][0]
            x = gi * 4.5 + pi * 1.4
            c, filled = STYLE[(edge, pr)]
            hy, ct = 100 * r["hybrid_basal_err"], 100 * r["ctrl_basal_err"]
            a_.vlines(x, min(hy, ct), max(hy, ct), color="#d9d9d9", lw=0.8, zorder=1)
            a_.scatter(x, ct, s=10, marker="o", facecolors="white", edgecolors=st.GREY, lw=0.7, zorder=3)
            a_.scatter(x, hy, s=24, marker="o" if reg == "Normal" else "D", color=(c if filled else "white"),
                       edgecolors=c, lw=0.9, zorder=4)
            a_.text(x, hy * 1.9, f"{hy:.1f}" if hy >= 0.95 else f"{hy:.2f}", fontsize=5.8, va="bottom",
                    ha="center", color=st.INK)
            rho = r["anchor_ratio"]
            ticks.append(x); tl.append(f"{plab[pr]}\n{'0' if rho < 1e-9 else f'{rho:.3f}'}")
            prot_audit.append(dict(edge=edge, regime=reg, protocol=pr, anchor_ratio=rho,
                                   hybrid_basal_err_pct=hy, ctrl_basal_err_pct=ct,
                                   term_nrmse_pct=100 * r["term_nrmse"], n_conditions=r["n_conditions"],
                                   hybrid_eq_under10=f'{r["hybrid_eq_under10"]}/{r["hybrid_n_eq_params"]}'))
        a_.text(gi * 4.5 + 1.4, 1.0, st.REG_SHORT[reg], transform=a_.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6.8, color=st.REG_COLOR[reg], fontweight="bold")
    a_.set_yscale("log"); a_.set_ylim(3e-3, 900)
    a_.set_yticks([0.01, 0.1, 1, 10, 100]); a_.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    a_.yaxis.set_minor_locator(plt.NullLocator())
    a_.set_xticks(ticks); a_.set_xticklabels(tl, fontsize=5.9, linespacing=0.95)
    a_.tick_params(axis="x", length=0)
    a_.set_xlim(-0.7, 4.5 + 3.5)
    a_.axvline(3.6, color=st.GRID, lw=0.6)
    a_.axhline(10, color=st.INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    a_.grid(axis="x", visible=False)
    a_.set_ylabel(f"error of {bp} (%)")
    a_.set_title(f"{edge} (basal {bp})", fontsize=7.5, loc="left", pad=11)
for a_ in axP:
    a_.text(-0.02, -0.155, "$\\rho$", transform=a_.transAxes, fontsize=6.2, color=st.INK2, ha="right",
            va="center")
st.panel(axP[0], "C", x=-0.30, y=1.08)

h1 = [Line2D([], [], color=c, marker=m, lw=1.0, ms=3.8, mec="white", label=l) for _, l, c, m in ARMS]
h2 = [Line2D([], [], color=st.INK2, marker="o", mfc="white", lw=0, ms=3, label="mech. control (hollow, small)"),
      Line2D([], [], color=st.INK2, marker="o", alpha=0.3, lw=0, ms=3, label="Severe, 2 starts"),
      Line2D([], [], color=st.INK2, marker="o", mfc="white", lw=0, ms=3.6, label="3 seed bases, mean + range")]
h3 = [Line2D([], [], color=st.ORANGE, marker="o", lw=0, ms=4, label="base design (10)"),
      Line2D([], [], color=st.ORANGE, marker="o", mfc="white", lw=0, ms=4, label="+1 KO, misses anchor"),
      Line2D([], [], color=st.BLUE, marker="o", lw=0, ms=4, label="+1 prescribed KO"),
      Line2D([], [], color=st.RED, marker="o", lw=0, ms=4, label="reaches anchor, deletes $W$")]
l1 = fig.legend(handles=h1 + h2, loc="lower left", ncol=2, bbox_to_anchor=(0.03, -0.005), fontsize=6.3,
                handletextpad=0.3, columnspacing=1.0, labelspacing=0.3)
fig.add_artist(l1)
fig.legend(handles=h3, loc="lower left", ncol=2, bbox_to_anchor=(0.66, -0.005), fontsize=6.3,
           handletextpad=0.2, columnspacing=0.6, labelspacing=0.3)


def k0(reg, arm, key="hybrid_basal_err"):
    return 100 * [r for r in main[(reg, arm)] if r["dose"] == 0][0][key]


audit = {"figure": "fig11_dose_protocol", "units": "percent",
         "sources": {"Normal": os.path.relpath(D2, R), "Severe_main": os.path.relpath(D5, R),
                     "Severe_faint_2start": os.path.relpath(D2, R), "seeds": os.path.relpath(DS, R),
                     "protocol": os.path.relpath(DP, R)},
         "dose_cells": audit_cells,
         "severe_2start_cells": {f"{a}/k={r['dose']}": 100 * r["hybrid_basal_err"] for a in FILES for r in faint[a]},
         "seed_replicates_k0": {f"{a}/{r}": v for (a, r), v in seeds.items()},
         "protocol_cells": prot_audit,
         "k0_hybrid_basal_err_pct": {f"{reg}/{a}": k0(reg, a) for reg in REGS for a in FILES},
         "floors": {"ratio_0_drawn_at": R0, "dose_0_drawn_at": K0, "y_floor_pct": YF,
                    "floored_points": "none below the floor (min 0.0009 % = bcat Severe k=0)"},
         "notes": "Severe 5-start k=0 equals seed base 100 of the seed study (same run settings). Normal "
                  "wnt k=0: 2.17 % (2 starts) vs 1.33-1.52 % (5 starts, three seed bases)."}
st.save(fig, "fig11_dose_protocol", audit)
print(audit["k0_hybrid_basal_err_pct"])
print({k: (round(v["mean"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in audit["seed_replicates_k0"].items()})
