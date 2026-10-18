"""Fig 9 -- the Fisher information of a learned term (training-free).

Source: experiments/15-hybrid-ude/runs/20261017_hybrid_fim/ (hybrid_fim.py,
hybrid_fim_analysis.py). S = CRLB sd(log basal) / sigma_r for the screen's
residual model (exact states, trapezoidal residual of the hosting equation),
with the learned term represented by the screen's own 2x5 `gated` network
fitted to the true Hill term and profiled out (SVD tolerance 1e-8).
(A) design factor S(design)/S(base10) per production edge and regime;
(B) dose-response: information lost to the network, S_net/S_mech, against the
    achieved anchor ratio, three dose arms x two regimes;
(C) S_net against the measured hybrid basal error of every screen fit
    (atlas, A/B, information-matched, dose, protocol; de-duplicated).
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import style as st

st.setup()
SRC = os.path.join(st.REPO, "experiments/15-hybrid-ude/runs/20261017_hybrid_fim")
res = json.load(open(os.path.join(SRC, "results.json")))["results"]
cor = json.load(open(os.path.join(SRC, "correlations.json")))
idx = {(r["regime"], r["design"], r["edge"]): r for r in res}
EDGES = ["ra_h5", "rc_cyp", "bm_myc", "bc_cyp", "b_h13", "m_h13", "h13_b"]
EDGE_LAB = {"ra_h5": "RA→HOXA5\n($a_5$)", "rc_cyp": "RA→CYP26\n($a_C$)",
            "bm_myc": "β-cat→MYC\n($a_M$)", "bc_cyp": "β-cat→CYP26\n($a_C$)",
            "b_h13": "β-cat→HOXA13\n($a_{13}$)", "m_h13": "MYC→HOXA13\n($a_{13}$)",
            "h13_b": "HOXA13→β-cat\n($W$)"}
audit = {"source": SRC}

fig = plt.figure(figsize=(st.COL2, 5.2))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.02], width_ratios=[1.0, 1.0],
              hspace=0.62, wspace=0.32)

# ---- A: design factors ---------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
designs = [("info", "+2 information-matched", st.ORANGE, "o", "none"),
           ("dep", "+2 depletion (WNT $k_W$=0.02, RA-free)", st.BLUE, "o", st.BLUE),
           ("depb", "+2 depletion (β-cat double KO, RA-free)", "#08306b", "s", "#08306b")]
off = {"info": -0.24, "dep": 0.0, "depb": 0.24}
audit["A"] = {}
for j, e in enumerate(EDGES):
    for d, lab, col, mk, fc in designs:
        vals = []
        for k_r, r in enumerate(st.REGIMES):
            b = idx[(r, "base10", e)]["S"]["gated"]
            v = idx[(r, d, e)]["S"]["gated"] / b
            vals.append(v)
            ax.plot(j + off[d] + (k_r - 1.5) * 0.04, v, marker=mk, ms=3.6, color=col,
                    mfc=fc, mew=0.8, ls="none", alpha=0.9)
        # mechanistic control factor (grey tick) for the same design
        mv = [idx[(r, d, e)]["S"]["mech"] / idx[(r, "base10", e)]["S"]["mech"]
              for r in st.REGIMES]
        ax.plot([j + off[d] - 0.09, j + off[d] + 0.09], [np.exp(np.mean(np.log(mv)))] * 2,
                color=st.GREY, lw=1.6, solid_capstyle="butt")
        audit["A"][f"{e}/{d}"] = dict(net_factor=vals, mech_factor=mv)
ax.axhline(1, color=st.INK2, lw=0.7)
ax.set_yscale("log")
ax.set_ylim(3e-9, 8)
ax.set_xlim(-0.55, len(EDGES) - 0.45)
ax.set_xticks(range(len(EDGES)))
ax.set_xticklabels([EDGE_LAB[e] for e in EDGES], fontsize=6.6)
ax.set_ylabel(r"CRLB of basal rate," + "\n" + r"design / base design")
ax.grid(axis="x", lw=0)
for j in range(len(EDGES) - 1):
    ax.axvline(j + 0.5, color=st.GRID, lw=0.6)
h = [Line2D([], [], marker=mk, ls="none", color=col, mfc=fc, mew=0.8, ms=4, label=lab)
     for _, lab, col, mk, fc in designs]
h.append(Line2D([], [], color=st.GREY, lw=1.6, label="mechanistic control (geo. mean)"))
ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.53, 1.01), ncol=4,
          fontsize=6.3, handlelength=1.2, columnspacing=1.0)
ax.annotate("deletes $W$ / removes the\nnet with the anchor", xy=(6.24, 0.9),
            xytext=(5.0, 1.5e-4), fontsize=6.1, color=st.RED, ha="center",
            arrowprops=dict(arrowstyle="-", color=st.RED, lw=0.6))
ax.annotate("fractional WNT\nknockdown pins $W$", xy=(6.0, 1e-5),
            xytext=(6.0, 2e-8), fontsize=6.1, color=st.BLUE, ha="center",
            arrowprops=dict(arrowstyle="-", color=st.BLUE, lw=0.6))
st.panel(ax, "A", x=-0.085, y=1.13)

# ---- B: dose response ----------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ARM = {"ra": ("ra_h5", st.BLUE, "o", "RA restriction ($a_5$)"),
       "wnt": ("bm_myc", st.ORANGE, "^", "WNT KD ($a_M$)"),
       "bcat": ("bm_myc", "#08306b", "s", "β-cat double KD ($a_M$)")}
ks = ["1", "0.3", "0.1", "0.03", "0.01", "0"]
audit["B"] = {}
for arm, (edge, col, mk, lab) in ARM.items():
    for r, ls, mfc in [("Normal", "-", col), ("Severe APC Loss", "--", "white")]:
        xs, ys = [], []
        for k in ks:
            row = idx.get((r, f"dose_{arm}_k{k}", edge))
            if row is None:
                continue
            xs.append(max(row["rho"], 2e-5))
            ys.append(row["S"]["gated"] / row["S"]["mech"])
        ax.plot(xs, ys, ls=ls, marker=mk, color=col, mfc=mfc, ms=3.4, lw=1.0, mew=0.8)
        audit["B"][f"{arm}/{r}"] = dict(rho=xs, loss=ys)
ax.set_xscale("log")
ax.set_yscale("log")
ax.axhline(1, color=st.GREY, lw=0.7, ls=":")
ax.set_xlabel(r"achieved anchor ratio $\rho$ (0 drawn at $2\times10^{-5}$)")
ax.set_ylabel(r"information lost to the net, $\Gamma_{\mathrm{net}}/\Gamma_{\mathrm{mech}}$")
h = [Line2D([], [], color=c, marker=m, ms=3.4, lw=1.0, label=l) for _, c, m, l in ARM.values()]
h += [Line2D([], [], color=st.INK2, ls="-", lw=1.0, label="Normal"),
      Line2D([], [], color=st.INK2, ls="--", lw=1.0, label="Severe")]
ax.legend(handles=h, loc="upper left", fontsize=6.0, ncol=1, handlelength=1.6)
dz = json.load(open(os.path.join(SRC, "analysis.json")))["dose"]["gated"]
audit["B_stats"] = dz
st.panel(ax, "B", x=-0.2, y=1.03)

# ---- C: predicted vs measured --------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
dedup = {}
for p in cor["pairs"]:
    k = (p["edge"], p["regime"], p["design"], p["form"])
    if k not in dedup or p["source"] == "dose5":
        dedup[k] = p
pts = list(dedup.values())
SRCSTY = {"atlas": (st.LIGHT, "o", "edge atlas (base design)"),
          "AB": (st.BLUE, "D", "depletion A/B"),
          "infoctl": (st.ORANGE, "v", "information-matched"),
          "dose": ("#08306b", "s", "dose-response"),
          "dose5": ("#08306b", "s", None),
          "protocol": (st.GREEN, "^", "prospective test")}
audit["C"] = []
for p in pts:
    c, m, _ = SRCSTY[p["source"]]
    e = max(100 * p["hybrid_basal_err"], 5e-4)
    ax.plot(p["S_form"], e, marker=m, color=c, ms=3.2, ls="none", alpha=0.85,
            mec="white", mew=0.3)
    audit["C"].append(dict(edge=p["edge"], regime=p["regime"], design=p["design"],
                           form=p["form"], source=p["source"], S=p["S_form"], err_pct=e))
s_net = spearmanr([p["S_form"] for p in pts], [p["hybrid_basal_err"] for p in pts])
s_rho = spearmanr([p["rho_design"] for p in pts], [p["hybrid_basal_err"] for p in pts])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"CRLB of basal rate, $\Gamma_{\mathrm{net}}$ (per unit residual noise)")
ax.set_ylabel("measured hybrid basal error (%)")
ax.text(0.03, 0.97, f"Spearman {s_net.statistic:+.2f} (n = {len(pts)})\n"
        f"anchor ratio alone {s_rho.statistic:+.2f}",
        transform=ax.transAxes, va="top", fontsize=6.4)
h = [Line2D([], [], marker=m, color=c, ls="none", ms=3.6, label=l)
     for c, m, l in SRCSTY.values() if l]
ax.legend(handles=h, loc="lower right", fontsize=5.9, handletextpad=0.3)
audit["C_stats"] = dict(spearman_S=float(s_net.statistic), p=float(s_net.pvalue),
                        n=len(pts), spearman_rho=float(s_rho.statistic))
st.panel(ax, "C", x=-0.2, y=1.03)

st.save(fig, "fig09_hybrid_fisher", audit)
print(audit["C_stats"])
