"""Fig 3 -- the information budget of the ten-condition design.

Source: experiments/17-sensitivity-pinn/runs/20261017_paper/fim_seed42/ (fim.py): Fisher
information of the exact benchmark design (150 obs/condition, 7 states,
sigma = 0.002, 10 conditions) at the true parameters, log-parameters.
(A) FIM eigenvalue spectra (36 x 36), (B) the null direction, (C) Cramer-Rao
relative sd of every parameter (35 x 35, deltaP1 known), (D) the CRLB-expected
number of parameters an efficient estimator recovers within 10%.
"""
import json
import os

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as st

st.setup()
SRC = os.path.join(st.REPO, "experiments/17-sensitivity-pinn/runs/20261017_paper/fim_seed42")
import sys
sys.path.insert(0, os.path.join(st.REPO, "src"))
from wnt_pinn.model.parameters import UNKNOWN  # noqa: E402

summ = json.load(open(os.path.join(SRC, "fim_summary.json")))
names35 = [k for k in UNKNOWN if k != "deltaP1"]
regular = [k for k in names35 if k != "thetaP"]
LO, HI = np.log(0.9), np.log(1.1)
audit = {"source": SRC}

fig = plt.figure(figsize=(st.COL2, 6.2))
gs = GridSpec(3, 2, figure=fig, width_ratios=[1.0, 1.05], height_ratios=[1, 1, 1],
              wspace=0.42, hspace=0.62)

# ---- A: spectra --------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
audit["A_eigs"] = {}
for r in st.REGIMES:
    F = np.load(os.path.join(SRC, r.replace(" ", "_") + "_F.npy"))
    ev = np.sort(np.linalg.eigvalsh(F))[::-1]
    ev = np.clip(ev, 1e-12, None)
    ax.semilogy(np.arange(1, 37), ev / ev[0], "-o", color=st.REG_COLOR[r], ms=2.2,
                lw=1.0, label=st.REG_SHORT[r])
    audit["A_eigs"][r] = (ev / ev[0]).tolist()
ax.axhspan(1e-19, 1e-13, color=st.RED, alpha=0.07, lw=0)
ax.text(12, 1e-17, "structural null direction", color=st.RED, fontsize=6.5)
ax.set_xlabel("eigenvalue index")
ax.set_ylabel(r"$\lambda_i/\lambda_{\max}$ of $\mathcal{I}(\log\theta)$")
ax.set_xlim(0, 37)
ax.set_ylim(1e-19, 3)
ax.set_yticks([1, 1e-4, 1e-8, 1e-12, 1e-16])
st.regime_legend(ax, loc="lower center", ncol=4, fontsize=6.3, bbox_to_anchor=(0.55, 1.0), handlelength=1.0, columnspacing=0.8)
st.panel(ax, "A")

# ---- B: null direction ---------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
idx_t, idx_d = UNKNOWN.index("thetaP"), UNKNOWN.index("deltaP1")
w = 0.36
audit["B_null"] = {}
xs = np.arange(4)
for j, r in enumerate(st.REGIMES):
    F = np.load(os.path.join(SRC, r.replace(" ", "_") + "_F.npy"))
    evals, evecs = np.linalg.eigh(F)
    v = evecs[:, 0]
    v = v * np.sign(v[idx_d] or 1)
    other = np.sqrt(max(0.0, 1 - v[idx_t] ** 2 - v[idx_d] ** 2))
    ax.bar(j - w / 2, abs(v[idx_t]), w, color=st.REG_COLOR[r], alpha=0.45, lw=0)
    ax.bar(j + w / 2, abs(v[idx_d]), w, color=st.REG_COLOR[r], lw=0)
    th = {"Normal": 1.0, "Early Adenoma": 0.75, "Advanced Adenoma": 0.5,
          "Severe APC Loss": 0.25}[r]
    pred = np.array([1 - th, th]) / np.hypot(1 - th, th)   # tangent of deltaP1(1-thetaP)=const
    ax.plot([j - w / 2, j + w / 2], pred, ls="none", marker="x", color=st.INK, ms=4, mew=0.9)
    audit["B_null"][r] = dict(thetaP=float(v[idx_t]), deltaP1=float(v[idx_d]),
                              rest_norm=float(other), eig=float(evals[0] / evals[-1]))
ax.set_xticks(xs)
ax.set_xticklabels([st.REG_SHORT[r] for r in st.REGIMES])
ax.set_ylabel("|loading| on null eigenvector")
ax.set_ylim(0, 1.05)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color=st.GREY, alpha=0.45, lw=0, label=r"$\log\theta_P$"),
                   Patch(color=st.GREY, lw=0, label=r"$\log\delta_{P1}$"),
                   Line2D([], [], ls="none", marker="x", color=st.INK, label=r"level set of $\delta_{P1}(1-\theta_P)$")],
          loc="upper center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.6, 1.2), handlelength=1.0, columnspacing=0.9)
st.panel(ax, "B")

# ---- D: expected recoverable counts -------------------------------------------
ax = fig.add_subplot(gs[2, 0])
audit["D"] = {}
for j, r in enumerate(st.REGIMES):
    sd = np.array([summ[r]["sd35_none"][k] for k in regular])
    p = norm.cdf(HI / sd) - norm.cdf(LO / sd)
    n5 = int((sd < 0.05).sum())
    ax.bar(j, p.sum(), 0.62, color=st.REG_COLOR[r], lw=0)
    ax.plot(j, n5, marker="_", color=st.INK, ms=14, mew=1.4)
    ax.text(j, p.sum() - 1.6, f"{p.sum():.1f}", ha="center", fontsize=6.8, color="white", fontweight="bold")
    audit["D"][r] = dict(expected_N10=float(p.sum()), n_sd_lt_5pct=n5)
ax.set_xticks(range(4))
ax.set_xticklabels([st.REG_SHORT[r] for r in st.REGIMES])
ax.set_ylim(20, 39.5)
ax.set_ylabel("parameters (of 34)")
ax.axhline(34, color=st.GREY, lw=0.6, ls="--")
ax.legend(handles=[Patch(color=st.GREY, lw=0, label="CRLB-expected count within 10%"),
                   Line2D([], [], ls="none", marker="_", ms=10, mew=1.4, color=st.INK, label=r"CRLB sd $<$ 5%")],
          loc="upper center", ncol=2, fontsize=6.2, bbox_to_anchor=(0.5, 1.02), handlelength=1.2)
st.panel(ax, "D")

# ---- C: per-parameter CRLB sd ---------------------------------------------------
ax = fig.add_subplot(gs[:, 1])
order = sorted(names35, key=lambda k: summ["Severe APC Loss"]["sd35_none"][k])
y = np.arange(len(order))
ax.axvspan(1e-4, 0.05, color=st.BLUE, alpha=0.06, lw=0)
ax.axvspan(0.05, 0.5, color=st.ORANGE, alpha=0.08, lw=0)
ax.axvspan(0.5, 3, color=st.RED, alpha=0.08, lw=0)
audit["C_sd"] = {}
for k_i, r in enumerate(st.REGIMES):
    sd = np.array([summ[r]["sd35_none"][k] for k in order])
    ax.scatter(sd, y + (k_i - 1.5) * 0.12, s=9, color=st.REG_COLOR[r], zorder=3,
               edgecolor="white", linewidth=0.3, label=st.REG_SHORT[r])
    audit["C_sd"][r] = dict(zip(order, sd.tolist()))
ax.set_xscale("log")
ax.set_xlim(1e-4, 3)
ax.set_yticks(y)
ax.set_yticklabels([st.param_tex(k) for k in order], fontsize=6.6)
ax.set_ylim(-0.8, len(order) - 0.2)
ax.set_xlabel(r"Cramér–Rao relative sd ($\delta_{P1}$ known)")
ax.grid(axis="y", lw=0.3)
for x, t, c in [(2.2e-3, "identifiable", st.BLUE), (0.16, "weak", st.ORANGE),
                (1.25, "non-\nident.", st.RED)]:
    ax.text(x, len(order) + 0.25, t, ha="center", va="bottom", fontsize=6.5,
            color=c, fontweight="bold")
st.panel(ax, "C", x=-0.24, y=1.015)

st.save(fig, "fig03_information", audit)
print(json.dumps(audit["D"], indent=1))
print(audit["B_null"])
