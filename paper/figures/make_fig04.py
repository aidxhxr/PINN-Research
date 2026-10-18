"""Fig 4 -- the Cramer-Rao bound is attained.

Sources: experiments/17-sensitivity-pinn/runs/20261017_paper/mc_tight/ (unit-weight MLE, rtol 1e-11, deltaP1
known, started at the truth, data seeds 101-124) and fim_seed42/ (the bound).
(A) empirical sd of each log-parameter across seeds vs its CRLB sd;
(B) standardised errors z = (x_hat - x_true)/s_CRLB pooled per regime vs N(0,1);
(C) coverage of the +-1.96 s_CRLB interval per parameter, per regime;
(D) 2*(cost(truth) - cost(estimate)) vs chi^2_35: the fits reach the optimum.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as st

st.setup()
SENS = os.path.join(st.REPO, "experiments/17-sensitivity-pinn")
sys.path.insert(0, os.path.join(st.REPO, "src"))
sys.path.insert(0, SENS)
import bench  # noqa: E402
from wnt_pinn.model.parameters import UNKNOWN  # noqa: E402

RUN = os.path.join(SENS, "runs/20261017_paper")
MCDIR = "mc_tight"
fim = json.load(open(os.path.join(RUN, "fim_seed42/fim_summary.json")))
names = [k for k in UNKNOWN if k != "deltaP1"]
audit = {"sources": {"mc": RUN + "/mc_tight", "fim": RUN + "/fim_seed42"}}

Z, EST = {}, {}
for r in st.REGIMES:
    xt = bench.to_x(bench.true_params(r))
    xs = []
    for fn in sorted(glob.glob(os.path.join(RUN, MCDIR, r.replace(" ", "_") + "_truth_s*.json"))):
        d = json.load(open(fn))
        x = np.array(d["x"])
        xs.append([x[UNKNOWN.index(k)] - xt[UNKNOWN.index(k)] for k in names])
    EST[r] = np.array(xs)
    sd = np.array([fim[r]["sd35_none"][k] for k in names])
    Z[r] = EST[r] / sd[None, :]

fig = plt.figure(figsize=(st.COL2, 4.6))
gs = GridSpec(2, 3, figure=fig, wspace=0.5, hspace=0.55, width_ratios=[1.7, 1, 1])

# A: empirical vs CRLB sd
ax = fig.add_subplot(gs[:, 0])
audit["A"] = {}
lo, hi = 1e-4, 1.2
for r in st.REGIMES:
    sd = np.array([fim[r]["sd35_none"][k] for k in names])
    emp = EST[r].std(axis=0, ddof=1)
    ax.scatter(sd, emp, s=9, color=st.REG_COLOR[r], edgecolor="white", lw=0.3,
               label=f"{st.REG_SHORT[r]} (n={len(EST[r])})", zorder=3)
    audit["A"][r] = dict(n=len(EST[r]), crlb=dict(zip(names, sd.tolist())),
                         empirical=dict(zip(names, emp.tolist())))
n = max(len(v) for v in EST.values())
g = np.logspace(np.log10(lo), np.log10(hi), 50)
q = np.sqrt(chi2.ppf([0.025, 0.975], n - 1) / (n - 1))
ax.fill_between(g, g * q[0], g * q[1], color=st.GREY, alpha=0.18, lw=0,
                label=f"95% band for n={n}")
ax.plot(g, g, color=st.INK, lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Cramér–Rao sd of log-parameter")
ax.set_ylabel("empirical sd across datasets (MLE)")
ax.legend(loc="upper left", fontsize=6.3)
ax.set_aspect("equal", adjustable="box")
st.panel(ax, "A", x=-0.2, y=1.02)

# B: z histograms
ax = fig.add_subplot(gs[0, 1:])
bins = np.linspace(-4.5, 4.5, 37)
xx = np.linspace(-4.5, 4.5, 200)
audit["B"] = {}
for r in st.REGIMES:
    z = Z[r].ravel()
    h, e = np.histogram(z, bins=bins, density=True)
    ax.step(0.5 * (e[1:] + e[:-1]), h, where="mid", color=st.REG_COLOR[r], lw=1.0,
            label=f"{st.REG_SHORT[r]}: var {z.var():.2f}")
    audit["B"][r] = dict(mean=float(z.mean()), var=float(z.var()), n=int(z.size))
ax.plot(xx, norm.pdf(xx), color=st.INK, lw=1.0, ls="--", label=r"$\mathcal{N}(0,1)$")
ax.set_xlabel(r"standardised error $(\hat x_j - x_j)/s_j^{\mathrm{CRLB}}$")
ax.set_ylabel("density")
ax.legend(fontsize=6.0, ncol=1, loc="upper right")
ax.set_xlim(-4.5, 4.5)
st.panel(ax, "B", x=-0.12, y=1.04)

# C: coverage per regime
ax = fig.add_subplot(gs[1, 1])
audit["C"] = {}
for j, r in enumerate(st.REGIMES):
    cov = (np.abs(Z[r]) < 1.96).mean(axis=0)
    ax.scatter(np.full(cov.size, j) + np.random.default_rng(j).uniform(-0.22, 0.22, cov.size),
               100 * cov, s=5, color=st.REG_COLOR[r], alpha=0.7, lw=0)
    ax.plot([j - 0.3, j + 0.3], [100 * cov.mean()] * 2, color=st.INK, lw=1.4)
    audit["C"][r] = dict(mean=float(cov.mean()), per_param=dict(zip(names, cov.tolist())))
ax.axhline(95, color=st.GREY, lw=0.7, ls="--")
ax.set_xticks(range(4)); ax.set_xticklabels([st.REG_SHORT[r] for r in st.REGIMES], fontsize=6.2, rotation=30, ha="right")
ax.set_ylabel(r"coverage of $\pm1.96\,s^{\mathrm{CRLB}}$ (%)")
ax.set_ylim(50, 102)
st.panel(ax, "C", x=-0.33, y=1.04)

# D: chi-square of the cost drop
ax = fig.add_subplot(gs[1, 2])
drops = json.load(open(os.path.join(RUN, MCDIR + "_costdrop.json"))) if os.path.exists(
    os.path.join(RUN, MCDIR + "_costdrop.json")) else {}
audit["D"] = drops
allv = np.concatenate([np.array(v) for v in drops.values()]) if drops else np.array([])
if allv.size:
    for r in st.REGIMES:
        v = np.sort(drops.get(r, []))
        if len(v):
            ax.plot(chi2.ppf((np.arange(1, len(v) + 1) - 0.5) / len(v), 35), v, "o",
                    ms=2.6, color=st.REG_COLOR[r])
    m = max(allv.max(), 70)
    ax.plot([0, m], [0, m], color=st.INK, lw=0.8)
    ax.set_xlim(10, m); ax.set_ylim(10, m)
ax.set_xlabel(r"$\chi^2_{35}$ quantile")
ax.set_ylabel(r"$2\,[\,\mathcal{L}(\theta^\ast)-\mathcal{L}(\hat\theta)\,]$")
st.panel(ax, "D", x=-0.33, y=1.04)

st.save(fig, "fig04_attained", audit)
print({r: (audit["B"][r]["var"], audit["C"][r]["mean"]) for r in st.REGIMES})
