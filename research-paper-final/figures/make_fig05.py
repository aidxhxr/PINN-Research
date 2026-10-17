"""Fig 5 -- the estimator gap.

(A) parameters recovered within 10% (of the 34 outside the degenerate
    thetaP/deltaP1 pair) per regime, for the estimator ladder on the benchmark
    data (bars: seed 42; diamonds: held-out seed 7) and the CRLB-expected count;
(B) per-parameter |log error| against the CRLB sd, integral PINN vs MLE;
(C) median standardised error |x_hat - x|/s_CRLB along the ladder, per regime;
    an efficient estimator sits at 0.67 (median of |N(0,1)|).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style as st
import paperdata as P

st.setup()
EST = P.ESTIMATORS
audit = {"estimators": EST, "A": {}, "B": {}, "C": {}}

fig = plt.figure(figsize=(st.COL2, 5.0))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.05], hspace=0.55, wspace=0.3)

# ---- A -------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
nE = len(EST)
w = 0.8 / (nE + 1)
for j, r in enumerate(st.REGIMES):
    for i, e in enumerate(EST):
        x = j + (i - nE / 2) * w
        for seed, kind in ((42, "bar"), (7, "pt")):
            rec = P.recovered(e, seed, r)
            if rec is None:
                continue
            n = P.count(rec, r, P.REGULAR)
            audit["A"][f"{r}/{e}/seed{seed}"] = n
            if kind == "bar":
                ax.bar(x, n, w * 0.92, color=P.EST_COLOR[e], lw=0, zorder=2)
            else:
                ax.plot(x, n, marker="D", ms=3.2, color="white", mec=st.INK, mew=0.7,
                        ls="none", zorder=4)
    ce = P.crlb_expected(r)
    audit["A"][f"{r}/CRLB"] = ce
    xb = j + (nE - nE / 2) * w
    ax.bar(xb, ce, w * 0.92, color="white", edgecolor=st.INK, lw=0.9, hatch="////", zorder=2)
ax.set_xticks(range(4))
ax.set_xticklabels([st.REG_SHORT[r] for r in st.REGIMES])
ax.set_ylabel("recovered within 10%\n(of 34 parameters)")
ax.set_ylim(0, 36)
ax.axhline(34, color=st.GREY, lw=0.6, ls="--")
h = [Patch(color=P.EST_COLOR[e], label=e) for e in EST]
h += [Patch(facecolor="white", edgecolor=st.INK, hatch="////", label="CRLB-expected"),
      Line2D([], [], marker="D", ls="none", color="white", mec=st.INK, ms=4,
             label="held-out dataset (seed 7)")]
ax.legend(handles=h, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          fontsize=6.3, columnspacing=1.0, handlelength=1.2)
st.panel(ax, "A", x=-0.075, y=1.12)

# ---- B -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
for e, mk in (("PINN (integral)", "o"), ("MLE", "s")):
    xs, ys = [], []
    for r in st.REGIMES:
        sd = P.crlb(r)
        for seed in (42, 7):
            rec = P.recovered(e, seed, r)
            if rec is None:
                continue
            le = P.log_errors(rec, r)
            xs += [sd[k] for k in P.REGULAR]
            ys += [abs(le[k]) for k in P.REGULAR]
            ax.scatter([sd[k] for k in P.REGULAR], [max(abs(le[k]), 1e-6) for k in P.REGULAR],
                       s=6, marker=mk, color=P.EST_COLOR[e], alpha=0.75, lw=0)
    audit["B"][e] = dict(n=len(xs), median_ratio=float(np.median(np.array(ys) / np.array(xs))))
g = np.logspace(-4.5, 0.5, 20)
ax.plot(g, g, color=st.INK, lw=0.8)
ax.plot(g, 10 * g, color=st.INK, lw=0.5, ls=":")
ax.plot(g, 100 * g, color=st.INK, lw=0.5, ls=":")
ax.axhline(np.log(1.1), color=st.GREY, lw=0.6, ls="--")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, 1); ax.set_ylim(1e-8, 10)
ax.set_xlabel("Cramér–Rao sd of log-parameter")
ax.set_ylabel("|log error| of the estimate")
ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=P.EST_COLOR["PINN (integral)"], label="PINN (integral)"),
                   Line2D([], [], marker="s", ls="none", color=P.EST_COLOR["MLE"], label="MLE"),
                   Line2D([], [], color=st.INK, lw=0.8, label="error = bound"),
                   Line2D([], [], color=st.INK, lw=0.5, ls=":", label="10×, 100× bound"),
                   Line2D([], [], color=st.GREY, lw=0.6, ls="--", label="10% error")],
          loc="lower right", fontsize=5.8, ncol=2, columnspacing=0.8, handlelength=1.6)
st.panel(ax, "B", x=-0.2, y=1.03)

# ---- C -------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
for r in st.REGIMES:
    sd = P.crlb(r)
    ys = []
    for e in EST:
        vals = []
        for seed in (42, 7):
            rec = P.recovered(e, seed, r)
            if rec is None:
                continue
            le = P.log_errors(rec, r)
            vals += [abs(le[k]) / sd[k] for k in P.REGULAR]
        ys.append(np.median(vals) if vals else np.nan)
    audit["C"][r] = dict(zip(EST, [None if np.isnan(v) else float(v) for v in ys]))
    ax.plot(range(len(EST)), ys, "-o", color=st.REG_COLOR[r], ms=3.4, lw=1.1,
            label=st.REG_SHORT[r])
ax.axhline(0.674, color=st.INK, lw=0.8, ls="--")
ax.text(len(EST) - 1, 0.674 * 1.35, "efficient estimator", ha="right", fontsize=6.0)
ax.set_yscale("log")
ax.set_xticks(range(len(EST)))
ax.set_xticklabels([e.replace(" (", "\n(") for e in EST], fontsize=6.0)
ax.set_ylabel("median |error| / CRLB sd")
ax.legend(fontsize=6.2, loc="upper right")
st.panel(ax, "C", x=-0.2, y=1.03)

st.save(fig, "fig05_estimator_gap", audit)
print(audit["A"])
print(audit["C"])
