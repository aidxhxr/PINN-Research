"""Fig 6 -- the Schur-complement Gauss-Newton PINN.

(A) structure of its Gauss-Newton matrix: ten dense head blocks coupled only
    through the 36 parameters (block arrowhead) -> one 36x36 Schur complement;
(B) lambda continuation on the held-out seed-7 data: physics residual and
    parameters within 10% along the iterations (chosen schedule per regime);
(C) per-parameter |log error|: SGN-PINN against the classical MLE (both seeds);
(D) SGN's own standard errors (diag of the inverse Schur complement) against
    the CRLB of the design.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import style as st
import paperdata as P

st.setup()
audit = {"B": {}, "C": {}, "D": {}}
fig = plt.figure(figsize=(st.COL2, 5.6))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1, 1],
              wspace=0.55, hspace=0.6)

# ---- A: block arrowhead ----------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
nC, hb, pb = 10, 1.0, 0.55
tot = nC * hb + pb
for c in range(nC):
    y0 = tot - (c + 1) * hb
    ax.add_patch(Rectangle((c * hb, y0), hb, hb, color="#6baed6", lw=0))
    ax.add_patch(Rectangle((nC * hb, y0), pb, hb, color="#fd8d3c", lw=0))
    ax.add_patch(Rectangle((c * hb, 0), hb, pb, color="#fd8d3c", lw=0))
ax.add_patch(Rectangle((nC * hb, 0), pb, pb, color=st.RED, lw=0))
ax.add_patch(Rectangle((0, 0), tot, tot, fill=False, lw=0.8, color=st.INK))
ax.set_xlim(-0.2, tot + 2.6); ax.set_ylim(-4.2, tot + 0.3)
ax.set_aspect("equal"); ax.axis("off")
ax.text(0.3, 2.2, "head blocks $\\mathcal{A}_c$\n(10 conditions,\n5551 unknowns each)", ha="left", va="center", fontsize=6.0)
ax.text(nC * hb + pb + 0.2, tot - 2.5, "$\\mathcal{B}_c$:\ncoupling\nthrough $\\theta$", ha="left", va="center", fontsize=6.0, color="#d9480f")
ax.text(nC * hb + pb + 0.2, pb / 2, "$\\mathcal{X}$ (36×36)", ha="left", va="center", fontsize=6.0, color=st.RED)
ax.text(tot / 2 + 1, -0.5,
        r"Schur complement $\mathcal{S}=\sum_c\mathcal{X}_c-\mathcal{B}_c^{\top}\mathcal{A}_c^{-1}\mathcal{B}_c$"
        "\n(36×36): the LM step, and the\nprofiled Fisher information",
        ha="center", va="top", fontsize=6.1)
st.panel(ax, "A", x=-0.12, y=1.0)

# ---- B: lambda continuation ------------------------------------------------------
sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1:], hspace=0.12)
axp = fig.add_subplot(sub[0]); axc = fig.add_subplot(sub[1], sharex=axp)
seed = 7
for r in st.REGIMES:
    run = P.sgn(seed, r)
    if run is None:
        continue
    h = run["history"]
    it = np.arange(len(h))
    axp.semilogy(it, [max(x["phys"], 1e-3) for x in h], color=st.REG_COLOR[r], lw=1.0)
    axc.plot(it, [x["under10"] for x in h], color=st.REG_COLOR[r], lw=1.1,
             label=st.REG_SHORT[r])
    audit["B"][r] = dict(tag=run["args"]["tag"], lam=[x["lam"] for x in h],
                         phys=[x["phys"] for x in h], under10=[x["under10"] for x in h])
    # lambda level boundaries (from the first regime only)
    if r == "Normal":
        lam = [x["lam"] for x in h]
        for i in range(1, len(lam)):
            if lam[i] != lam[i - 1]:
                for a in (axp, axc):
                    a.axvline(i - 0.5, color=st.GRID, lw=0.8, zorder=0)
        axp.text(0.5, 1.03, r"$\lambda$ continued upward to $10^3$ (grey lines: levels of the Normal run)",
                 transform=axp.transAxes, ha="center", fontsize=6.0, color=st.INK2)
axp.set_ylabel("physics\nresidual")
axc.set_ylabel("within 10%\n(of 36)")
axc.set_xlabel("Gauss–Newton iteration (held-out seed-7 data)")
axc.set_ylim(0, 37)
plt.setp(axp.get_xticklabels(), visible=False)
axc.legend(ncol=4, fontsize=6.0, loc="lower right")
st.panel(axp, "B", x=-0.13, y=1.08)

# ---- C: SGN vs MLE -----------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
for r in st.REGIMES:
    for seed, mk in ((42, "o"), (7, "D")):
        a, b = P.recovered("SGN-PINN", seed, r), P.recovered("MLE (4 starts)", seed, r)
        if a is None or b is None:
            continue
        la, lb = P.log_errors(a, r), P.log_errors(b, r)
        xs = [max(abs(lb[k]), 1e-6) for k in P.REGULAR]
        ys = [max(abs(la[k]), 1e-6) for k in P.REGULAR]
        ax.scatter(xs, ys, s=7, marker=mk, color=st.REG_COLOR[r], alpha=0.75, lw=0)
        audit["C"][f"{r}/seed{seed}"] = dict(zip(P.REGULAR, zip(xs, ys)))
g = np.logspace(-6, 1, 10)
ax.plot(g, g, color=st.INK, lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-6, 10); ax.set_ylim(1e-6, 10)
ax.set_xlabel("|log error|, classical MLE (best of 4)")
ax.set_ylabel("|log error|, SGN-PINN")
ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=st.INK2, ms=3.5, label="seed 42"),
                   Line2D([], [], marker="D", ls="none", color=st.INK2, ms=3.5, label="seed 7 (held out)")],
          fontsize=6.0, loc="upper left")
st.panel(ax, "C", x=-0.32, y=1.03)

# ---- D: own standard errors vs CRLB --------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
for r in st.REGIMES:
    sd = P.crlb(r)
    for seed, mk in ((42, "o"), (7, "D")):
        run = P.sgn(seed, r)
        if run is None:
            continue
        own = run["sd_logtheta"]
        ks = [k for k in P.REGULAR if k in own]
        ax.scatter([sd[k] for k in ks], [own[k] for k in ks], s=7, marker=mk,
                   color=st.REG_COLOR[r], alpha=0.75, lw=0)
        audit["D"][f"{r}/seed{seed}"] = {k: (sd[k], own[k]) for k in ks}
g = np.logspace(-4.5, 1, 10)
ax.plot(g, g, color=st.INK, lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, 3); ax.set_ylim(1e-4, 3)
ax.set_xlabel("Cramér–Rao sd (at truth)")
ax.set_ylabel("SGN-PINN's own sd")
st.panel(ax, "D", x=-0.3, y=1.03)

# ---- E: wall-clock --------------------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
COST = json.load(open(os.path.join(P.RUN, "wallclock.json"))) if os.path.exists(
    os.path.join(P.RUN, "wallclock.json")) else {}
labs = list(COST.keys())
for i, k in enumerate(labs):
    lo, hi = COST[k]["range_s"]
    ax.plot([lo / 60, hi / 60], [i, i], color=COST[k]["color"], lw=5, solid_capstyle="butt")
    ax.text(hi / 60 * 1.15, i, COST[k]["device"], va="center", fontsize=5.8, color=st.INK2)
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=6.2)
ax.set_xscale("log"); ax.set_xlim(0.5, 600)
ax.set_xlabel("wall clock per regime (min)")
ax.invert_yaxis(); ax.grid(axis="y", lw=0)
audit["E"] = COST
st.panel(ax, "E", x=-0.55, y=1.03)

st.save(fig, "fig06_sgn", audit)
