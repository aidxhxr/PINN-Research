"""S1 Fig -- how the information budget depends on the design.

Source: experiments/17-sensitivity-pinn/runs/20261017_paper/fim2_base/<regime>_Jcond.npy, the
per-condition sensitivities d y / d log(theta) of the benchmark design
(nC=10, n_t=150, 7 states, 36 parameters) at the truth.
(A) CRLB-expected count E[N10] (34 regular parameters, deltaP1 known) against
    the noise sd (the bound scales linearly in sigma);
(B) E[N10] at sigma = 0.002 when only a subset of states is observed;
(C) CRLB of thetaP when one base condition is left out, relative to the full
    design (all ten conditions, not only the WNT/MYC four).
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as st

st.setup()
sys.path.insert(0, os.path.join(st.REPO, "src"))
from wnt_pinn.model.parameters import UNKNOWN, CONDITIONS  # noqa: E402

SRC = os.path.join(st.REPO, "experiments/17-sensitivity-pinn/runs/20261017_paper/fim2_base")
SIG = 0.002
keep = [i for i, k in enumerate(UNKNOWN) if k != "deltaP1"]
reg_idx = [keep.index(UNKNOWN.index(k)) for k in UNKNOWN if k not in ("thetaP", "deltaP1")]
th_idx = keep.index(UNKNOWN.index("thetaP"))
LO, HI = np.log(0.9), np.log(1.1)


def sds(J, sigma=SIG):
    F = J.T @ J / sigma ** 2
    F = F[np.ix_(keep, keep)]
    ev, V = np.linalg.eigh(0.5 * (F + F.T))
    ev = np.maximum(ev, 1e-14 * ev.max())
    return np.sqrt((V ** 2 / ev[None, :]).sum(axis=1))


def expected(sd):
    s = sd[reg_idx]
    return float((norm.cdf(HI / s) - norm.cdf(LO / s)).sum())


Jc = {r: np.load(os.path.join(SRC, r.replace(" ", "_") + "_Jcond.npy")) for r in st.REGIMES}
audit = {"source": SRC, "A": {}, "B": {}, "C": {}}
fig = plt.figure(figsize=(st.COL2, 2.7))
gs = GridSpec(1, 3, figure=fig, wspace=0.85, width_ratios=[1, 1.0, 1.05])

# A: noise
ax = fig.add_subplot(gs[0])
sig = np.logspace(-4, -0.5, 30)
for r in st.REGIMES:
    J = Jc[r].reshape(-1, Jc[r].shape[-1])
    base = sds(J, 1.0)
    e = [expected(base * s) for s in sig]
    ax.semilogx(sig, e, color=st.REG_COLOR[r], lw=1.2, label=st.REG_SHORT[r])
    audit["A"][r] = dict(sigma=sig.tolist(), expected=e)
ax.axvline(SIG, color=st.GREY, lw=0.7, ls="--")
ax.text(SIG * 1.15, 34.5, "benchmark", fontsize=6, color=st.INK2, va="top", rotation=90)
ax.set_xlabel(r"noise sd $\sigma$")
ax.set_ylabel(r"CRLB-expected count (of 34)")
ax.set_ylim(0, 35)
ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 0.0))
st.panel(ax, "A", x=-0.3, y=1.03)

# B: partial observation
SUBS = [("all 7", [0, 1, 2, 3, 4, 5, 6]), ("no APC", [0, 2, 3, 4, 5, 6]),
        ("β-cat, HOXs, MYC", [0, 2, 3, 4]), ("β-cat, MYC, RA", [0, 4, 5]),
        ("β-cat, MYC", [0, 4]), ("β-cat only", [0])]
ax = fig.add_subplot(gs[1])
w = 0.2
for j, r in enumerate(st.REGIMES):
    vals = []
    for name, ss in SUBS:
        J = Jc[r][:, :, ss, :].reshape(-1, Jc[r].shape[-1])
        vals.append(expected(sds(J)))
    ax.barh(np.arange(len(SUBS)) + (j - 1.5) * w, vals, w, color=st.REG_COLOR[r], lw=0)
    audit["B"][r] = dict(zip([n for n, _ in SUBS], vals))
ax.set_yticks(range(len(SUBS))); ax.set_yticklabels([n for n, _ in SUBS], fontsize=6.4)
ax.invert_yaxis()
ax.set_xlabel("CRLB-expected count (of 34)")
ax.set_xlim(0, 35)
ax.grid(axis="y", lw=0)
st.panel(ax, "B", x=-0.62, y=1.03)

# C: leave-one-out for thetaP
ax = fig.add_subplot(gs[2])
names = [c["name"] for c in CONDITIONS]
for j, r in enumerate(st.REGIMES):
    full = sds(Jc[r].reshape(-1, Jc[r].shape[-1]))[th_idx]
    rat = []
    for ci in range(len(names)):
        J = np.delete(Jc[r], ci, axis=0).reshape(-1, Jc[r].shape[-1])
        rat.append(sds(J)[th_idx] / full)
    ax.plot(rat, np.arange(len(names)) + (j - 1.5) * 0.12, "o", ms=3.2,
            color=st.REG_COLOR[r], label=st.REG_SHORT[r])
    audit["C"][r] = dict(full_sd=float(full), ratio=dict(zip(names, rat)))
ax.axvline(1, color=st.GREY, lw=0.7)
ax.set_xlim(0.7, 7)
ax.set_xticks([1, 2, 4, 6]); ax.minorticks_off()
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.2)
ax.invert_yaxis()
ax.set_xlabel(r"$\theta_P$ bound, one condition" + "\n" + "left out / all ten")
ax.grid(axis="y", lw=0.3)
st.panel(ax, "C", x=-0.45, y=1.03)

st.save(fig, "figS1_design_sensitivity", audit)
print(json.dumps(audit["B"], indent=0))
print({r: {k: round(v, 2) for k, v in audit["C"][r]["ratio"].items()} for r in st.REGIMES})
