"""Intro figure: baseline dynamics across the four regimes.

Left: beta-catenin b(t) — the WNT drive saturating it as severity rises.
Right: APC p(t) — the loss that defines the disease axis.
Solved with the repo's own Radau reference (rtol 1e-10), baseline condition.
"""
import sys, os
sys.path.insert(0, "/home/29/aidahxr/PINN-Research/PINN-hybrid-ude")
import numpy as np
from scipy.integrate import solve_ivp
from config import BASELINE, REGIMES, CONDITIONS, Y0
from odes import _ode_rhs
import style
from style import REGIME_COLORS, INK2
import matplotlib.pyplot as plt

style.setup()

forcing = CONDITIONS[0]["forcing"]
T, n = 150.0, 1500
sols = {}
for name in REGIMES:
    p = {**BASELINE, **REGIMES[name], **forcing}
    s = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0,
                  t_eval=np.linspace(0, T, n), method="Radau",
                  rtol=1e-10, atol=1e-12)
    assert s.success
    sols[name] = (s.t, s.y.T)

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
for ax, (idx, label) in zip(axes, [(0, "β-catenin  b(τ)"), (1, "APC  p(τ)")]):
    for name, (t, y) in sols.items():
        ax.plot(t, y[:, idx], color=REGIME_COLORS[name], lw=2.4,
                solid_capstyle="round", zorder=3)
    ax.set_title(label, loc="left", fontsize=18, pad=8)
    ax.set_xlabel("dimensionless time τ")
    ax.set_xlim(0, 150)
    style.clean(ax)

# direct end labels on the left panel (series separate there), legend-free right
t, _ = sols["Normal"]
ends = {name: y[-1, 0] for name, (_, y) in sols.items()}
short = {"Normal": "Normal", "Early Adenoma": "Early",
         "Advanced Adenoma": "Advanced", "Severe APC Loss": "Severe"}
# nudge collided labels apart
order = sorted(ends, key=lambda k: ends[k])
ys = [ends[k] for k in order]
for i in range(1, len(ys)):
    if ys[i] - ys[i-1] < 0.055:
        ys[i] = ys[i-1] + 0.055
for name, yy in zip(order, ys):
    axes[0].annotate(short[name], (150, ends[name]), xytext=(154, yy),
                     color=REGIME_COLORS[name], fontsize=15, fontweight=600,
                     va="center", annotation_clip=False)
axes[0].set_xlim(0, 150)
axes[0].margins(x=0)

# APC panel: label only the two extremes
apc_ends = {n2: y[-1, 1] for n2, (_, y) in sols.items()}
for name in ("Normal", "Severe APC Loss"):
    axes[1].annotate(short[name], (150, apc_ends[name]),
                     xytext=(154, apc_ends[name]), va="center",
                     color=REGIME_COLORS[name], fontsize=15, fontweight=600,
                     annotation_clip=False)

fig.get_layout_engine().set(rect=(0, 0, 0.94, 1))
style.save(fig, "fig_regimes")
