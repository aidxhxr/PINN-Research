"""Forward PINN fit, two panels: beta-catenin b and HOXA5 h5.

Replaces the 4-panel assets/ready/forward_fit.png. Per regime the scipy Radau
reference is drawn thick + translucent and the trained forward PINN's own
prediction z_theta(t) thin + dashed on top (logic copied from
PINN-smaller/forward-pinn-train-hybrid/pinn_core_dynamics.py). Nets come from
the fixed run dir below (40 sparse observations + IC + physics).
"""
import sys
import os

SRC = "/home/29/aidahxr/PINN-Research/PINN-smaller/forward-pinn-train-hybrid"
sys.path.insert(0, SRC)

import numpy as np
import torch
import matplotlib
from scipy.integrate import solve_ivp

from config import BASELINE, REGIMES, Y0, VAR_NAMES
from model import ForwardPINN
from odes import _ode_rhs

import style
from style import REGIME_COLORS, INK, INK2, GREY, SURFACE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

RUN_DIR = os.path.join(SRC, "runs", "20260712_204546")
DEVICE = torch.device("cpu")
T, N = 150.0, 6000
tau1, tau2 = BASELINE["tau1"], BASELINE["tau2"]

# ODE reference = thick translucent solid; PINN = thin dark dashed on top.
ODE_KW = dict(ls="-", lw=4.0, alpha=0.42, solid_capstyle="round")
PINN_KW = dict(ls="--", lw=1.9, dashes=(5.5, 3.5))

PANELS = [("b", r"$\beta$-catenin   b($\tau$)"),
          (r"$h_5$", r"HOXA5   $h_5$($\tau$)")]


def _dark(hexcolor, f=0.55):
    c = matplotlib.colors.to_rgb(hexcolor)
    return tuple(f * ch for ch in c)


def pinn_solution(name):
    safe = name.replace(" ", "_").replace("/", "_")
    net = ForwardPINN(T_max=T, width=256, depth=4).to(DEVICE)
    net.load_state_dict(torch.load(os.path.join(RUN_DIR, f"{safe}_final.pt"),
                                   map_location="cpu"))
    net.eval()
    with torch.no_grad():
        t = torch.linspace(0, T, N, device=DEVICE).reshape(-1, 1)
        y = net(t).cpu().numpy()
    return np.linspace(0, T, N), y


def ode_solution(name):
    p = {**BASELINE, **REGIMES[name]}
    t_eval = np.linspace(0, T, N)
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0,
                    t_eval=t_eval, method="Radau", rtol=1e-10, atol=1e-12)
    assert sol.success, f"{name} failed: {sol.message}"
    return t_eval, sol.y.T


print(f"Loading forward-PINN nets from {RUN_DIR} ...")
sols = {n: pinn_solution(n) for n in REGIMES}
print("Solving the ODE reference (scipy Radau) ...")
refs = {n: ode_solution(n) for n in REGIMES}

style.setup()
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))

for ax, (vn, title) in zip(axes, PANELS):
    i = VAR_NAMES.index(vn)
    for name in REGIMES:
        t, y = sols[name]
        tr, yr = refs[name]
        ax.plot(tr, yr[:, i], color=REGIME_COLORS[name], zorder=3, **ODE_KW)
        ax.plot(t, y[:, i], color=_dark(REGIME_COLORS[name]), zorder=4,
                **PINN_KW)
    ax.set_title(title, loc="left", fontsize=27, pad=10)
    ax.set_xlabel(r"dimensionless time $\tau$")
    ax.set_xlim(0, T)
    ax.set_xticks([0, 50, 100, 150])
    style.clean(ax)

axes[0].set_ylabel("concentration")

# headroom for the ATRA arrow, then draw it in blended (data-x, axes-y) coords
for ax in axes:
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.16 * (hi - lo))
    tf = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate("", xy=(tau2, 0.93), xytext=(tau1, 0.93), xycoords=tf,
                textcoords=tf, zorder=6,
                arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.4",
                                color=INK2, lw=1.6, shrinkA=0, shrinkB=0))
    ax.text(tau2 + 4, 0.93, "ATRA", transform=tf, fontsize=20,
            color=INK2, ha="left", va="center", zorder=6)

SHORT = {"Normal": "Normal", "Early Adenoma": "Early",
         "Advanced Adenoma": "Advanced", "Severe APC Loss": "Severe APC"}
handles = [Line2D([], [], color=REGIME_COLORS[n], lw=4.0, alpha=0.75,
                  solid_capstyle="round", label=SHORT[n])
           for n in REGIME_COLORS]
handles.append(Line2D([], [], color="0.35", lw=1.9, ls="--",
                      dashes=(5.5, 3.5), label="PINN (dashed)"))
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=19,
           handlelength=1.5, columnspacing=1.0, handletextpad=0.5,
           borderaxespad=0.0)
fig.get_layout_engine().set(rect=(0, 0.11, 1, 0.89))
style.save(fig, "fig_forward_fit")
