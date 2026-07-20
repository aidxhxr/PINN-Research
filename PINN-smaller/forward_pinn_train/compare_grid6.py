"""6-panel forward-PINN vs scipy-reference overlay (all 4 regimes).

Matches the inverse-plot style: 2x3 grid over 6 states
(beta-catenin, MYC, HOXA13 / APC, RA, HOXA5 -- CYP26 dropped),
PINN solid + scipy reference dashed, per-panel legend, ATRA band.

Loads the trained final nets from the most-recent runs/<ts>/ and the
Radau reference (reference.py settings). Saves compare_grid6.png there.
"""
import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BASELINE, REGIMES, Y0, VAR_NAMES, VAR_LABELS, DEVICE
from model import ForwardPINN
from odes import _ode_rhs
from scipy.integrate import solve_ivp

RUN_DIR = max(glob.glob(os.path.join("runs", "*")), key=os.path.getmtime)
T = 150.0
N = 6000
XMAX = T

# SA-big regime palette (blue / green / purple / magenta)
COLORS = dict(zip(REGIMES, ["#1f77b4", "#2ca02c", "#7d3fbf", "#e0559a"]))


def ref_solution(name):
    p = {**BASELINE, **REGIMES[name]}
    t_eval = np.linspace(0, T, N)
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0,
                    t_eval=t_eval, method="Radau", rtol=1e-10, atol=1e-12)
    return t_eval, sol.y.T


def pinn_solution(name):
    safe = name.replace(" ", "_").replace("/", "_")
    net = ForwardPINN(T_max=T, width=256, depth=4).to(DEVICE)
    net.load_state_dict(torch.load(
        os.path.join(RUN_DIR, f"{safe}_final.pt"), map_location=DEVICE))
    net.eval()
    with torch.no_grad():
        t = torch.linspace(0, T, N, device=DEVICE).reshape(-1, 1)
        y = net(t).cpu().numpy()
    return t.cpu().numpy().ravel(), y


print(f"Loading references + PINN solutions from {RUN_DIR} ...")
refs = {name: ref_solution(name) for name in REGIMES}
pinns = {name: pinn_solution(name) for name in REGIMES}

# panels: (var symbol, full name), ordered like the inverse plot
PANELS = ["b", "m", "$h_{13}$", "p", "r", "$h_5$"]

fig, axes = plt.subplots(2, 3, figsize=(20, 9))
for ax, vn in zip(axes.flat, PANELS):
    i = VAR_NAMES.index(vn)
    for name in REGIMES:
        tr, yr = refs[name]
        tp, yp = pinns[name]
        ax.plot(tp, yp[:, i], "-", lw=2.0, color=COLORS[name],
                label=f"{name} (PINN)")
        ax.plot(tr, yr[:, i], "--", lw=1.2, color=COLORS[name], alpha=0.65,
                label=f"{name} (ref)")
    ax.axvspan(BASELINE["tau1"], BASELINE["tau2"], alpha=0.12,
               color="orange", label="ATRA")
    ax.set_xlim(0, XMAX)
    ax.set_title(VAR_LABELS[i])
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(vn)
    ax.legend(fontsize=6, ncol=2, loc="best")

fig.tight_layout()
out = os.path.join(RUN_DIR, "compare_grid6.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")
