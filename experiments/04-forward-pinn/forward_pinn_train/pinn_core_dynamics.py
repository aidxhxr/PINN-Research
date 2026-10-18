"""Core WNT-HOX regulatory dynamics -- FORWARD PINN output (no scipy).

Same 4 proteins / 4 regimes as scipy_core_dynamics*.py, but every curve is
the trained forward PINN's own prediction z_theta(t) (the network output),
loaded from the most recent runs/<ts>/<regime>_final.pt.

Writes two figures next to this file:
  pinn_forward_core_dynamics.png         (2x2 panels)
  pinn_forward_core_dynamics_single.png  (all 16 curves on one axes)
"""
import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from config import BASELINE, REGIMES, Y0, VAR_NAMES, DEVICE
from model import ForwardPINN
from odes import _ode_rhs

T = 150.0
N = 6000
TITLE = "Core WNT-HOX Regulatory Dynamics via Forward PINN"
MODEL = "PINN"          # legend tag for the network curves

RUN_DIR = max(glob.glob(os.path.join("runs", "*")), key=os.path.getmtime)

COLORS = {
    "Normal":           "#1f77b4",
    "Early Adenoma":    "#2ca02c",
    "Advanced Adenoma": "#ff7f0e",
    "Severe APC Loss":  "#d40000",
}

# (state symbol, panel title, single-panel label, line style)
SPECIES = [
    ("b",        r"$\beta$-catenin: WNT activity",
     r"$\beta$-catenin (WNT activity)", "-"),
    ("$h_{13}$", "HOXA13: WNT reinforcement",
     "HOXA13 (WNT reinforcement)", "--"),
    ("p",        "APC: WNT suppression",
     "APC (WNT suppression)", "-."),
    ("$h_5$",    "HOXA5: differentiation",
     "HOXA5 (differentiation)", ":"),
]

tau1, tau2 = BASELINE["tau1"], BASELINE["tau2"]
BAND = dict(color="0.6", alpha=0.13)

# ODE reference = thick translucent solid (the ground truth the eye reads as
# the curve), PINN = thin dark dashed drawn on top, so the two stay
# distinguishable even where they overlap (which is nearly all of the domain --
# that agreement is the point of the figure).
ODE_KW = dict(ls="-", lw=3.4, alpha=0.45, solid_capstyle="round")
PINN_KW = dict(ls="--", lw=1.7, dashes=(5.5, 3.5))


def _dark(hexcolor, f=0.55):
    """Darkened variant of a hex colour (f = fraction of the original kept)."""
    c = matplotlib.colors.to_rgb(hexcolor)
    return tuple(f * ch for ch in c)


def pinn_solution(name):
    safe = name.replace(" ", "_").replace("/", "_")
    net = ForwardPINN(T_max=T, width=256, depth=4).to(DEVICE)
    net.load_state_dict(torch.load(os.path.join(RUN_DIR, f"{safe}_final.pt"),
                                   map_location=DEVICE))
    net.eval()
    with torch.no_grad():
        t = torch.linspace(0, T, N, device=DEVICE).reshape(-1, 1)
        y = net(t).cpu().numpy()
    return np.linspace(0, T, N), y


def ode_solution(name):
    """Ground-truth trajectory from the stiff ODE solver (scipy Radau)."""
    p = {**BASELINE, **REGIMES[name]}
    t_eval = np.linspace(0, T, N)
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0,
                    t_eval=t_eval, method="Radau", rtol=1e-10, atol=1e-12)
    assert sol.success, f"{name} failed: {sol.message}"
    return t_eval, sol.y.T


def response_window(sols, panels):
    """Half-response ATRA window: the 50% crossing of the switch-on and the
    switch-off transition, median over the plotted states and regimes.

    The pulse itself is exactly [tau1, tau2], but the states lag it (~1-2 tau
    at switch-on, ~2-4 tau at switch-off), so a box drawn at [tau1, tau2] puts
    the whole ON ramp inside and the whole OFF ramp outside -- it reads as
    starting too early and ending too soon. Cutting each transition at its
    halfway point puts half of each ramp inside the box, so the bump sits
    centred. The dotted guides stay at the true tau1 / tau2.
    """
    ons, offs = [], []
    for name in REGIMES:
        t, y = sols[name]
        for vn, *_ in panels:
            s = y[:, VAR_NAMES.index(vn)]
            base = np.median(s[(t > 30) & (t < tau1 - 1)])
            plat = np.median(s[(t > 60) & (t < tau2 - 3)])
            post = np.median(s[(t > 100) & (t < 130)])
            if abs(plat - base) < 1e-3 or abs(post - plat) < 1e-3:
                continue
            m = (t >= tau1) & (t <= tau1 + 20)
            ons.append(t[m][np.argmin(np.abs(s[m] - (base + 0.5*(plat-base))))])
            m = (t >= tau2) & (t <= tau2 + 20)
            offs.append(t[m][np.argmin(np.abs(s[m] - (plat + 0.5*(post-plat))))])
    return float(np.median(ons)), float(np.median(offs))


def _band(ax):
    ax.axvspan(BAND_LO, BAND_HI, lw=0, zorder=0, **BAND)


def _clean(ax):
    ax.set_xlim(0, T)
    ax.grid(axis="y", color="0.9", lw=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_grid(sols, refs, out):
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 17,
                         "axes.labelsize": 14})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (vn, title, _lab, _ls) in zip(axes.flat, SPECIES):
        i = VAR_NAMES.index(vn)
        _band(ax)
        for name in REGIMES:
            t, y = sols[name]
            tr, yr = refs[name]
            ax.plot(tr, yr[:, i], color=COLORS[name], zorder=3, **ODE_KW)
            ax.plot(t, y[:, i], color=_dark(COLORS[name]), zorder=4,
                    **PINN_KW)
        _clean(ax)
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("Dimensionless Concentration")

    # column-major legend: one column per regime, PINN (solid) over ODE (dashed)
    handles, labels = [], []
    for n in REGIMES:
        handles += [plt.Line2D([], [], color=_dark(COLORS[n]), lw=2.0,
                               ls="--", dashes=(5.5, 3.5)),
                    plt.Line2D([], [], color=COLORS[n], lw=3.4, alpha=0.65)]
        labels += [f"{n} ({MODEL})", f"{n} (ODE solver)"]
    handles += [plt.Rectangle((0, 0), 1, 1, **BAND),
                plt.Line2D([], [], color="none")]
    labels += ["ATRA Window", ""]
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.955), fontsize=13, handlelength=2.4,
               columnspacing=2.0, labelspacing=0.4)
    fig.suptitle(TITLE, fontsize=24, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.875])
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def plot_single(sols, out):
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
    fig, ax = plt.subplots(figsize=(16, 9))
    _band(ax)
    for name in REGIMES:
        t, y = sols[name]
        for vn, _title, _lab, ls in SPECIES:
            ax.plot(t, y[:, VAR_NAMES.index(vn)], ls, lw=2.0,
                    color=COLORS[name], zorder=3)
    _clean(ax)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Dimensionless Concentration")

    reg_h = [plt.Line2D([], [], color=COLORS[n], lw=3.5) for n in REGIMES]
    reg_h.append(plt.Rectangle((0, 0), 1, 1, **BAND))
    leg1 = ax.legend(reg_h, list(REGIMES) + ["ATRA Window"], loc="upper left",
                     bbox_to_anchor=(0.0, -0.10), ncol=5, frameon=False,
                     fontsize=13, title="Regime",
                     title_fontproperties={"weight": "bold", "size": 13})
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)

    sp_h = [plt.Line2D([], [], color="0.25", lw=2.2, ls=ls)
            for _vn, _t, _lab, ls in SPECIES]
    leg2 = ax.legend(sp_h, [lab for _vn, _t, lab, _ls in SPECIES],
                     loc="upper left", bbox_to_anchor=(0.0, -0.24), ncol=4,
                     frameon=False, fontsize=13, title="Species",
                     title_fontproperties={"weight": "bold", "size": 13})
    leg2._legend_box.align = "left"

    ax.set_title(TITLE, fontsize=24, fontweight="bold", pad=16)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print(f"Loading forward-PINN nets from {RUN_DIR} ...")
    sols = {name: pinn_solution(name) for name in REGIMES}
    print("Solving the ODE reference (scipy Radau) ...")
    refs = {name: ode_solution(name) for name in REGIMES}
    BAND_LO, BAND_HI = response_window(refs, SPECIES)
    print(f"ATRA pulse window   [{tau1:.0f}, {tau2:.0f}]")
    print(f"half-response band  [{BAND_LO:.2f}, {BAND_HI:.2f}]")
    here = os.path.dirname(os.path.abspath(__file__))
    plot_grid(sols, refs, os.path.join(here, "pinn_forward_core_dynamics.png"))
    plot_single(sols, os.path.join(here,
                                   "pinn_forward_core_dynamics_single.png"))
