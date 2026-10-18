"""Core WNT-HOX regulatory dynamics -- pure scipy (Radau) solutions.

No neural networks: the 4 regimes are integrated directly with solve_ivp at
reference tolerances and plotted as a 2x2 figure
(A beta-catenin, B HOXA13, C APC, D HOXA5) with the ATRA window shaded.

Saves scipy_core_dynamics.png next to this file.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from config import BASELINE, REGIMES, Y0, VAR_NAMES
from odes import _ode_rhs

T = 150.0
N = 6000

# SA-big regime palette; Severe APC Loss forced to red
COLORS = {
    "Normal":           "#1f77b4",
    "Early Adenoma":    "#2ca02c",
    "Advanced Adenoma": "#ff7f0e",
    "Severe APC Loss":  "#d40000",
}

PANELS = [
    ("b",        r"$\beta$-catenin: WNT activity"),
    ("$h_{13}$", "HOXA13: WNT reinforcement"),
    ("p",        "APC: WNT suppression"),
    ("$h_5$",    "HOXA5: differentiation"),
]


def ref_solution(name):
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
    tau1, tau2 = BASELINE["tau1"], BASELINE["tau2"]
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


print("Solving all 4 regimes with scipy Radau ...")
sols = {name: ref_solution(name) for name in REGIMES}
BAND_LO, BAND_HI = response_window(sols, PANELS)
print(f"ATRA pulse window   [{BASELINE['tau1']:.0f}, {BASELINE['tau2']:.0f}]")
print(f"half-response band  [{BAND_LO:.2f}, {BAND_HI:.2f}]")

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
})

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
tau1, tau2 = BASELINE["tau1"], BASELINE["tau2"]

for ax, (vn, title) in zip(axes.flat, PANELS):
    i = VAR_NAMES.index(vn)
    ax.axvspan(BAND_LO, BAND_HI, color="0.6", alpha=0.13, lw=0, zorder=0)
    for name in REGIMES:
        t, y = sols[name]
        ax.plot(t, y[:, i], "-", lw=2.6, color=COLORS[name], label=name,
                zorder=3)
    ax.set_xlim(0, T)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Dimensionless Concentration")
    ax.grid(axis="y", color="0.9", lw=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

handles = [plt.Line2D([], [], color=COLORS[n], lw=3.5) for n in REGIMES]
handles.append(plt.Rectangle((0, 0), 1, 1, color="0.6", alpha=0.13))
labels = list(REGIMES) + ["ATRA Window"]
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 0.945), fontsize=14, handlelength=2.2,
           columnspacing=2.0)
fig.suptitle("Core WNT-HOX Regulatory Dynamics", fontsize=24,
             fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.90])

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "scipy_core_dynamics.png")
fig.savefig(out, dpi=170, facecolor="white")
plt.close(fig)
print(f"Saved {out}")
