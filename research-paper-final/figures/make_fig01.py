"""Figure 1 (panels B, C): reference dynamics and the ten-condition design.

Panel A (the WNT-RA-HOX network schematic) is placed separately by LaTeX from
network-diagram/schematic-better.pdf; this script draws only B and C.

  B  scipy Radau reference (rtol 1e-10, atol 1e-12) in the control condition
     `ctrl` for the four regimes: beta-catenin b, APC and the stemness index
     S = b (1 + h13) / (1 + apc + h5).
  C  the ten experimental conditions as a design ribbon: per condition the
     RA drive mu_R(t), the additive WNT input and the additive MYC input,
     each against the control protocol (grey).

Usage:  PYTHONPATH=<repo>/src python3 make_fig01.py
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from wnt_pinn.model.numpy_rhs import _ode_rhs, _ra_input, _wnt_input, _myc_input  # noqa: E402
from wnt_pinn.model.parameters import (BASELINE, REGIMES, CONDITIONS,  # noqa: E402
                                       INITIAL_STATE)

T = 150.0
N = 1501
SRC_RHS = "src/wnt_pinn/model/numpy_rhs.py::_ode_rhs"
SRC_PAR = "src/wnt_pinn/model/parameters.py (BASELINE, REGIMES, CONDITIONS, INITIAL_STATE)"


def reference(regime, forcing=None):
    p = {**BASELINE, **REGIMES[regime], **(forcing or {})}
    t = np.linspace(0.0, T, N)
    sol = solve_ivp(lambda tt, y: _ode_rhs(tt, y, p), (0.0, T), INITIAL_STATE,
                    t_eval=t, method="Radau", rtol=1e-10, atol=1e-12)
    assert sol.success, sol.message
    return t, sol.y


def amp_label(cond, j):
    """Short amplitude label for the input that a condition perturbs."""
    f = cond["forcing"]
    if j == 0:
        if cond["name"] == "ctrl":
            return "$D_R$ 1.5"
        if "AR" in f:
            return f"$A_R$ {f['AR']:g}\n$D_R$ {f.get('DR', 1.5):g}"
        if "DR" in f:
            return f"$D_R$ {f['DR']:g}"
        return ""
    key = "DW" if j == 1 else "DM"
    if key in f and f[key] != 0:
        return f"{f[key]:+.1f}".replace("-", "\u2212")
    return ""


def main():
    audit = {"figure": "fig01_design",
             "sources": {"rhs": SRC_RHS, "parameters": SRC_PAR,
                         "solver": "scipy.integrate.solve_ivp Radau rtol=1e-10 atol=1e-12",
                         "grid": f"t in [0,{T:.0f}], {N} points"},
             "panel_B": {}, "panel_C": {}}

    # content is kept inside the 7.0 in canvas so the tight bbox stays <= 7.0 in
    fig = plt.figure(figsize=(style.COL2, 5.4))

    # ---------------- panel B: reference dynamics ----------------
    gB = GridSpec(1, 3, figure=fig, left=0.066, right=0.978, top=0.955, bottom=0.675,
                  wspace=0.3)
    axB = [fig.add_subplot(gB[0, i]) for i in range(3)]
    traj = {r: reference(r) for r in style.REGIMES}
    for reg in style.REGIMES:
        t, y = traj[reg]
        b, apc, h5, h13 = y[0], y[1], y[2], y[3]
        S = b * (1 + h13) / (1 + apc + h5)
        col = style.REG_COLOR[reg]
        for ax, v in zip(axB, (b, apc, S)):
            ax.plot(t, v, color=col, lw=1.2)
        audit["panel_B"][reg] = {
            "W": REGIMES[reg]["W"], "thetaP": REGIMES[reg]["thetaP"],
            "b_t150": b[-1], "apc_t150": apc[-1], "S_t150": S[-1],
            "b_max": b.max(), "S_max": S.max(),
            "S_mean_in_ATRA_window_40_88": S[(t >= 40) & (t <= 88)].mean(),
            "S_mean_pre_ATRA_20_40": S[(t >= 20) & (t < 40)].mean(),
        }
    titles = [r"$\beta$-catenin  $b$", r"APC  $p$",
              r"Stemness  $S = b\,(1+h_{13})\,/\,(1+p+h_5)$"]
    for ax, ttl in zip(axB, titles):
        ax.axvspan(BASELINE["tau1"], BASELINE["tau2"], color="#f1f1f1", lw=0, zorder=0)
        ax.set_xlim(0, T)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_xlabel("time $t$ (dimensionless)")
        ax.set_title(ttl, loc="left", fontsize=7.5, fontweight="normal", pad=3)
        ax.set_ylim(bottom=0)
    axB[0].text(0.5 * (BASELINE["tau1"] + BASELINE["tau2"]), 0.03, "ATRA",
                transform=axB[0].get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.3, color=style.INK2)
    style.regime_legend(axB[1], loc="upper right", ncol=1, fontsize=6.5,
                        handlelength=1.4, borderaxespad=0.2, labelspacing=0.3)
    fig.text(0.0, 0.995, "B", fontsize=10, fontweight="bold", va="top", ha="left")
    audit["panel_B"]["shaded_ATRA_window"] = [BASELINE["tau1"], BASELINE["tau2"]]
    audit["panel_B"]["stemness_definition"] = "S = b(1+h13)/(1+apc+h5)"
    audit["panel_B"]["condition"] = "ctrl"

    # ---------------- panel C: design ribbon ----------------
    conds = CONDITIONS
    groups = [("Retinoid-only", conds[:6]), ("WNT / MYC perturbing", conds[6:])]
    nrow = len(conds)
    gC = GridSpec(nrow + 1, 3, figure=fig, left=0.175, right=0.93, top=0.535,
                  bottom=0.065, wspace=0.32, hspace=0.22, height_ratios=[1] * 6 + [0.45] + [1] * 4)
    t = np.linspace(0.0, T, 1501)
    ctrl_p = {**BASELINE, **REGIMES["Normal"]}
    inputs = [("RA drive  $\\mu_R(t)$", _ra_input, (0.0, 3.0)),
              ("WNT input  $\\Delta_W(t)$", _wnt_input, (-0.9, 1.15)),
              ("MYC input  $\\Delta_M(t)$", _myc_input, (-0.9, 1.4))]
    colours = ["#2d7c73", "#315b7d", "#9c2745"]  # paper deck Teal / Cobalt / Garnet
    ctrl_curves = [np.array([f(tt, ctrl_p) for tt in t]) for _, f, _ in inputs]

    rows = list(range(6)) + list(range(7, 11))
    first_ax = None
    col0 = []
    for k, (cond, gr) in enumerate(zip(conds, rows)):
        p = {**ctrl_p, **cond["forcing"]}
        rec = {"forcing_overrides": cond["forcing"]}
        for j, (lab, f, ylim) in enumerate(inputs):
            ax = fig.add_subplot(gC[gr, j])
            if first_ax is None:
                first_ax = ax
            y = np.array([f(tt, p) for tt in t])
            base = 0.0
            changed = bool(np.max(np.abs(y - ctrl_curves[j])) > 1e-9)
            active = changed or (k == 0 and j == 0)
            c_line = colours[j] if active else style.LIGHT
            if j == 0 and k > 0:
                ax.plot(t, ctrl_curves[j], color=style.LIGHT, lw=0.8, zorder=1)
            if active:
                ax.fill_between(t, base, y, color=colours[j], alpha=0.18, lw=0, zorder=2)
            ax.plot(t, y, color=c_line, lw=0.9, zorder=3)
            lab_txt = amp_label(cond, j)
            if lab_txt:
                ax.text(1.015, 0.08, lab_txt, transform=ax.transAxes, ha="left",
                        va="bottom", linespacing=1.2, fontsize=6.2, color=colours[j], clip_on=False)
            ax.axhline(0, color=style.LIGHT, lw=0.5, zorder=1)
            ax.set_xlim(0, T)
            ax.set_ylim(*ylim)
            ax.grid(False)
            for s in ("left", "top", "right"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_yticks([])
            last = (k == nrow - 1)
            if last:
                ax.spines["bottom"].set_visible(True)
                ax.set_xticks([0, 50, 100, 150])
                ax.set_xlabel("time $t$")
            else:
                ax.set_xticks([])
            if k == 0:
                ax.set_title(lab, fontsize=7.5, fontweight="normal", pad=3)
            if j == 0:
                col0.append(ax)
                ax.text(-0.03, 0.42, cond["name"], transform=ax.transAxes,
                        ha="right", va="center", fontsize=7)
            key = ["mu_R", "Delta_W", "Delta_M"][j]
            rec[key + "_min"] = float(y.min())
            rec[key + "_max"] = float(y.max())
            rec[key + "_differs_from_ctrl"] = bool(np.max(np.abs(y - ctrl_curves[j])) > 1e-9)
        audit["panel_C"][cond["name"]] = rec
    fig.text(0.0, 0.585, "C", fontsize=10, fontweight="bold", va="top", ha="left")

    # group brackets on the far left
    fig.canvas.draw()
    for gi, (gname, members) in enumerate(groups):
        a0, a1 = (col0[0], col0[5]) if gi == 0 else (col0[6], col0[9])
        y_top = a0.get_position().y1
        y_bot = a1.get_position().y0
        x_br = 0.035
        fig.add_artist(plt.Line2D([x_br, x_br], [y_bot + 0.004, y_top - 0.004],
                                  transform=fig.transFigure, color=style.INK2, lw=0.7))
        fig.text(x_br - 0.013, 0.5 * (y_top + y_bot), f"{gname} ({len(members)})",
                 rotation=90, ha="center", va="center", fontsize=7, color=style.INK)
    audit["panel_C"]["groups"] = {g: [c["name"] for c in m] for g, m in groups}
    audit["panel_C"]["control_curve"] = "grey line = ctrl protocol (DR=1.5, tau 40-88, AR=0.04; DW=DM=0)"
    audit["panel_C"]["ylims"] = {lab: ylim for lab, _, ylim in inputs}

    style.save(fig, "fig01_design", audit)
    print("wrote fig01_design.{pdf,png} + audit")


if __name__ == "__main__":
    main()
