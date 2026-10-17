"""Figure 2: the forward PINN (sparse-data, 40 noise-free observations).

  A  the 7 states for Normal and Severe APC Loss: scipy Radau reference
     (thin ink), the trained PINN (regime colour, dashed) and the 40
     observations the PINN was trained on (open markers).
  B  relative L2 error per state x regime, re-evaluated here on CPU from the
     saved networks; the dense-label (100 evenly spaced labels) baseline is
     shown for comparison. Numbers are cross-checked against each run's
     forward_error_table.json.

The networks are re-evaluated from <run>/<regime>_final.pt with the folder's
own ForwardPINN class (256x4, 16 Fourier features); the reference is the
shared src model (numpy_rhs._ode_rhs) with Radau rtol 1e-10 / atol 1e-12 on
the same 6000-point grid as score_forward.py. Observations are regenerated
exactly as training.sample_sparse does (seed 42, 5000-point reference grid,
t=0 plus 39 random grid points, no noise).

Usage:  PYTHONPATH=<repo>/src python3 make_fig02.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from wnt_pinn.model.numpy_rhs import _ode_rhs  # noqa: E402
from wnt_pinn.model.parameters import BASELINE, REGIMES, INITIAL_STATE  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

T = 150.0
N_EVAL = 6000
SPARSE = "PINN-smaller/forward-pinn-train-hybrid/runs/20260712_204546"
DENSE = "PINN-smaller/forward_pinn_train/runs/20260711_203325"
MODEL_PY = "PINN-smaller/forward-pinn-train-hybrid/model.py"


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        "fwd_model", os.path.join(style.REPO, MODEL_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ForwardPINN


ForwardPINN = _load_model_class()


def reference(regime, n):
    p = {**BASELINE, **REGIMES[regime]}
    t = np.linspace(0.0, T, n)
    sol = solve_ivp(lambda tt, y: _ode_rhs(tt, y, p), (0.0, T), INITIAL_STATE,
                    t_eval=t, method="Radau", rtol=1e-10, atol=1e-12)
    assert sol.success, sol.message
    return t, sol.y.T


def pinn(run, regime, t):
    safe = regime.replace(" ", "_")
    net = ForwardPINN(T_max=T, width=256, depth=4)
    sd = torch.load(os.path.join(style.REPO, run, f"{safe}_final.pt"),
                    map_location="cpu")
    net.load_state_dict(sd)
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(t[:, None])).numpy()


def observations(regime, n_data=40, seed=42):
    """Mirror of training.sample_sparse on the 5000-point reference grid."""
    t_ref, y_ref = reference(regime, 5000)
    rng = np.random.default_rng(seed)
    pool = np.arange(1, len(t_ref))
    pick = np.sort(rng.choice(pool, size=n_data - 1, replace=False))
    idx = np.concatenate([[0], pick])
    return t_ref[idx], y_ref[idx]


def rel_l2(yp, yr):
    return np.linalg.norm(yp - yr, axis=0) / np.linalg.norm(yr, axis=0)


def main():
    audit = {"figure": "fig02_forward",
             "sources": {"sparse_run": SPARSE, "dense_run": DENSE,
                         "model_class": MODEL_PY,
                         "rhs": "src/wnt_pinn/model/numpy_rhs.py::_ode_rhs",
                         "reference": "Radau rtol=1e-10 atol=1e-12, 6000-pt grid on [0,150]",
                         "observations": "regenerated as training.sample_sparse(seed=42, n_data=40, noise 0) on 5000-pt grid"},
             "panel_A": {}, "panel_B": {}, "crosscheck": {}}

    t = np.linspace(0.0, T, N_EVAL)
    err = {"sparse": {}, "dense": {}}
    curves = {}
    for reg in style.REGIMES:
        _, yr = reference(reg, N_EVAL)
        for key, run in (("sparse", SPARSE), ("dense", DENSE)):
            yp = pinn(run, reg, t)
            err[key][reg] = rel_l2(yp, yr)
            if key == "sparse":
                curves[reg] = (yr, yp)

    # cross-check against the on-disk tables
    max_dev = 0.0
    for key, run in (("sparse", SPARSE), ("dense", DENSE)):
        path = os.path.join(style.REPO, run, "forward_error_table.json")
        tab = json.load(open(path))
        cc = {}
        for reg in style.REGIMES:
            disk = np.array(tab["per_regime"][reg]["rel_l2"])
            dev = float(np.max(np.abs(disk - err[key][reg])))
            max_dev = max(max_dev, dev)
            cc[reg] = {"disk_rel_l2": disk.tolist(), "max_abs_dev": dev}
        cc["disk_grand_mean_rel_l2"] = tab["grand_mean_rel_l2"]
        cc["recomputed_grand_mean_rel_l2"] = float(np.mean([err[key][r].mean() for r in style.REGIMES]))
        audit["crosscheck"][key] = {"table": os.path.join(run, "forward_error_table.json"), **cc}
    print(f"max |recomputed - forward_error_table.json| rel L2 = {max_dev:.2e}")

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(style.COL2, 5.15))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.05, 1.25], hspace=0.5,
                     left=0.075, right=0.982, top=0.975, bottom=0.05)
    regsA = ["Normal", "Severe APC Loss"]
    gA = GridSpecFromSubplotSpec(2, 7, subplot_spec=outer[0], wspace=0.55, hspace=0.32)
    for i, reg in enumerate(regsA):
        yr, yp = curves[reg]
        to, yo = observations(reg)
        col = style.REG_COLOR[reg]
        audit["panel_A"][reg] = {"t_obs": to.tolist(),
                                 "y_obs": {s: yo[:, j].tolist() for j, s in enumerate(style.STATES)},
                                 "rel_l2": dict(zip(style.STATES, err["sparse"][reg].tolist()))}
        for j, s in enumerate(style.STATES):
            ax = fig.add_subplot(gA[i, j])
            ax.plot(t, yr[:, j], color=style.INK, lw=0.7, zorder=2)
            ax.plot(t, yp[:, j], color=col, lw=1.3, ls=(0, (3.2, 1.6)), zorder=3)
            ax.plot(to, yo[:, j], ls="none", marker="o", ms=2.6, mfc="white",
                    mec=style.INK2, mew=0.55, zorder=4)
            ax.set_xlim(0, T)
            ax.set_xticks([0, 75, 150])
            ax.tick_params(labelsize=6.3, pad=1.5)
            ax.locator_params(axis="y", nbins=3)
            if i == 0:
                ax.set_title(style.STATE_LABEL[s], fontsize=7.5, fontweight="normal", pad=3)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time $t$", labelpad=1.5)
            if i == 1 and j == 3:
                ax_leg = ax
            if j == 0:
                ax.text(-0.62, 0.5, style.REG_SHORT[reg], transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=7.5,
                        color=col, fontweight="bold")
                if i == 0:
                    fig.text(0.0, 1.0, "A", fontsize=10, fontweight="bold",
                             va="top", ha="left")
    # legend for A (figure level, above the grid)
    hA = [Line2D([], [], color=style.INK, lw=0.7, label="Radau reference"),
          Line2D([], [], color=style.GREY, lw=1.3, ls=(0, (3.2, 1.6)),
                 label="forward PINN (regime colour)"),
          Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec=style.INK2,
                 mew=0.6, label="40 training observations")]
    ax_leg.legend(handles=hA, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3,
                  fontsize=6.8, handlelength=2.4, columnspacing=2.0, borderaxespad=0)

    # ---------------- panel B: error dot plot ----------------
    axB = fig.add_subplot(outer[1])
    cats = style.STATES + ["mean"]
    x0 = np.arange(len(cats), dtype=float)
    x0[-1] += 0.5
    offs = np.linspace(-0.27, 0.27, 4)
    offs_mean = np.linspace(-0.42, 0.42, 4)
    for k, reg in enumerate(style.REGIMES):
        col = style.REG_COLOR[reg]
        es = np.append(err["sparse"][reg], err["sparse"][reg].mean()) * 100
        ed = np.append(err["dense"][reg], err["dense"][reg].mean()) * 100
        xs = x0 + offs[k]
        xs[-1] = x0[-1] + offs_mean[k]
        axB.vlines(xs, ed, es, color=col, lw=0.6, alpha=0.5, zorder=2)
        axB.plot(xs, ed, ls="none", marker="o", ms=3.0, mfc="white", mec=col,
                 mew=0.7, zorder=3)
        axB.plot(xs, es, ls="none", marker="o", ms=3.6, color=col, mec="white",
                 mew=0.3, zorder=4)
        axB.text(xs[-1], es[-1] * 1.13, f"{es[-1]:.2f}", ha="center",
                 va="bottom", fontsize=5.8, color=col, rotation=0)
        audit["panel_B"][reg] = {
            "sparse_rel_l2_pct": dict(zip(cats, es.tolist())),
            "dense_rel_l2_pct": dict(zip(cats, ed.tolist()))}
    gs = float(np.mean([err["sparse"][r].mean() for r in style.REGIMES]) * 100)
    gd = float(np.mean([err["dense"][r].mean() for r in style.REGIMES]) * 100)
    audit["panel_B"]["grand_mean_sparse_pct"] = gs
    audit["panel_B"]["grand_mean_dense_pct"] = gd
    audit["panel_B"]["grand_mean_sparse_no_apc_pct"] = float(np.mean(
        [np.delete(err["sparse"][r], 1).mean() for r in style.REGIMES]) * 100)
    axB.axvline(x0[-2] + 0.72, color=style.LIGHT, lw=0.6)
    axB.set_yscale("log")
    axB.set_ylim(0.15, 25)
    axB.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20])
    axB.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    axB.minorticks_off()
    axB.set_xticks(x0)
    axB.set_xticklabels([style.STATE_LABEL[s] for s in style.STATES] + ["mean"])
    axB.get_xticklabels()[-1].set_fontweight("bold")
    axB.set_xlim(-0.5, x0[-1] + 0.62)
    axB.set_ylabel("relative $L^2$ error (%)")
    axB.grid(axis="x", visible=False)
    axB.text(x0[-1], 20, f"grand mean\n40 obs: {gs:.2f}%\n100 labels: {gd:.2f}%",
             ha="center", va="top", fontsize=6.2, color=style.INK, linespacing=1.15)
    hB = [Line2D([], [], color=style.REG_COLOR[r], lw=1.6, label=style.REG_SHORT[r])
          for r in style.REGIMES]
    hB += [Line2D([], [], ls="none", marker="o", ms=3.6, color=style.GREY,
                  label="40 sparse observations + physics"),
           Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=style.GREY,
                  mew=0.7, label="100 dense labels + physics (baseline)")]
    axB.legend(handles=hB, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=6,
               fontsize=6.6, borderaxespad=0.15, columnspacing=1.3,
               handlelength=1.3, handletextpad=0.4)
    fig.text(0.0, axB.get_position().y1 + 0.045, "B", fontsize=10,
             fontweight="bold", va="bottom", ha="left")

    style.save(fig, "fig02_forward", audit)
    print(f"sparse means: " + " / ".join(
        f"{100 * err['sparse'][r].mean():.2f}" for r in style.REGIMES) + f"  grand {gs:.2f}")
    print(f"dense  means: " + " / ".join(
        f"{100 * err['dense'][r].mean():.2f}" for r in style.REGIMES) + f"  grand {gd:.2f}")


if __name__ == "__main__":
    main()
