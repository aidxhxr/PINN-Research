"""Parity plot: recovered vs true value for all 36 parameters, integral inverse PINN,
Normal and Severe APC Loss regimes, linear axes with a ±10% wedge. Uses saved recovered values only (data/inverse_recovery.json)."""
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#7f7f7f", "#cc3333"]
REGIMES = [("Normal", "Normal"), ("Early Adenoma", "Early"),
           ("Advanced Adenoma", "Advanced"), ("Severe APC Loss", "Severe APC")]
INK, LIGHT = "#152a3a", "#D3D6DA"


def make_parity(data, out, width=7.65, height=5.5):
    inv = json.loads((data / "inverse_recovery.json").read_text())["integral"]
    shown = [("Normal", "Normal", COLORS[0]), ("Severe APC Loss", "Severe APC", COLORS[3])]
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=.17, right=.97, bottom=.36, top=.96)
    xmax, ymax = 3.8, 5.4
    xs = np.array([0, xmax])
    ax.fill_between(xs, .9*xs, 1.1*xs, color=LIGHT, lw=0, zorder=0)
    ax.plot(xs, xs, color=INK, lw=1.2, zorder=1)
    audit = {}
    for regime, short, color in shown:
        d = inv[regime]
        t = np.array([d["true"][k] for k in d["true"]])
        r = np.array([d["recovered"][k] for k in d["true"]])
        assert len(t) == 36 and (r > 0).all() and r.max() < ymax and t.max() < xmax
        n = int((np.abs(r-t)/np.abs(t) < .1).sum())
        assert n == d["under10"]
        audit[regime] = {"under10": n, "max_rel_err_pct": float(100*np.max(np.abs(r-t)/np.abs(t)))}
        ax.scatter(t, r, s=46, color=color, alpha=.85, linewidths=0, zorder=3)
    ax.set(xlim=(0, xmax), ylim=(0, ymax), xticks=[0, 1, 2, 3], yticks=[0, 1, 2, 3, 4, 5])
    ax.tick_params(labelsize=22, length=4)
    ax.set_xlabel("true value", fontsize=24)
    ax.set_ylabel("recovered value", fontsize=24, labelpad=6)
    ax.grid(False)
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=9, label=s) for _, s, c in shown]
    handles.append(Patch(color=LIGHT, label="±10% margin"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.56, -.012), ncol=3,
               frameon=False, fontsize=20, handletextpad=.3, columnspacing=1.2, handlelength=1.4)
    fig.savefig(out / "parity.pdf")
    fig.savefig(out / "parity.png", dpi=110)
    plt.close(fig)
    (out / "parity_audit.json").write_text(json.dumps(audit, indent=1) + "\n")
