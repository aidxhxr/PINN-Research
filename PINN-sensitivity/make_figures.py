"""Figures for the SGN-PINN study (palette: SA-big / regime colours).

Usage: python3 make_figures.py <out_dir>
Reads: runs/20261002_mle, runs/20261002_sgn_from_pinn0711, runs/20261002_ablate,
       runs/20261002_multistart, the integral-PINN run, and the repo FIMs.
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench, sgn
from wnt_pinn.model.parameters import UNKNOWN

BLUE, ORANGE, GREY, RED = "#1f77b4", "#ff7f0e", "#bbbbbb", "#cc3333"
REGC = {"Normal": "#1f77b4", "Early Adenoma": "#2ca02c",
        "Advanced Adenoma": "#ff7f0e", "Severe APC Loss": "#d40000"}
REG = list(REGC)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444444", "font.size": 11})
PINN = "../PINN-inverse-pinn-boost/runs/20260711_203325_integral"


def safe(r):
    return r.replace(" ", "_")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def n10(rec, r):
    return int(sum(e < 0.1 for e in bench.rel_errors(rec, r).values()))


def crlb_count(r):
    F = np.load(f"../PINN-fisher-matrix/runs/20260711_203325_fisher/{safe(r)}_FIM.npy") * 150 / 200
    return int((sgn.crlb_sd(F) < 0.05).sum())


def fig_counts(out):
    cols = [("CRLB sd < 5%", GREY), ("Inverse PINN (Adam)", ORANGE),
            ("PINN + SGN", BLUE), ("Classical MLE, best of 4 starts", "#444444")]
    vals = {c: [] for c, _ in cols}
    for r in REG:
        vals[cols[0][0]].append(crlb_count(r))
        vals[cols[1][0]].append(n10(json.load(open(f"{PINN}/{safe(r)}_recovered.json"))["recovered"], r))
        g = load(f"runs/20261002_sgn_from_pinn0711/{safe(r)}_pinn.json")
        best = g["under10"] if g else 0
        for f in glob.glob(f"runs/20261002_multistart/{safe(r)}_*.json"):
            pass  # multistart is reported separately (best-cost selection)
        vals[cols[2][0]].append(best)
        ms = [load(f"runs/20261002_mle/{safe(r)}_{s}.json") for s in ["init", "init_j1", "init_j2", "init_j3"]]
        ms = [m for m in ms if m]
        vals[cols[3][0]].append(min(ms, key=lambda m: m["cost"])["under10"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.2
    for i, (c, col) in enumerate(cols):
        xs = np.arange(4) + (i - 1.5) * w
        ax.bar(xs, vals[c], w, color=col, label=c)
        for xx, v in zip(xs, vals[c]):
            ax.text(xx, v + 0.4, str(v), ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(REG)
    ax.set_ylabel("parameters recovered within 10% (of 36)")
    ax.set_ylim(0, 40); ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "counts.png"), dpi=200); plt.close(fig)
    return vals


def fig_convergence(out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in ["Normal", "Early Adenoma"]:
        g = load(f"runs/20261002_sgn_from_pinn0711/{safe(r)}_pinn.json")
        h = json.load(open(f"{PINN}/{safe(r)}_history.json"))
        pin = [n10({k: h[k][i] for k in UNKNOWN}, r) for i in range(len(h["epoch"]))]
        med = [100 * np.median(list(bench.rel_errors({k: h[k][i] for k in UNKNOWN}, r).values()))
               for i in range(len(h["epoch"]))]
        axes[0].plot(h["epoch"], pin, "--", color=REGC[r], label=f"{r}: PINN Adam")
        axes[1].semilogy(h["epoch"], med, "--", color=REGC[r])
        if g:
            hs = [e for e in g["history"] if e.get("acc", True)]
            it = 2000 + 100 * np.arange(1, len(hs) + 1)
            axes[0].plot(it, [e["under10"] for e in hs], "-", color=REGC[r], label=f"{r}: + SGN steps")
            axes[1].semilogy(it, [e["med"] for e in hs], "-", color=REGC[r])
    for ax in axes:
        ax.axvline(2000, color=GREY, lw=1)
        ax.set_xlabel("Adam epoch  |  then SGN iteration (x100, right of line)")
    axes[0].set_ylabel("params within 10%"); axes[1].set_ylabel("median rel. error (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "convergence.png"), dpi=200); plt.close(fig)


def fig_fisher(out):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for r in ["Normal", "Early Adenoma"]:
        g = load(f"runs/20261002_sgn_from_pinn0711/{safe(r)}_pinn.json")
        m = load(f"runs/20261002_mle/{safe(r)}_init.json")
        if not (g and m):
            continue
        a = sgn.crlb_sd(np.array(g["fisher"]))
        b = np.array([m["sd_x"][k] for k in UNKNOWN])
        ok = (a < 1) & (b < 1)
        ax.loglog(100 * b[ok], 100 * a[ok], "o", ms=4, color=REGC[r], label=r)
    lim = [0.05, 50]
    ax.plot(lim, lim, color=GREY, lw=1)
    ax.set_xlabel("classical MLE: GN sd of log(theta) (%)")
    ax.set_ylabel("SGN profiled-Fisher sd (%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fisher_sd.png"), dpi=200); plt.close(fig)


def fig_ablation(out):
    tags = [("_full", "SGN (full)", BLUE), ("_norff", "no random features", GREY),
            ("_thetaonly", "theta-only GN (heads frozen)", ORANGE),
            ("_nocont", "no lambda continuation", "#888888"),
            ("_adam", "Adam, same objective", RED)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = 0.16
    for i, (t, lab, col) in enumerate(tags):
        v = []
        for r in ["Normal", "Early Adenoma"]:
            g = load(f"runs/20261002_ablate/{safe(r)}_pinn{t}.json")
            v.append(g["under10"] if g else np.nan)
        xs = np.arange(2) + (i - 2) * w
        ax.bar(xs, v, w, color=col, label=lab)
        for xx, vv in zip(xs, v):
            if np.isfinite(vv):
                ax.text(xx, vv + 0.4, str(int(vv)), ha="center", fontsize=8)
    pin = [n10(json.load(open(f"{PINN}/{safe(r)}_recovered.json"))["recovered"], r)
           for r in ["Normal", "Early Adenoma"]]
    for xx, v in zip(range(2), pin):
        ax.hlines(v, xx - 0.45, xx + 0.45, color="k", ls=":", lw=1)
    ax.set_xticks(range(2)); ax.set_xticklabels(["Normal", "Early Adenoma"])
    ax.set_ylabel("params within 10% (of 36)"); ax.set_ylim(0, 40)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout(); fig.savefig(os.path.join(out, "ablation.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "figures"
    os.makedirs(out, exist_ok=True)
    print(fig_counts(out))
    fig_convergence(out)
    fig_fisher(out)
    fig_ablation(out)
    print("wrote", sorted(os.listdir(out)))
