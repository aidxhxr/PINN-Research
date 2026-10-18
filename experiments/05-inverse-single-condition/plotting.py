import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (BASELINE, REGIMES, VAR_NAMES, VAR_LABELS,
                    UNKNOWN, INIT_GUESS)
from metrics import stemness

XMAX = 150.0
COLORS = dict(zip(REGIMES, ["tab:blue", "tab:green", "tab:purple", "tab:pink"]))


def _grid(n):
    """Near-square (rows, cols) layout for n subplots (36 -> 6x6)."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def plot_states(solutions, refs, observations, xmax=XMAX):
    """Per-variable: recovered PINN trajectory (solid), scipy reference
    (dashed), and the sparse observations actually fed to the inverse
    solver (scatter)."""
    for i, (key, label) in enumerate(zip(VAR_NAMES, VAR_LABELS)):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name in REGIMES:
            sol = solutions[name]
            tr, yr = refs[name]
            c = COLORS[name]
            ax.plot(sol["t"], sol[key], "-", lw=2.0, color=c,
                    label=f"{name} (PINN)")
            ax.plot(tr, yr[:, i], "--", lw=1.0, color=c, alpha=0.55)
            obs = observations[name]
            ax.scatter(obs["t"], obs["y"][:, i], s=12, color=c,
                       edgecolor="k", linewidth=0.3, zorder=5, alpha=0.7)
        ax.axvspan(BASELINE["tau1"], BASELINE["tau2"],
                   alpha=0.08, color="orange", label="ATRA")
        ax.set_xlim(0, xmax)
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(label)
        ax.set_title(f"{label}  —  PINN (solid) vs ref (dashed), dots=data")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(f"inv_state_{key}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in REGIMES:
        sol = solutions[name]
        pc  = {**BASELINE, **REGIMES[name]}
        ax.plot(sol["t"], stemness(sol, pc), lw=2.0, color=COLORS[name],
                label=name)
    ax.axvspan(BASELINE["tau1"], BASELINE["tau2"],
               alpha=0.08, color="orange", label="ATRA")
    ax.set_xlim(0, xmax)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("Stemness")
    ax.set_title("Stemness index (from recovered solution)")
    ax.legend()
    fig.tight_layout(); fig.savefig("inv_stemness.png", dpi=150)
    plt.close(fig)


def plot_param_convergence(all_hist, true_params):
    """For each unknown parameter: estimate vs iteration for every regime,
    with the true value as a horizontal dashed line of the matching colour
    and the init guess dotted. Wrapped into a near-square grid (36 -> 6x6)
    so it stays readable at full parameter count."""
    n = len(UNKNOWN)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4*cols, 2.6*rows),
                             squeeze=False)
    axflat = axes.ravel()
    for ax, pk in zip(axflat, UNKNOWN):
        for name in REGIMES:
            h = all_hist[name]; c = COLORS[name]
            ax.plot(h["epoch"], h[pk], "-", lw=1.3, color=c, label=name)
            ax.axhline(true_params[name][pk], ls="--", lw=1.0,
                       color=c, alpha=0.7)
        ax.axhline(INIT_GUESS[pk], ls=":", lw=0.8, color="gray")
        ax.set_title(pk, fontsize=9)
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.3)
    for ax in axflat[n:]:
        ax.axis("off")
    handles, labels = axflat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(REGIMES),
               fontsize=9)
    fig.suptitle("Inverse PINN — parameter convergence "
                 "(dashed = true, dotted = init)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig("inv_param_convergence.png", dpi=150)
    plt.close(fig)


def plot_recovery_bars(recovered, true_params):
    """Grouped bar chart (true vs recovered) per unknown, all regimes,
    wrapped into a near-square grid (36 -> 6x6)."""
    n = len(UNKNOWN)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4*cols, 2.6*rows),
                             squeeze=False)
    axflat = axes.ravel()
    names = list(REGIMES)
    x = np.arange(len(names)); w = 0.38
    for ax, pk in zip(axflat, UNKNOWN):
        tru = [true_params[nm][pk] for nm in names]
        rec = [recovered[nm][pk]   for nm in names]
        ax.bar(x - w/2, tru, w, label="true", color="0.6")
        ax.bar(x + w/2, rec, w, label="recovered", color="tab:red",
               alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([nm[:4] for nm in names], fontsize=6)
        ax.set_title(pk, fontsize=9); ax.tick_params(labelsize=6)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axflat[n:]:
        ax.axis("off")
    handles, labels = axflat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Inverse PINN — true vs recovered (all unknowns)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig("inv_recovery_bars.png", dpi=150)
    plt.close(fig)


def plot_losses(all_hist):
    n = len(all_hist)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    for ax, (name, h) in zip(axes[0], all_hist.items()):
        ax.semilogy(h["epoch"], h["loss"], "k-",  label="total")
        ax.semilogy(h["epoch"], h["Ld"],   "b--", label="data")
        ax.semilogy(h["epoch"], h["Lp"],   "r--", label="physics")
        ax.semilogy(h["epoch"], h["Lic"],  "g--", label="IC")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
        ax.set_title(name); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("inv_losses.png", dpi=150)
    plt.close(fig)


def print_summary(recovered, true_params):
    """Param-major table: one row per unknown, relative error (%) per
    regime, sorted ascending by mean error so the least-identifiable
    parameters (the sensitivity-pruning candidates) surface at the
    bottom."""
    names = list(REGIMES)
    print("\n" + "=" * 70)
    print("Recovered parameters (inverse PINN) — rel. error %, by regime")
    print("(sorted best -> worst; worst = sensitivity-pruning candidates)")
    print("=" * 70)
    header = f"{'param':<10s}" + "".join(f"{nm[:8]:>10s}" for nm in names) \
             + f"{'mean':>9s}"
    print(header); print("-" * len(header))

    rows = []
    for pk in UNKNOWN:
        errs = []
        for nm in names:
            tv = true_params[nm][pk]; rv = recovered[nm][pk]
            errs.append(abs(rv - tv) / abs(tv) * 100 if tv != 0
                        else float("nan"))
        rows.append((float(np.nanmean(errs)), pk, errs))

    rows.sort(key=lambda r: r[0])
    for mean_e, pk, errs in rows:
        line = f"{pk:<10s}" + "".join(f"{e:>9.1f}%" for e in errs)
        line += f"{mean_e:>8.1f}%"
        print(line)
