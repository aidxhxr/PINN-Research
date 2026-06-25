import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (BASELINE, REGIMES, VAR_NAMES, VAR_LABELS,
                    UNKNOWN, INIT_GUESS)
from metrics import stemness

# Active-dynamics window: the ATRA pulse + relaxation live in the first
# ~150 tau, which is also the full trained horizon here.
XMAX = 150.0
COLORS = dict(zip(REGIMES, ["tab:blue", "tab:green", "tab:purple", "tab:pink"]))


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

    # stemness index
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
    """For each unknown parameter: estimate vs epoch for every regime, with
    the true value drawn as a horizontal dashed line of the matching colour.
    This is the headline inverse-problem result."""
    n = len(UNKNOWN)
    fig, axes = plt.subplots(1, n, figsize=(6.5*n, 4.5), squeeze=False)
    for ax, pk in zip(axes[0], UNKNOWN):
        for name in REGIMES:
            h = all_hist[name]
            c = COLORS[name]
            ax.plot(h["epoch"], h[pk], "-", lw=2, color=c, label=name)
            ax.axhline(true_params[name][pk], ls="--", lw=1.2,
                       color=c, alpha=0.7)
        ax.axhline(INIT_GUESS[pk], ls=":", lw=1.0, color="gray",
                   label="init guess")
        ax.set_xlabel("Iteration (Adam + L-BFGS)")
        ax.set_ylabel(pk)
        ax.set_title(f"Recovering {pk}  (dashed = true)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Inverse PINN — parameter convergence", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("inv_param_convergence.png", dpi=150)
    plt.close(fig)


def plot_recovery_bars(recovered, true_params):
    """Grouped bar chart: true vs recovered for each unknown, all regimes."""
    n = len(UNKNOWN)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4.5), squeeze=False)
    names = list(REGIMES)
    x = np.arange(len(names)); w = 0.38
    for ax, pk in zip(axes[0], UNKNOWN):
        tru = [true_params[nm][pk] for nm in names]
        rec = [recovered[nm][pk]   for nm in names]
        ax.bar(x - w/2, tru, w, label="true", color="0.6")
        ax.bar(x + w/2, rec, w, label="recovered", color="tab:red", alpha=0.85)
        for xi, (tv, rv) in enumerate(zip(tru, rec)):
            err = abs(rv - tv) / abs(tv) * 100
            ax.text(xi + w/2, rv, f"{err:.1f}%", ha="center",
                    va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.set_ylabel(pk); ax.set_title(f"{pk}: true vs recovered")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Inverse PINN — recovered parameters (labels = rel. error)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    print("\n" + "=" * 70)
    print("Recovered parameters (inverse PINN)")
    print("=" * 70)
    header = f"{'regime':<20s}" + "".join(
        f"{pk+' true':>12s}{pk+' rec':>12s}{'err%':>8s}" for pk in UNKNOWN)
    print(header)
    for name in REGIMES:
        row = f"{name:<20s}"
        for pk in UNKNOWN:
            tv = true_params[name][pk]; rv = recovered[name][pk]
            err = abs(rv - tv) / abs(tv) * 100
            row += f"{tv:>12.4f}{rv:>12.4f}{err:>7.2f}%"
        print(row)
