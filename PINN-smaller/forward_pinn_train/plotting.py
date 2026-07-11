import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BASELINE, REGIMES, VAR_NAMES, VAR_LABELS
from metrics import stemness

# Active-dynamics window. The model is trained over the full horizon
# (T=3000), but all the interesting behaviour — the ATRA pulse and the
# relaxation back to steady state — lives in roughly the first 150 tau.
# Plotting the whole horizon leaves a long flat tail, so zoom in here.
# (matches XMAX in compare.py)
XMAX = 150.0


def plot_all(solutions, refs, xmax=XMAX):
    for i, (key, label) in enumerate(zip(VAR_NAMES, VAR_LABELS)):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name in REGIMES:
            sol   = solutions[name]
            tr, yr = refs[name]
            ax.plot(sol["t"],  sol[key], lw=2.0, label=f"{name} (PINN)")
            ax.plot(tr, yr[:, i], "--", lw=1.0, alpha=0.6,
                    label=f"{name} (ref)")
        ax.axvspan(BASELINE["tau1"], BASELINE["tau2"],
                   alpha=0.08, color="orange", label="ATRA")
        ax.set_xlim(0, xmax)
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(key)
        ax.set_title(label); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(f"pinn7_{key}.png", dpi=150)
        plt.show(); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in REGIMES:
        sol = solutions[name]
        pc  = {**BASELINE, **REGIMES[name]}
        ax.plot(sol["t"], stemness(sol, pc), lw=2.0, label=name)
    ax.axvspan(BASELINE["tau1"], BASELINE["tau2"],
               alpha=0.08, color="orange", label="ATRA")
    ax.set_xlim(0, xmax)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("Stemness")
    ax.set_title("Stemness index"); ax.legend()
    fig.tight_layout(); fig.savefig("pinn7_stemness.png", dpi=150)
    plt.show(); plt.close(fig)


def plot_losses(all_hist):
    n = len(all_hist)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    for ax, (name, h) in zip(axes[0], all_hist.items()):
        ax.semilogy(h["epoch"], h["loss"], "k-",  label="total")
        ax.semilogy(h["epoch"], h["Ld"],   "b--", label="data")
        ax.semilogy(h["epoch"], h["Lp"],   "r--", label="physics")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(name); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("pinn7_losses.png", dpi=150)
    plt.show(); plt.close(fig)


def print_summary(solutions):
    print("\n" + "="*70)
    print("Final values & stemness")
    print("="*70)
    for name in REGIMES:
        sol = solutions[name]
        pc  = {**BASELINE, **REGIMES[name]}
        S   = stemness(sol, pc)
        t   = sol["t"]
        mask = (t >= pc["tau1"]) & (t <= pc["tau2"])
        print(f"\n{name}")
        for vn, vl in zip(VAR_NAMES, VAR_LABELS):
            print(f"  {vl:<16s} {sol[vn][-1]:.4f}")
        print(f"  {'Stemness':<16s} {S[-1]:.4f}")
        if mask.any():
            print(f"  Min S (ATRA)    {S[mask].min():.4f}")
            print(f"  Mean S (ATRA)   {S[mask].mean():.4f}")
