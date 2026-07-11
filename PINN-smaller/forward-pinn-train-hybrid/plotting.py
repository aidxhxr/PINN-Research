import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BASELINE, REGIMES, VAR_NAMES, VAR_LABELS
from metrics import stemness

# Active-dynamics window. The model is trained over the full horizon
# (T=150), and all the interesting behaviour — the ATRA pulse and the
# circadian relaxation — lives here, so plot the whole thing.
XMAX = 150.0


def plot_all(solutions, refs, observations=None, xmax=XMAX):
    """PINN (solid) vs scipy reference (dashed). If `observations` is given
    (the sparse samples the hybrid was trained on), scatter them so it is
    obvious how little data the network actually saw."""
    for i, (key, label) in enumerate(zip(VAR_NAMES, VAR_LABELS)):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name in REGIMES:
            sol   = solutions[name]
            tr, yr = refs[name]
            line, = ax.plot(sol["t"], sol[key], lw=2.0, label=f"{name} (PINN)")
            ax.plot(tr, yr[:, i], "--", lw=1.0, alpha=0.6,
                    color=line.get_color(), label=f"{name} (ref)")
            if observations is not None:
                ob = observations[name]
                ax.scatter(ob["t"], ob[key], s=18, color=line.get_color(),
                           edgecolors="k", linewidths=0.4, zorder=5)
        ax.axvspan(BASELINE["tau1"], BASELINE["tau2"],
                   alpha=0.08, color="orange", label="ATRA")
        ax.set_xlim(0, xmax)
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(label)
        loc = "upper right" if key == "r" else "best"
        ax.set_title(label); ax.legend(fontsize=7, ncol=2, loc=loc)
        fig.tight_layout(); fig.savefig(f"pinn7_{key}.png", dpi=150)
        plt.close(fig)

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
    plt.close(fig)


def plot_losses(all_hist):
    n = len(all_hist)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    for ax, (name, h) in zip(axes[0], all_hist.items()):
        ax.semilogy(h["epoch"], h["loss"], "k-",  label="total")
        ax.semilogy(h["epoch"], h["Ld"],   "b--", label="data")
        ax.semilogy(h["epoch"], h["Lp"],   "r--", label="physics")
        if "Lic" in h:
            ax.semilogy(h["epoch"], h["Lic"], "g:", label="IC")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(name); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("pinn7_losses.png", dpi=150)
    plt.close(fig)


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
