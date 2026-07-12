"""Cross-regime aggregator for the Bayesian inverse PINN.

Reads the 4 per-regime posterior sample files in a run dir and produces:
  1. bayes_W_thetaP.png  — the HEADLINE figure: W and thetaP marginals overlaid
     across the 4 regimes, so the thetaP posterior visibly collapses toward the
     prior as WNT drive increases (Normal -> Strong APC-mutant).
  2. bayes_identifiability.png — IDENT/WEAK/NON-IDENT count per regime + the
     shrink-vs-prior heatmap over all 36 params x 4 regimes.
  3. bayes_summary.md — a compact table (the profile-likelihood / FIM analogue,
     now in posterior-width currency).
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIME_ORDER = ["Normal", "Early_Adenoma", "Advanced_Adenoma", "Severe_APC_Loss"]
REGIME_LABEL = {"Normal": "Normal", "Early_Adenoma": "Early Adenoma",
                "Advanced_Adenoma": "Advanced Adenoma",
                "Severe_APC_Loss": "Severe APC Loss"}
COLORS = {"Normal": "#2ca02c", "Early_Adenoma": "#1f77b4",
          "Advanced_Adenoma": "#ff7f0e", "Severe_APC_Loss": "#d62728"}


def load_run(run_dir):
    data = {}
    for safe in REGIME_ORDER:
        npz = os.path.join(run_dir, f"{safe}_posterior_samples.npz")
        js = os.path.join(run_dir, f"{safe}_posterior_summary.json")
        if not (os.path.exists(npz) and os.path.exists(js)):
            print(f"  [skip] {safe}: outputs missing")
            continue
        z = np.load(npz, allow_pickle=True)
        with open(js) as fh:
            summ = json.load(fh)
        data[safe] = dict(values=z["values"], names=list(z["names"]),
                          true=z["true"], summary=summ["params"],
                          meta=summ["meta"])
    return data


def plot_W_thetaP(data, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pname in zip(axes, ["W", "thetaP"]):
        for safe in REGIME_ORDER:
            if safe not in data:
                continue
            d = data[safe]
            i = d["names"].index(pname)
            vals = d["values"][:, i]
            tv = d["true"][i]
            c = COLORS[safe]
            ax.hist(vals, bins=50, density=True, histtype="stepfilled",
                    alpha=0.35, color=c)
            ax.hist(vals, bins=50, density=True, histtype="step",
                    lw=2, color=c, label=f"{REGIME_LABEL[safe]}")
            ax.axvline(tv, color=c, ls=":", lw=1.5)
        ax.set_title(f"Posterior marginal of {pname}\n"
                     "(dotted = each regime's true value)")
        ax.set_xlabel(pname)
        ax.set_ylabel("posterior density")
        ax.legend(fontsize=9)
    fig.suptitle("Bayesian inverse PINN — W stays identifiable; thetaP "
                 "reverts to prior as WNT drive rises", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(run_dir, "bayes_W_thetaP.png"), dpi=130)
    plt.close(fig)


def plot_identifiability(data, run_dir):
    names = None
    for safe in REGIME_ORDER:
        if safe in data:
            names = data[safe]["names"]
            break
    if names is None:
        return
    regimes = [s for s in REGIME_ORDER if s in data]

    # shrink-vs-prior heatmap (0 = pinned/identifiable, 1 = reverts to prior)
    H = np.full((len(names), len(regimes)), np.nan)
    for j, safe in enumerate(regimes):
        summ = data[safe]["summary"]
        for i, k in enumerate(names):
            H[i, j] = summ[k]["shrink_vs_prior"]

    fig, (a0, a1) = plt.subplots(
        1, 2, figsize=(15, 12), gridspec_kw={"width_ratios": [1.1, 2.4]})

    # left: stacked verdict counts per regime
    cats = ["IDENT", "WEAK", "NON-IDENT"]
    ccol = {"IDENT": "#2ca02c", "WEAK": "#ff7f0e", "NON-IDENT": "#d62728"}
    counts = {c: [] for c in cats}
    for safe in regimes:
        summ = data[safe]["summary"]
        for c in cats:
            counts[c].append(sum(1 for k in names
                                 if summ[k]["verdict"] == c))
    bottom = np.zeros(len(regimes))
    x = np.arange(len(regimes))
    for c in cats:
        a0.bar(x, counts[c], bottom=bottom, color=ccol[c], label=c)
        bottom += np.array(counts[c])
    a0.set_xticks(x)
    a0.set_xticklabels([REGIME_LABEL[s] for s in regimes], rotation=30,
                       ha="right", fontsize=9)
    a0.set_ylabel("# of 36 parameters")
    a0.set_title("Identifiability verdict by regime")
    a0.legend(fontsize=9)

    # right: heatmap
    im = a1.imshow(H, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    a1.set_yticks(np.arange(len(names)))
    a1.set_yticklabels(names, fontsize=7)
    a1.set_xticks(np.arange(len(regimes)))
    a1.set_xticklabels([REGIME_LABEL[s] for s in regimes], rotation=30,
                       ha="right", fontsize=9)
    a1.set_title("posterior shrink vs prior  "
                 "(0 = identifiable, 1 = reverts to prior)")
    for i in range(len(names)):
        for j in range(len(regimes)):
            a1.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=a1, fraction=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "bayes_identifiability.png"), dpi=120)
    plt.close(fig)


def write_summary_md(data, run_dir):
    regimes = [s for s in REGIME_ORDER if s in data]
    lines = ["# Bayesian inverse PINN — posterior identifiability\n"]
    lines.append("Tier-1 B-PINN: HMC over the 36 parameters with the "
                 "pinn-boost state nets frozen and the derivative-free "
                 "integral residual as the physics likelihood.\n")
    # convergence/honesty gate: if a regime's chain did not mix (ESS) or the
    # posterior is not calibrated (covers truth), its IDENT/NON-IDENT counts are
    # meaningless — flag it loudly at the top so the headline can't mislead.
    any_fail = any(not data[s]["meta"].get("gate_pass", False) for s in regimes)
    if any_fail:
        lines.append("> **⚠ GATE FAILED for one or more regimes — the "
                     "IDENT/NON-IDENT counts below are NOT trustworthy.** "
                     "A chain needs ESS(median) ≥ 200 and CI coverage ≥ 33/36 "
                     "before its verdicts mean anything.\n")
    # per-regime headline
    lines.append("## Per-regime summary\n")
    lines.append("| regime | gate | ESS med | covers | accept | IDENT | WEAK | "
                 "NON-IDENT | W post | thetaP post (true) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for safe in regimes:
        summ = data[safe]["summary"]
        meta = data[safe]["meta"]
        nid = sum(1 for k in summ if summ[k]["verdict"] == "IDENT")
        nw = sum(1 for k in summ if summ[k]["verdict"] == "WEAK")
        nn = sum(1 for k in summ if summ[k]["verdict"] == "NON-IDENT")
        W, th = summ["W"], summ["thetaP"]
        gate = "✅PASS" if meta.get("gate_pass", False) else "❌FAIL"
        essm = meta.get("ess_median", float("nan"))
        cov = meta.get("covers_true", "?")
        lines.append(
            f"| {REGIME_LABEL[safe]} | {gate} | {essm:.0f} | {cov}/36 | "
            f"{meta['accept']:.2f} | {nid} | {nw} | "
            f"{nn} | {W['post_mean']:.3f}±{W['post_std']:.3f} | "
            f"{th['post_mean']:.3f}±{th['post_std']:.3f} "
            f"({th['true']:.2f}) |")
    # thetaP / W detail
    lines.append("\n## W and thetaP across regimes\n")
    lines.append("| regime | W true | W post (CI95) | shrink | "
                 "thetaP true | thetaP post (CI95) | shrink | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for safe in regimes:
        summ = data[safe]["summary"]
        W, th = summ["W"], summ["thetaP"]
        lines.append(
            f"| {REGIME_LABEL[safe]} | {W['true']:.2f} | "
            f"{W['post_mean']:.3f} "
            f"[{W['ci95'][0]:.2f},{W['ci95'][1]:.2f}] | "
            f"{W['shrink_vs_prior']:.2f} | {th['true']:.2f} | "
            f"{th['post_mean']:.3f} "
            f"[{th['ci95'][0]:.2f},{th['ci95'][1]:.2f}] | "
            f"{th['shrink_vs_prior']:.2f} | {th['verdict']} |")
    with open(os.path.join(run_dir, "bayes_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    data = load_run(args.run)
    if not data:
        print("no regime outputs found in", args.run)
        return
    plot_W_thetaP(data, args.run)
    plot_identifiability(data, args.run)
    write_summary_md(data, args.run)
    print(f"\naggregate figures + bayes_summary.md written to {args.run}")


if __name__ == "__main__":
    main()
