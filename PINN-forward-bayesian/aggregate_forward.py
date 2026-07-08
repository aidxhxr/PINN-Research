"""Cross-regime aggregator for the Bayesian FORWARD PINN.

Reads the 4 per-regime predictive files in a run dir and produces:
  1. fwd_beta_bands.png — HEADLINE: the beta-catenin posterior-predictive band
     for all 4 regimes side by side, so the reader sees the band widen / the
     dynamics stiffen as WNT drive rises (Normal -> Strong APC-mutant).
  2. fwd_coverage.png — per-state 95% coverage and relative band width as a
     heatmap over the 7 states x 4 regimes (the forward UQ analogue of the
     inverse identifiability heatmap).
  3. fwd_bayes_summary.md — compact table: accept, ESS, mean coverage, mean
     relative RMSE, mean band width per regime, plus per-state coverage.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIME_ORDER = ["Normal", "Early_adenoma", "Cancer-like", "Strong_APC-mutant"]
REGIME_LABEL = {"Normal": "Normal", "Early_adenoma": "Early adenoma",
                "Cancer-like": "Cancer-like",
                "Strong_APC-mutant": "Strong APC-mutant"}
COLORS = {"Normal": "#2ca02c", "Early_adenoma": "#1f77b4",
          "Cancer-like": "#ff7f0e", "Strong_APC-mutant": "#d62728"}
VAR_NAMES = ["b", "apc", "h5", "h13", "m", "r", "c"]
VAR_LABELS = [r"$\beta$-catenin", "APC", "HOXA5", "HOXA13",
              "MYC", "RA", "CYP26A1"]


def load_run(run_dir):
    data = {}
    for safe in REGIME_ORDER:
        npz = os.path.join(run_dir, f"{safe}_predictive.npz")
        js = os.path.join(run_dir, f"{safe}_predictive_summary.json")
        if not (os.path.exists(npz) and os.path.exists(js)):
            print(f"  [skip] {safe}: outputs missing")
            continue
        z = np.load(npz, allow_pickle=True)
        with open(js) as fh:
            summ = json.load(fh)
        data[safe] = dict(z=z, summary=summ)
    return data


def plot_beta_bands(data, run_dir):
    regimes = [s for s in REGIME_ORDER if s in data]
    fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4.2),
                             sharex=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, safe in zip(axes, regimes):
        z = data[safe]["z"]
        t, mean, lo, hi, ref = z["t"], z["mean"], z["lo"], z["hi"], z["ref"]
        c = COLORS[safe]
        ax.fill_between(t, lo[:, 0], hi[:, 0], color=c, alpha=0.30,
                        label="95% band")
        ax.plot(t, mean[:, 0], color=c, lw=1.6, label="post. mean")
        ax.plot(t, ref[:, 0], color="k", lw=1.2, ls="--", label="truth")
        ax.scatter(z["t_obs"], z["y_obs"][:, 0], s=12, color="k", zorder=5)
        cov = data[safe]["summary"]["per_state"]["b"]["coverage95"]
        ax.set_title(f"{REGIME_LABEL[safe]}   cov95={cov:.2f}", fontsize=11)
        ax.set_xlabel("t")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\beta$-catenin")
    fig.suptitle("Bayesian forward PINN — beta-catenin posterior predictive "
                 "across WNT-drive regimes", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(run_dir, "fwd_beta_bands.png"), dpi=130)
    plt.close(fig)


def plot_coverage(data, run_dir):
    regimes = [s for s in REGIME_ORDER if s in data]
    if not regimes:
        return
    COV = np.full((7, len(regimes)), np.nan)
    WID = np.full((7, len(regimes)), np.nan)
    for j, safe in enumerate(regimes):
        ps = data[safe]["summary"]["per_state"]
        for i, vn in enumerate(VAR_NAMES):
            COV[i, j] = ps[vn]["coverage95"]
            WID[i, j] = ps[vn]["rel_band_width"]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 7))
    for ax, M, title, cmap, vmax in [
            (a0, COV, "95% credible-band coverage\n(1.0 = calibrated)",
             "RdYlGn", 1.0),
            (a1, WID, "relative band width\n(posterior uncertainty)",
             "viridis", None)]:
        im = ax.imshow(M, aspect="auto", cmap=cmap,
                       vmin=0, vmax=vmax if vmax else np.nanmax(M))
        ax.set_yticks(np.arange(7))
        ax.set_yticklabels(VAR_LABELS, fontsize=9)
        ax.set_xticks(np.arange(len(regimes)))
        ax.set_xticklabels([REGIME_LABEL[s] for s in regimes], rotation=30,
                           ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        for i in range(7):
            for j in range(len(regimes)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="black" if cmap == "RdYlGn" else "white")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fwd_coverage.png"), dpi=120)
    plt.close(fig)


def write_summary_md(data, run_dir):
    regimes = [s for s in REGIME_ORDER if s in data]
    lines = ["# Bayesian forward PINN — posterior-predictive UQ\n"]
    lines.append("Forward B-PINN (Yang/Karniadakis 2020): HMC over the network "
                 "WEIGHTS with the ODE parameters KNOWN, given ~sparse noisy "
                 "observations + the derivative-free trapezoidal physics "
                 "residual. Deliverable = a 95% credible band on the "
                 "reconstructed trajectory (UQ on the forward solve).\n")
    lines.append("## Per-regime summary\n")
    lines.append("| regime | accept | ess(U) | ess(pred) | mean cov95 | "
                 "mean relRMSE | mean rel band |")
    lines.append("|---|---|---|---|---|---|---|")
    for safe in regimes:
        s = data[safe]["summary"]
        ov = s["overall"]
        lines.append(
            f"| {REGIME_LABEL[safe]} | {s['accept']:.2f} | {s['ess_U']:.0f} | "
            f"{s['ess_pred_beta']:.0f} | {ov['mean_coverage95']:.2f} | "
            f"{ov['mean_rel_rmse']:.3f} | {ov['mean_rel_band_width']:.3f} |")

    lines.append("\n## Per-state 95% coverage (rows = state, cols = regime)\n")
    header = "| state | " + " | ".join(REGIME_LABEL[s] for s in regimes) + " |"
    lines.append(header)
    lines.append("|---" * (len(regimes) + 1) + "|")
    for vn in VAR_NAMES:
        cells = " | ".join(
            f"{data[s]['summary']['per_state'][vn]['coverage95']:.2f}"
            for s in regimes)
        lines.append(f"| {vn} | {cells} |")

    lines.append("\n_Reading: coverage near 0.95 = the band is calibrated; a "
                 "state whose band is wide (high rel band width) but still "
                 "tracks the truth is where the sparse data + physics leave the "
                 "forward solve least pinned down._\n")
    with open(os.path.join(run_dir, "fwd_bayes_summary.md"), "w") as fh:
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
    plot_beta_bands(data, args.run)
    plot_coverage(data, args.run)
    write_summary_md(data, args.run)
    print(f"\naggregate figures + fwd_bayes_summary.md written to {args.run}")


if __name__ == "__main__":
    main()
