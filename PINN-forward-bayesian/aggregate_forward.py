"""Cross-regime aggregator for the Bayesian FORWARD PINN.

Reads the 4 per-regime predictive files in a run dir and produces:
  1. fwd_beta_bands.png — HEADLINE: the beta-catenin posterior-PREDICTIVE band
     for all 4 regimes side by side, so the reader sees the band widen / the
     dynamics stiffen as WNT drive rises (Normal -> Strong APC-mutant).
  2. fwd_coverage.png — per-state predictive coverage (vs noisy obs) and
     relative predictive-band width as a heatmap over the 7 states x 4 regimes
     (the forward UQ analogue of the inverse identifiability heatmap).
  3. fwd_bayes_summary.md — compact table per regime, plus per-state coverage.

Also regenerates the 4 per-regime {regime}_predictive.png figures, so a stale
run whose .npz predates the predictive-band change is fully replotted from the
stored arrays with no HMC rerun (the wide predictive band = reconstruction (+)
observation noise is reconstructed via bayesian_forward.predictive_band).
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesian_forward import predictive_band, plot_predictive

REGIME_ORDER = ["Normal", "Early_adenoma", "Cancer-like", "Strong_APC-mutant"]
REGIME_LABEL = {"Normal": "Normal", "Early_adenoma": "Early adenoma",
                "Cancer-like": "Cancer-like",
                "Strong_APC-mutant": "Strong APC-mutant"}
COLORS = {"Normal": "#2ca02c", "Early_adenoma": "#1f77b4",
          "Cancer-like": "#ff7f0e", "Strong_APC-mutant": "#d62728"}
VAR_NAMES = ["b", "apc", "h5", "h13", "m", "r", "c"]
VAR_LABELS = [r"$\beta$-catenin", "APC", "HOXA5", "HOXA13",
              "MYC", "RA", "CYP26A1"]


def _augment_pred_metrics(z, summ):
    """Return (pred_lo, pred_hi) and inject predictive metrics into `summ`.

    Uses stored pred_lo/pred_hi when present (new runs); otherwise reconstructs
    them from the epistemic percentiles + observation noise sigma (old runs),
    reading sigma from z['noise_std'] or the summary meta. Adds per-state
    pred_coverage_obs / rel_pred_band_width and the overall means, so both the
    aggregate figures and the per-regime replot see a consistent schema."""
    mean, lo, hi, ref = z["mean"], z["lo"], z["hi"], z["ref"]
    t, t_obs, y_obs = z["t"], z["t_obs"], z["y_obs"]
    if "pred_lo" in z.files and "pred_hi" in z.files:
        pred_lo, pred_hi = z["pred_lo"], z["pred_hi"]
    else:
        sigma = (float(z["noise_std"]) if "noise_std" in z.files
                 else float(summ.get("meta", {}).get("noise_std", 0.0)))
        pred_lo, pred_hi = predictive_band(mean, lo, hi, sigma)

    scale = np.maximum(np.abs(ref).max(axis=0), 0.05)
    ps = summ["per_state"]
    pred_covs, rel_pred_widths = [], []
    for i, vn in enumerate(VAR_NAMES):
        plo_o = np.interp(t_obs, t, pred_lo[:, i])
        phi_o = np.interp(t_obs, t, pred_hi[:, i])
        pc = float(np.mean((y_obs[:, i] >= plo_o) & (y_obs[:, i] <= phi_o)))
        rpw = float(np.mean(pred_hi[:, i] - pred_lo[:, i])) / scale[i]
        ps[vn].setdefault("coverage95_truth", ps[vn].get("coverage95"))
        ps[vn]["pred_coverage_obs"] = pc
        ps[vn]["rel_pred_band_width"] = rpw
        pred_covs.append(pc)
        rel_pred_widths.append(rpw)
    summ["overall"]["mean_pred_coverage_obs"] = float(np.mean(pred_covs))
    summ["overall"]["mean_rel_pred_band_width"] = float(np.mean(rel_pred_widths))
    return pred_lo, pred_hi


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
        pred_lo, pred_hi = _augment_pred_metrics(z, summ)
        data[safe] = dict(z=z, summary=summ, pred_lo=pred_lo, pred_hi=pred_hi)
    return data


def replot_regimes(data, run_dir):
    """Regenerate the per-regime {regime}_predictive.png from stored arrays."""
    for safe in [s for s in REGIME_ORDER if s in data]:
        z = data[safe]["z"]
        pred = dict(t=z["t"], mean=z["mean"], lo=z["lo"], hi=z["hi"],
                    pred_lo=data[safe]["pred_lo"], pred_hi=data[safe]["pred_hi"],
                    ref=z["ref"])
        plot_predictive(pred, z["t_obs"], z["y_obs"],
                        data[safe]["summary"], run_dir, safe)


def plot_beta_bands(data, run_dir):
    regimes = [s for s in REGIME_ORDER if s in data]
    fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4.2),
                             sharex=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, safe in zip(axes, regimes):
        z = data[safe]["z"]
        t, mean, lo, hi, ref = z["t"], z["mean"], z["lo"], z["hi"], z["ref"]
        pred_lo, pred_hi = data[safe]["pred_lo"], data[safe]["pred_hi"]
        c = COLORS[safe]
        ax.fill_between(t, pred_lo[:, 0], pred_hi[:, 0], color=c, alpha=0.25,
                        label="95% predictive band")
        ax.fill_between(t, lo[:, 0], hi[:, 0], color=c, alpha=0.55,
                        label="reconstruction 95%")
        ax.plot(t, mean[:, 0], color=c, lw=1.4, label="post. mean")
        ax.plot(t, ref[:, 0], color="k", lw=1.2, ls="--", label="truth")
        ax.scatter(z["t_obs"], z["y_obs"][:, 0], s=12, color="k", zorder=5)
        cov = data[safe]["summary"]["per_state"]["b"]["pred_coverage_obs"]
        ax.set_title(f"{REGIME_LABEL[safe]}   pred cov(obs)={cov:.2f}",
                     fontsize=11)
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
            COV[i, j] = ps[vn]["pred_coverage_obs"]
            WID[i, j] = ps[vn]["rel_pred_band_width"]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 7))
    for ax, M, title, cmap, vmax in [
            (a0, COV, "95% predictive-band coverage vs obs\n(0.95 = calibrated)",
             "RdYlGn", 1.0),
            (a1, WID, "relative predictive-band width\n(observation uncertainty)",
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
                 "residual. Deliverable = a 95% posterior-predictive band on the "
                 "reconstructed trajectory (reconstruction uncertainty (+) "
                 "observation noise sigma) — UQ on the forward solve.\n")
    lines.append("## Per-regime summary\n")
    lines.append("| regime | accept | ess(U) | ess(pred) | pred cov (obs) | "
                 "mean relRMSE | rel pred band | epi cov (truth) | rel epi band |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for safe in regimes:
        s = data[safe]["summary"]
        ov = s["overall"]
        lines.append(
            f"| {REGIME_LABEL[safe]} | {s['accept']:.2f} | {s['ess_U']:.0f} | "
            f"{s['ess_pred_beta']:.0f} | {ov['mean_pred_coverage_obs']:.2f} | "
            f"{ov['mean_rel_rmse']:.3f} | {ov['mean_rel_pred_band_width']:.3f} | "
            f"{ov['mean_coverage95']:.2f} | {ov['mean_rel_band_width']:.3f} |")

    lines.append("\n## Per-state predictive coverage vs obs (rows = state, "
                 "cols = regime)\n")
    header = "| state | " + " | ".join(REGIME_LABEL[s] for s in regimes) + " |"
    lines.append(header)
    lines.append("|---" * (len(regimes) + 1) + "|")
    for vn in VAR_NAMES:
        cells = " | ".join(
            f"{data[s]['summary']['per_state'][vn]['pred_coverage_obs']:.2f}"
            for s in regimes)
        lines.append(f"| {vn} | {cells} |")

    lines.append("\n_Reading: 'pred cov (obs)' near 0.95 = the predictive band "
                 "(reconstruction (+) noise sigma) is calibrated against the "
                 "held-out noisy observations. 'epi cov (truth)' is the thin "
                 "reconstruction-only band's coverage of the noise-free truth "
                 "(~1.0 when the mean is well-pinned); a large 'rel pred band' "
                 "flags states where the sparse data + physics leave the forward "
                 "solve least pinned down._\n")
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
    replot_regimes(data, args.run)
    plot_beta_bands(data, args.run)
    plot_coverage(data, args.run)
    write_summary_md(data, args.run)
    print(f"\nper-regime + aggregate figures + fwd_bayes_summary.md "
          f"written to {args.run}")


if __name__ == "__main__":
    main()
