import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (BASELINE, REGIMES, CONDITIONS, VAR_NAMES, VAR_LABELS,
                    UNKNOWN, INIT_GUESS)
from metrics import stemness
from odes import _ode_rhs

XMAX = 150.0
COLORS = dict(zip(REGIMES, ["tab:blue", "tab:green", "tab:purple", "tab:pink"]))
CTRL = CONDITIONS[0]["name"]


def _grid(n):
    """Near-square (rows, cols) layout for n subplots (36 -> 6x6)."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def best_params(recovered, true_params, k=8):
    """The k best-recovered (best-converging) unknowns: those with the smallest
    final relative error averaged over the regimes. Used to draw the reduced
    'best-k' companion figures alongside the full 36-param ones."""
    names = list(REGIMES)
    scored = []
    for pk in UNKNOWN:
        errs = []
        for nm in names:
            tv = true_params[nm][pk]; rv = recovered[nm][pk]
            if tv != 0:
                errs.append(abs(rv - tv) / abs(tv))
        if errs:
            scored.append((float(np.mean(errs)), pk))
    scored.sort(key=lambda r: r[0])
    return [pk for _, pk in scored[:k]]


def plot_states(solutions, refs, observations, xmax=XMAX):
    """Per-variable, CONTROL condition: recovered PINN trajectory (solid),
    scipy reference (dashed), and the sparse observations fed to the solver
    (scatter)."""
    for i, (key, label) in enumerate(zip(VAR_NAMES, VAR_LABELS)):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name in REGIMES:
            sol = solutions[name]
            tr, yr = refs[name][CTRL]
            c = COLORS[name]
            ax.plot(sol["t"], sol[key], "-", lw=2.0, color=c,
                    label=f"{name} (PINN)")
            ax.plot(tr, yr[:, i], "--", lw=1.0, color=c, alpha=0.55,
                    label=f"{name} (ref)")
            obs = observations[name]
            ax.scatter(obs["t"], obs["y"][:, i], s=12, color=c,
                       edgecolor="k", linewidth=0.3, zorder=5, alpha=0.7)
        ax.axvspan(BASELINE["tau1"], BASELINE["tau2"],
                   alpha=0.08, color="orange", label="ATRA")
        # proxy handle so the meaning of the scatter markers lives in the legend
        ax.scatter([], [], s=12, color="0.4", edgecolor="k", linewidth=0.3,
                   label=f"{CTRL} condition")
        ax.set_xlim(0, xmax)
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(key)
        ax.set_title(label)
        # legend inside the axes; 'best' picks the least-obstructive spot,
        # except RA where we pin it top-right
        loc = "upper right" if key == "r" else "best"
        ax.legend(fontsize=7, ncol=2, loc=loc, framealpha=0.85)
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


def plot_param_convergence(all_hist, true_params, params=None,
                           fname="inv_param_convergence.png", title_suffix=""):
    params = list(UNKNOWN) if params is None else list(params)
    n = len(params)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4*cols, 2.6*rows),
                             squeeze=False)
    axflat = axes.ravel()
    for ax, pk in zip(axflat, params):
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
                 "(dashed = true, dotted = init)" + title_suffix, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_recovery_bars(recovered, true_params, params=None,
                       fname="inv_recovery_bars.png",
                       suptitle="Inverse PINN — true vs recovered (all unknowns)"):
    params = list(UNKNOWN) if params is None else list(params)
    n = len(params)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4*cols, 2.6*rows),
                             squeeze=False)
    axflat = axes.ravel()
    names = list(REGIMES)
    x = np.arange(len(names)); w = 0.38
    for ax, pk in zip(axflat, params):
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
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_param_error(all_hist, true_params, params=None,
                     fname="inv_param_error.png"):
    """Parameter-recovery error vs training iteration — the headline 'is it
    getting better?' curve. Left: mean & median relative error (%) over all
    unknowns, per regime (log-y, so you literally watch it fall). Right: how
    many of the unknowns are within 5% as training proceeds."""
    params = list(UNKNOWN) if params is None else list(params)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name in REGIMES:
        h = all_hist[name]; c = COLORS[name]
        eps = np.asarray(h["epoch"], dtype=float)
        tv = true_params[name]
        # (n_iter, n_param) relative error in %
        E = np.array([
            [abs(h[pk][i] - tv[pk]) / abs(tv[pk]) * 100.0
             for pk in params if tv[pk] != 0]
            for i in range(len(eps))
        ])
        axL.semilogy(eps, E.mean(axis=1), "-", lw=1.8, color=c,
                     label=f"{name} (mean)")
        axL.semilogy(eps, np.median(E, axis=1), "--", lw=1.0, color=c,
                     alpha=0.7, label=f"{name} (median)")
        axR.plot(eps, (E <= 5.0).sum(axis=1), "-", lw=1.8, color=c,
                 label=name)

    for ax in (axL, axR):
        ax.set_xlabel("Iteration (Adam + L-BFGS)")
        ax.grid(True, alpha=0.3)
    axL.axhline(5.0, ls=":", color="k", alpha=0.6, label="5% target")
    axL.set_ylabel("Relative error  (%)")
    axL.set_title("Parameter recovery error vs iteration")
    axL.legend(fontsize=7, ncol=2)
    axR.set_ylabel("# params within 5%")
    axR.set_ylim(0, len(params))
    axR.set_title(f"Params recovered <5%  (of {len(params)})")
    axR.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=150)
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
    parameters surface at the bottom."""
    names = list(REGIMES)
    print("\n" + "=" * 70)
    print("Recovered parameters (inverse PINN) — rel. error %, by regime")
    print("(sorted best -> worst; worst = least-identifiable)")
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
    n_good = 0
    for mean_e, pk, errs in rows:
        line = f"{pk:<10s}" + "".join(f"{e:>9.1f}%" for e in errs)
        line += f"{mean_e:>8.1f}%"
        print(line)
        if mean_e <= 5.0:
            n_good += 1
    print("-" * len(header))
    print(f"{n_good}/{len(UNKNOWN)} parameters recovered within 5% (mean over regimes)")


def print_sensitivity(refs):
    """Local practical-identifiability diagnostic, computed analytically on
    the scipy reference (no trained net needed). For each unknown we measure
    the relative sensitivity of the ODE right-hand side to a +-1% change in
    that parameter, aggregated over time, all 7 equations, AND all
    experimental conditions. Parameters with tiny sensitivity cannot be
    pinned down by any amount of data — they explain the residual error and
    are the sensitivity-pruning candidates. Higher = more identifiable."""
    from config import REGIMES as _REG
    names = list(_REG)
    print("\n" + "=" * 70)
    print("Practical identifiability — relative RHS sensitivity per parameter")
    print("(aggregated over time, 7 equations, and all conditions; "
          "higher = easier)")
    print("(sorted most -> least identifiable)")
    print("=" * 70)

    # use a coarse time grid for speed
    agg = {pk: [] for pk in UNKNOWN}
    for nm in names:
        p_true = {**BASELINE, **_REG[nm]}
        for c in CONDITIONS:
            tr, yr = refs[nm][c["name"]]
            sel = np.linspace(0, len(tr) - 1, 200).astype(int)
            tsub, ysub = tr[sel], yr[sel]
            pc = {**p_true, **c["forcing"]}
            base = np.array([_ode_rhs(t, y, pc) for t, y in zip(tsub, ysub)])
            scale = np.abs(base).mean() + 1e-9
            for pk in UNKNOWN:
                d = 0.01 * abs(pc[pk]) if pc[pk] != 0 else 0.01
                pp = {**pc, pk: pc[pk] + d}; pm = {**pc, pk: pc[pk] - d}
                fp = np.array([_ode_rhs(t, y, pp) for t, y in zip(tsub, ysub)])
                fm = np.array([_ode_rhs(t, y, pm) for t, y in zip(tsub, ysub)])
                dfdp = (fp - fm) / (2 * d)
                # relative sensitivity: |df/dp| * |p| / typical |f|
                agg[pk].append(np.abs(dfdp).mean() * abs(pc[pk]) / scale)

    ranking = sorted(((float(np.mean(v)), pk) for pk, v in agg.items()),
                     reverse=True)
    print(f"{'param':<10s}{'rel.sensitivity':>18s}")
    print("-" * 28)
    for s, pk in ranking:
        print(f"{pk:<10s}{s:>18.4f}")
    return {pk: s for s, pk in ranking}


# verdict thresholds (mean relative recovery error over regimes, %)
IDENT_GOOD = 5.0     # <= 5%   -> identifiable
IDENT_MARG = 20.0    # 5-20%   -> marginal
# a-priori sensitivity flag: fraction of the most-sensitive parameter
SENS_GOOD = 0.05     # >= 5% of max sensitivity -> sensitive enough
SENS_MARG = 0.01     # 1-5% of max             -> weak


def report_identifiability(recovered, true_params, sens, out_dir="."):
    """Per-parameter identifiability verdict — the 'viable output' that says,
    for EACH of the unknowns, whether it is identifiable (YES / MARGINAL / NO)
    and the numbers behind that call:

      - rel.sens   : analytic local RHS sensitivity (from print_sensitivity),
                     and its share of the most-sensitive parameter (a-priori,
                     data-independent practical identifiability).
      - recov.err% : mean relative recovery error over the 4 regimes (the
                     EMPIRICAL test — did the inverse PINN actually pin it?).
      - verdict    : YES if recov.err <= 5%, MARGINAL if <= 20%, else NO.

    Sorted best -> worst and written to identifiability.csv / .json so it can
    feed the downstream sensitivity-based parameter reduction directly."""
    names = list(REGIMES)
    smax = max(sens.values()) if sens else 1.0

    rows = []
    for pk in UNKNOWN:
        errs = []
        for nm in names:
            tv = true_params[nm][pk]; rv = recovered[nm][pk]
            errs.append(abs(rv - tv) / abs(tv) * 100.0 if tv != 0
                        else float("nan"))
        mean_e = float(np.nanmean(errs))
        s = float(sens.get(pk, 0.0))
        s_rel = s / smax if smax else 0.0
        if mean_e <= IDENT_GOOD:
            verdict = "YES"
        elif mean_e <= IDENT_MARG:
            verdict = "MARGINAL"
        else:
            verdict = "NO"
        if s_rel >= SENS_GOOD:
            apriori = "identifiable"
        elif s_rel >= SENS_MARG:
            apriori = "weak"
        else:
            apriori = "non-ident"
        rows.append(dict(param=pk, sensitivity=s, sens_frac=s_rel,
                         recov_err_pct=mean_e, identifiable=verdict,
                         apriori=apriori,
                         per_regime={nm: e for nm, e in zip(names, errs)}))

    rows.sort(key=lambda r: r["recov_err_pct"])

    print("\n" + "=" * 78)
    print("PARAMETER IDENTIFIABILITY — verdict per unknown "
          "(sorted best -> worst)")
    print("=" * 78)
    header = (f"{'param':<10s}{'rel.sens':>11s}{'%ofmax':>9s}"
              f"{'recov.err':>11s}{'identifiable':>14s}{'a-priori':>14s}")
    print(header); print("-" * len(header))
    n_yes = n_marg = 0
    for r in rows:
        print(f"{r['param']:<10s}{r['sensitivity']:>11.4f}"
              f"{r['sens_frac']*100:>8.1f}%{r['recov_err_pct']:>10.1f}%"
              f"{r['identifiable']:>14s}{r['apriori']:>14s}")
        n_yes += r["identifiable"] == "YES"
        n_marg += r["identifiable"] == "MARGINAL"
    print("-" * len(header))
    print(f"{n_yes} identifiable (<5%) | {n_marg} marginal (5-20%) | "
          f"{len(UNKNOWN)-n_yes-n_marg} not identifiable (>20%)  "
          f"of {len(UNKNOWN)}")

    # ---- file outputs ----
    with open(os.path.join(out_dir, "identifiability.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    with open(os.path.join(out_dir, "identifiability.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["param", "rel_sensitivity", "sens_frac_of_max",
                    "recov_err_pct_mean", "identifiable", "apriori"]
                   + [f"err_{nm}" for nm in names])
        for r in rows:
            w.writerow([r["param"], f"{r['sensitivity']:.6f}",
                        f"{r['sens_frac']:.6f}", f"{r['recov_err_pct']:.4f}",
                        r["identifiable"], r["apriori"]]
                       + [f"{r['per_regime'][nm]:.4f}" for nm in names])
    print(f"wrote {os.path.join(out_dir, 'identifiability.csv')} "
          f"and identifiability.json")
    return rows
