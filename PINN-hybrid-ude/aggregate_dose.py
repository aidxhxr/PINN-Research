"""Score the anchor dose-response against its pre-registered predictions.

Reads every `dose_<arm>_<edge>.json` written by `dose_response.py`, prints the
curves, and scores P1-P6 from `PREREGISTERED.md` with the actual numbers. Runs
on partial output, so a sweep can be read while it is still going.

usage:  python3 aggregate_dose.py runs/20260802_dose_response
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

# SA-big palette (the professor's convention used on every figure in this repo)
BLUE, ORANGE, GREY, RED = "#1f77b4", "#ff7f0e", "#7f7f7f", "#cc3333"
ARM_STYLE = {
    "ra":   (BLUE,   "o", "graded retinoid restriction (RA $\\to$ HOXA5)"),
    "wnt":  (RED,    "s", "WNT knockdown alone ($\\beta$-cat $\\to$ MYC)"),
    "bcat": (ORANGE, "^", "WNT + HOXA13 feedback knockdown "
                          "($\\beta$-cat $\\to$ MYC)"),
}
ARM_ORDER = ["ra", "wnt", "bcat"]


def load(run_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "dose_*.json"))):
        with open(path) as fh:
            rows.extend(json.load(fh))
    return [r for r in rows if r.get("excess_basal_err") is not None]


def spearman(x, y):
    """Rank correlation without a scipy dependency (n is 6; ties are absent
    in practice but averaged if they occur)."""
    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ranks within ties so a plateau does not fake a correlation
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def series(rows, arm, regime):
    s = sorted((r for r in rows if r["arm"] == arm and r["regime"] == regime),
               key=lambda r: -r["dose"])
    return s


def print_tables(rows, regimes):
    for arm in ARM_ORDER:
        for regime in regimes:
            s = series(rows, arm, regime)
            if not s:
                continue
            edge, basal = s[0]["edge"], s[0]["basal_param"]
            print(f"\n  arm {arm!r}  edge {edge}  basal {basal}  "
                  f"regime {regime}")
            print(f"    {'dose':>6s} {'ratio':>8s} {'fNRMSE':>8s} "
                  f"{'hybrid':>8s} {'control':>8s} {'EXCESS':>8s} "
                  f"{'eq params':>12s}")
            for r in s:
                print(f"    {r['dose']:6g} {r['anchor_ratio']:8.4f} "
                      f"{r['term_nrmse']:8.1%} {r['hybrid_basal_err']:8.1%} "
                      f"{r['ctrl_basal_err']:8.1%} "
                      f"{r['excess_basal_err']:8.1%} "
                      f"{r['hybrid_eq_under10']:>5d}/"
                      f"{r['hybrid_n_eq_params']:<6d}")


# The primary statistic is the hybrid's OWN basal error -- the quantity an
# unvisited anchor lets the network absorb a constant out of. The excess over a
# mechanistic control fit on the same data is reported beside it at every test,
# because it can flip sign where the control is itself badly conditioned (which
# is documented for Severe APC Loss). PREREGISTERED.md records this split and
# when it was made.
METRICS = (("hybrid", "hybrid_basal_err"), ("excess", "excess_basal_err"))


def _both(r):
    return {name: r[key] for name, key in METRICS}


def score(rows, regimes):
    print("\n" + "=" * 72)
    print("  PRE-REGISTERED PREDICTIONS  (see PREREGISTERED.md)")
    print("  primary = hybrid basal error;  excess = hybrid - control, "
          "reported alongside")
    print("=" * 72)
    verdicts = {}

    # P1 monotone in achieved anchor ratio, per arm and regime
    print("\n  P1  basal error is monotone in anchor ratio (Spearman >= 0.8)")
    p1 = []
    for arm in ARM_ORDER:
        for regime in regimes:
            s = series(rows, arm, regime)
            if len(s) < 3:
                continue
            x = [r["anchor_ratio"] for r in s]
            rho_h = spearman(x, [r["hybrid_basal_err"] for r in s])
            rho_e = spearman(x, [r["excess_basal_err"] for r in s])
            ok = rho_h >= 0.8
            p1.append(ok)
            print(f"        {arm:<5s} {regime:<18s} rho(hybrid)={rho_h:+.2f}  "
                  f"rho(excess)={rho_e:+.2f}  n={len(s)}  "
                  f"{'PASS' if ok else 'FAIL'}")
    verdicts["P1"] = all(p1) if p1 else None

    # P2 the ra arm collapses below 2% at k=0 and below 3% by k=0.03
    print("\n  P2  `ra` arm: basal error < 2% at k=0, and < 3% by k=0.03")
    p2 = []
    for regime in regimes:
        s = {r["dose"]: r for r in series(rows, "ra", regime)}
        for dose, thresh in ((0.0, 0.02), (0.03, 0.03)):
            if dose not in s:
                continue
            m = _both(s[dose])
            ok = m["hybrid"] < thresh
            p2.append(ok)
            print(f"        {regime:<18s} k={dose:<5g} "
                  f"hybrid={m['hybrid']:6.1%} (excess {m['excess']:+.1%})  "
                  f"vs {thresh:.0%}  {'PASS' if ok else 'FAIL'}")
    verdicts["P2"] = all(p2) if p2 else None

    # P3 the wnt arm never gets below 2% -- the FALSIFIER
    print("\n  P3  `wnt` arm plateaus: basal error never below 2% at any dose")
    print("      (this is the falsifier -- if it collapses, the anchor "
          "account is wrong)")
    p3 = []
    for regime in regimes:
        s = series(rows, "wnt", regime)
        if not s:
            continue
        best = min(s, key=lambda r: r["hybrid_basal_err"])
        m = _both(best)
        ok = m["hybrid"] >= 0.02
        p3.append(ok)
        print(f"        {regime:<18s} best over all doses: k={best['dose']:g} "
              f"hybrid={m['hybrid']:.1%} (excess {m['excess']:+.1%})  "
              f"{'PASS' if ok else 'FAIL (falsified)'}")
    verdicts["P3"] = all(p3) if p3 else None

    # P4 bcat at k=0 breaks the plateau
    print("\n  P4  `bcat` at k=0 (ratio 0) has basal error < 2%, and below "
          "the `wnt` arm's k=0")
    p4 = []
    for regime in regimes:
        b = {r["dose"]: r for r in series(rows, "bcat", regime)}
        w = {r["dose"]: r for r in series(rows, "wnt", regime)}
        if 0.0 not in b or 0.0 not in w:
            continue
        mb, mw = _both(b[0.0]), _both(w[0.0])
        ok = mb["hybrid"] < 0.02 and mb["hybrid"] < mw["hybrid"]
        p4.append(ok)
        print(f"        {regime:<18s} bcat={mb['hybrid']:6.1%} "
              f"(excess {mb['excess']:+.1%})  vs  wnt={mw['hybrid']:6.1%} "
              f"(excess {mw['excess']:+.1%})  {'PASS' if ok else 'FAIL'}")
    verdicts["P4"] = all(p4) if p4 else None

    # P5 the ra and bcat arms collapse onto one curve against ratio
    print("\n  P5  `ra` and `bcat` lie on one curve vs achieved ratio "
          "(different equations, basal params, knockdowns)")
    p5 = []
    for regime in regimes:
        ra = [(r["anchor_ratio"], r["hybrid_basal_err"])
              for r in series(rows, "ra", regime) if r["anchor_ratio"] > 0]
        bc = [(r["anchor_ratio"], r["hybrid_basal_err"])
              for r in series(rows, "bcat", regime) if r["anchor_ratio"] > 0]
        if len(ra) < 3 or len(bc) < 3:
            continue
        # nearest neighbour in log-ratio across arms: where the two arms reach
        # comparable ratios by different means, how far apart are they?
        gaps = []
        for x, e in ra:
            xb, eb = min(bc, key=lambda t: abs(np.log10(t[0]) - np.log10(x)))
            if abs(np.log10(xb) - np.log10(x)) < 0.35:   # within ~2.2x in ratio
                gaps.append(abs(e - eb))
        if gaps:
            med = float(np.median(gaps))
            ok = med < 0.05
            p5.append(ok)
            print(f"        {regime:<18s} median basal-error gap at matched "
                  f"ratio = {med:.1%}  (n={len(gaps)})  "
                  f"{'PASS' if ok else 'FAIL'}")
    verdicts["P5"] = all(p5) if p5 else None

    # P6 wnt k=0 vs bcat k=0.1 -- comparable ratio by very different means
    print("\n  P6  `wnt` k=0 and `bcat` k=0.1 reach comparable ratios by "
          "different means; basal error within ~2.5x")
    p6 = []
    for regime in regimes:
        w = {r["dose"]: r for r in series(rows, "wnt", regime)}
        b = {r["dose"]: r for r in series(rows, "bcat", regime)}
        if 0.0 not in w or 0.1 not in b:
            continue
        ew, eb = w[0.0]["hybrid_basal_err"], b[0.1]["hybrid_basal_err"]
        fold = max(ew, eb) / max(min(ew, eb), 1e-6)
        ok = fold <= 2.5
        p6.append(ok)
        print(f"        {regime:<18s} wnt k=0 (ratio "
              f"{w[0.0]['anchor_ratio']:.4f}) basal={ew:.1%}   "
              f"bcat k=0.1 (ratio {b[0.1]['anchor_ratio']:.4f}) "
              f"basal={eb:.1%}   x{fold:.1f}  {'PASS' if ok else 'FAIL'}")
    verdicts["P6"] = all(p6) if p6 else None

    # Not pre-registered -- an observation the curves suggested while running,
    # reported as exploratory. Across arms the basal error tracks the anchor
    # ratio itself, roughly one-for-one, which would make the ratio not just an
    # ordering variable but a calibrated prediction of the error.
    print("\n  [exploratory, NOT pre-registered]  basal error vs the anchor "
          "ratio itself")
    for regime in regimes:
        pts = [(r["anchor_ratio"], r["hybrid_basal_err"])
               for arm in ARM_ORDER for r in series(rows, arm, regime)
               if r["anchor_ratio"] > 1e-6]
        if len(pts) < 4:
            continue
        folds = [e / x for x, e in pts]
        print(f"        {regime:<18s} basal_err / anchor_ratio: "
              f"median {np.median(folds):.2f}x  "
              f"range {min(folds):.2f}-{max(folds):.2f}  (n={len(pts)})")
    print("        a median near 1.0 would mean the training-free ratio "
          "PREDICTS the error,\n        not merely orders it. Treat as a "
          "hypothesis for the next pre-registration, not a result.")

    print("\n  " + "-" * 68)
    for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
        v = verdicts.get(k)
        print(f"    {k}  " + ("PASS" if v is True else
                              "FAIL" if v is False else "not yet scorable"))
    print("  " + "-" * 68)
    return verdicts


def plot(rows, regimes, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                   # pragma: no cover
        print(f"  (no plot: {exc})")
        return
    n = len(regimes)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 9.0), squeeze=False)
    floor = 3e-4      # so an exactly-zero ratio is drawable on a log axis
    for j, regime in enumerate(regimes):
        ax = axes[0][j]
        for arm in ARM_ORDER:
            s = series(rows, arm, regime)
            if not s:
                continue
            colour, marker, label = ARM_STYLE[arm]
            x = [max(r["anchor_ratio"], floor) for r in s]
            y = [r["hybrid_basal_err"] * 100 for r in s]
            ax.plot(x, y, marker=marker, color=colour, lw=1.8, ms=7,
                    label=label)
            for r, xi, yi in zip(s, x, y):
                if r["dose"] in (1.0, 0.0):
                    ax.annotate(f"k={r['dose']:g}", (xi, yi),
                                textcoords="offset points", xytext=(6, 5),
                                fontsize=8, color=colour)
        # y = x: the error the anchor ratio would predict if the relationship
        # were one-for-one. Exploratory, not a fitted line.
        lim = [floor, 0.2]
        ax.plot(lim, [v * 100 for v in lim], color=GREY, ls="--", lw=1,
                zorder=0, label="basal error $=$ anchor ratio")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("achieved anchor ratio  $x_{lo}/x_{hi}$  "
                      "(0 plotted at the axis floor)")
        ax.set_ylabel("basal-parameter error of the hybrid  (%)")
        ax.set_title(f"{regime} — against what the data reach")
        ax.axhline(2.0, color=GREY, ls=":", lw=1)
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8, loc="upper left")

        ax = axes[1][j]
        for arm in ARM_ORDER:
            s = series(rows, arm, regime)
            if not s:
                continue
            colour, marker, _ = ARM_STYLE[arm]
            ax.plot([max(r["dose"], 3e-3) for r in s],
                    [r["hybrid_basal_err"] * 100 for r in s],
                    marker=marker, color=colour, lw=1.8, ms=7)
        ax.set_xscale("log")
        ax.set_xlabel("knockdown dose $k$  (0 plotted at the axis floor)")
        ax.set_ylabel("basal-parameter error of the hybrid  (%)")
        ax.set_title(f"{regime} — the same data against dose")
        ax.axhline(2.0, color=GREY, ls=":", lw=1)
        ax.grid(alpha=0.25)
    fig.suptitle("UDE basal-parameter bias tracks anchor visitation,\n"
                 "not knockdown depth", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, "dose_response.png")
    fig.savefig(path, dpi=160)
    print(f"\n  wrote {path}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/20260802_dose_response"
    rows = load(run_dir)
    if not rows:
        print(f"no scorable rows in {run_dir} yet")
        return
    regimes = []
    for r in rows:
        if r["regime"] not in regimes:
            regimes.append(r["regime"])
    print(f"\nanchor dose-response — {len(rows)} cells from {run_dir}")
    print_tables(rows, regimes)
    score(rows, regimes)
    plot(rows, regimes, run_dir)


if __name__ == "__main__":
    main()
