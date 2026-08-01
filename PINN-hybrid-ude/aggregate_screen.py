"""Summarise the equation-local edge screen: the learnability atlas.

Two questions per regulatory edge, kept separate (Loman & Baker,
arXiv:2510.14140):

  A. FUNCTIONAL   -- can a small network recover the relationship it replaced?
                     (NRMSE of f_NN against the closed form, on the visited
                     support)
  B. PARAMETRIC   -- what does hosting that network cost the surviving
                     mechanistic parameters of the SAME equation, measured
                     against a mechanistic control fit of that equation?

and the experimental variable across both is the constraint set:
`gated` (non-negative + soft zero-gate, the 2026-07 baseline) versus `sc`
(monotone + exact anchor).

usage: python3 aggregate_screen.py <screen_dir> [<screen_dir> ...]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import HYBRID_TERMS

ORDER = ["Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss"]
# SA-big palette (standing preference across this repo's figures)
C = {"gated": "#ff7f0e", "sc": "#1f77b4", "sc_bounded": "#2ca02c",
     "ctrl": "#7f7f7f"}
CLASS = {"production": "prod", "modulator": "mod", "ratio": "ratio"}


def term_class(term):
    spec = HYBRID_TERMS[term]
    if len(spec["inputs"]) > 1:
        return "ratio"
    if spec.get("factor"):
        return "modulator"
    return "production"


def load(dirs):
    rows = []
    for d in dirs:
        with open(os.path.join(d, "screen.json")) as fh:
            rows.extend(json.load(fh))
    return rows


def main(dirs):
    rows = load(dirs)
    if not rows:
        print("no screen.json rows")
        return
    params = sorted({r["param"] for r in rows},
                    key=lambda p: ["gated", "sc", "sc_bounded"].index(p)
                    if p in C else 9)
    terms = sorted({r["term"] for r in rows},
                   key=lambda t: (term_class(t), t))

    def get(term, param, regime):
        for r in rows:
            if (r["term"] == term and r["param"] == param
                    and r["regime"] == regime):
                return r
        return None

    # ---------------- table ------------------------------------------------
    print(f"\n{'edge':<10s} {'class':<6s} {'eq':<5s} {'param':<11s} "
          f"{'fNRMSE (N/E/A/S)':>34s} {'mean':>7s} "
          f"{'eq params vs mechanistic control':>34s}")
    summary = {}
    for term in terms:
        for param in params:
            got = [get(term, param, r) for r in ORDER]
            if not any(got):
                continue
            nr = [g["term_nrmse"] if g else np.nan for g in got]
            hy = [g["eq_under10"] if g else 0 for g in got]
            ct = [g["ctrl_eq_under10"] if g else 0 for g in got]
            npar = next(g["n_eq_params"] for g in got if g)
            nctl = next(g["ctrl_n_eq_params"] for g in got if g)
            summary[(term, param)] = dict(
                nrmse=float(np.nanmean(nr)), hybrid=sum(hy), control=sum(ct),
                n=npar * 4, nctl=nctl * 4,
                basal=[g.get("basal_err") if g else None for g in got])
            cells = "/".join(f"{100*v:5.1f}" for v in nr)
            print(f"{term:<10s} {CLASS[term_class(term)]:<6s} "
                  f"{HYBRID_TERMS[term]['eq']:<5s} {param:<11s} "
                  f"{cells:>34s} {100*np.nanmean(nr):6.1f}% "
                  f"{sum(hy):>13d}/{npar*4} vs {sum(ct):>3d}/{nctl*4} "
                  f"({sum(hy)-sum(ct):+d} of the shared {npar*4})")

    # the basal-production compensation, the sharp test of the anchor
    print(f"\nbasal-production parameter error (the constant the network can "
          f"absorb):")
    print(f"{'edge':<10s} {'basal':<7s} " +
          " ".join(f"{p:>10s}" for p in params) + f" {'control':>10s}")
    for term in terms:
        spec = HYBRID_TERMS[term]
        if not spec.get("basal"):
            continue
        cells = []
        for param in params:
            vals = [summary.get((term, param), {}).get("basal") or []]
            v = [x for x in (vals[0] or []) if x is not None]
            cells.append(f"{100*np.mean(v):9.1f}%" if v else f"{'-':>10s}")
        cv = [get(term, params[0], r)["ctrl_basal_err"] for r in ORDER
              if get(term, params[0], r) is not None
              and get(term, params[0], r).get("ctrl_basal_err") is not None]
        ctrl = f"{100*np.mean(cv):9.1f}%" if cv else f"{'-':>10s}"
        print(f"{term:<10s} {spec['basal']:<7s} " + " ".join(cells) +
              f" {ctrl:>10s}")

    # ---------------- figure ----------------------------------------------
    labels = [t for t in terms if (t, params[0]) in summary]
    y = np.arange(len(labels))
    h = 0.8 / max(len(params), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.46 * len(labels) + 3.0),
                             sharey=True)

    ax = axes[0]
    for i, param in enumerate(params):
        vals = [100 * summary[(t, param)]["nrmse"] for t in labels]
        ax.barh(y + (i - (len(params)-1)/2) * h, vals, height=h * 0.86,
                color=C.get(param, "#555"), label=param, zorder=3)
        for yy, v in zip(y + (i - (len(params)-1)/2) * h, vals):
            ax.text(v + 0.6, yy, f"{v:.0f}", va="center", fontsize=7.5,
                    color="#333")
    ax.axvline(10, color=C["ctrl"], lw=1.2, ls="--", zorder=2)
    ax.text(10.4, len(labels) - 0.35, "10%", fontsize=8, color=C["ctrl"])
    ax.set_xlabel("functional NRMSE of the learned term  (%, mean over regimes)")
    ax.set_title("A  Can the edge be learned?", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1]
    for i, param in enumerate(params):
        vals = [summary[(t, param)]["hybrid"] - summary[(t, param)]["control"]
                for t in labels]
        ax.barh(y + (i - (len(params)-1)/2) * h, vals, height=h * 0.86,
                color=C.get(param, "#555"), zorder=3)
        for yy, v in zip(y + (i - (len(params)-1)/2) * h, vals):
            ax.text(v + (0.1 if v >= 0 else -0.1), yy, f"{v:+d}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=7.5, color="#333")
    ax.axvline(0, color=C["ctrl"], lw=1.4, zorder=2)
    ax.set_xlabel("equation parameters within 10%:  hybrid $-$ mechanistic "
                  "control\n(summed over the 4 regimes)")
    ax.set_title("B  What does hosting it cost the mechanism?", loc="left",
                 fontsize=11)

    for a in axes:
        a.set_yticks(y)
        a.set_yticklabels([f"{t}  ({CLASS[term_class(t)]})" for t in labels],
                          fontsize=9)
        a.grid(axis="x", alpha=0.25, lw=0.6, zorder=0)
        a.invert_yaxis()
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.suptitle("Neural-mechanistic edge atlas — every regulatory "
                 "relationship in the WNT-RA-HOX model, learned one at a time",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(dirs[0], "edge_atlas.png")
    fig.savefig(out, dpi=170)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
