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
import re
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


LOG_RE = re.compile(
    r"^\s+(?P<regime>\S.*?)\s{2,}(?P<term>\S+)\s+(?P<param>\S+)\s+"
    r"fNRMSE=\s*(?P<nrmse>[-\d.]+)%\s+eq params (?P<hy>\d+)/(?P<n>\d+) "
    r"\(ctrl (?P<cn>\d+)/(?P<cnn>\d+)\)\s+basal_err=\s*(?P<be>[-\d.n]+)%?"
    r"(?:\s+\(ctrl\s*(?P<cbe>[-\d.n]+)%\))?")


def load(dirs):
    """Rows from screen.json; falls back to parsing screen.log.

    A 4-regime sweep runs for hours, and runs started before screen.json was
    written incrementally only have the log -- which carries every number in
    the tables below (it lacks x_lo/x_hi, so the anchor-visitation scatter is
    simply skipped for those).
    """
    rows = []
    for d in dirs:
        jf = os.path.join(d, "screen.json")
        if os.path.exists(jf):
            with open(jf) as fh:
                rows.extend(json.load(fh))
            continue
        lf = os.path.join(d, "screen.log")
        if not os.path.exists(lf):
            print(f"[warn] neither screen.json nor screen.log in {d}")
            continue
        print(f"[note] {d}: no screen.json yet -- parsing screen.log. "
              f"Log lines carry only the control's OWN count, so its column "
              f"below sits on the equation's full parameter set while the "
              f"hybrid's sits on the surviving one -- the delta is a lower "
              f"bound, not a like-for-like difference.")
        for line in open(lf):
            m = LOG_RE.match(line.rstrip("\n"))
            if not m:
                continue
            g = m.groupdict()
            row = dict(regime=g["regime"].strip(), term=g["term"],
                       param=g["param"], term_nrmse=float(g["nrmse"]) / 100,
                       eq_under10=int(g["hy"]), n_eq_params=int(g["n"]),
                       ctrl_eq_under10=int(g["cn"]),
                       ctrl_n_eq_params=int(g["cnn"]))
            for key, src in (("basal_err", g["be"]), ("ctrl_basal_err",
                                                      g["cbe"])):
                if src not in (None, "nan"):
                    row[key] = float(src) / 100
            rows.append(row)
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
            # The control fits ALL of the equation's parameters; the hybrid
            # fits only those the network did not absorb. Counting the control
            # over its own larger set would compare different denominators and
            # inflate the apparent hybrid cost, so restrict it to the hybrid's
            # surviving parameters wherever the per-parameter errors are on
            # record (screen.json; log-parsed rows fall back to the raw count).
            ct, nctl = [], next(g["ctrl_n_eq_params"] for g in got if g)
            for g in got:
                if not g:
                    ct.append(0)
                elif "ctrl_param_err" in g and "param_err" in g:
                    shared = set(g["param_err"])
                    ct.append(sum(1 for k, v in g["ctrl_param_err"].items()
                                  if k in shared and v < 0.10))
                    nctl = len(shared)
                else:
                    ct.append(g["ctrl_eq_under10"])
            npar = next(g["n_eq_params"] for g in got if g)
            nreg = sum(1 for g in got if g)      # regimes actually screened
            summary[(term, param)] = dict(
                nrmse=float(np.nanmean(nr)), hybrid=sum(hy), control=sum(ct),
                n=npar * nreg, nctl=nctl * nreg, nreg=nreg,
                basal=[g.get("basal_err") if g else None for g in got])
            cells = "/".join(f"{100*v:5.1f}" for v in nr)
            print(f"{term:<10s} {CLASS[term_class(term)]:<6s} "
                  f"{HYBRID_TERMS[term]['eq']:<5s} {param:<11s} "
                  f"{cells:>34s} {100*np.nanmean(nr):6.1f}% "
                  f"{sum(hy):>13d}/{npar*nreg} vs {sum(ct):>3d}/{nctl*nreg} "
                  f"({sum(hy)-sum(ct):+d})")

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
    # ---- the diagnostic: is the anchor a point the data ever visit? --------
    # x_lo/x_hi is computable BEFORE any training, from reference trajectories
    # alone, so if it predicts the basal-production damage it is a usable
    # pre-flight check rather than a post-hoc explanation.
    pts = [(r["x_lo"] / max(r["x_hi"], 1e-9), r["basal_err"], r["param"],
            r["term"])
           for r in rows if r.get("basal_err") is not None and "x_lo" in r]
    if len(pts) >= 4:
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        with np.errstate(all="ignore"):
            rho = float(np.corrcoef(xs, ys)[0, 1])
        print(f"\nanchor-visitation diagnostic: corr(x_lo/x_hi, basal error) "
              f"= {rho:+.2f} over {len(pts)} fits")
        print(f"  x_lo/x_hi < 0.05 : mean basal err "
              f"{100*np.mean(ys[xs < 0.05]) if (xs < 0.05).any() else float('nan'):.1f}%"
              f"  (n={(xs < 0.05).sum()})")
        print(f"  x_lo/x_hi >= 0.05: mean basal err "
              f"{100*np.mean(ys[xs >= 0.05]) if (xs >= 0.05).any() else float('nan'):.1f}%"
              f"  (n={(xs >= 0.05).sum()})")

    # only edges that have a result under EVERY parameterisation shown, so the
    # grouped bars never compare a full row against a half-finished one
    labels = [t for t in terms if all((t, p) in summary for p in params)]
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
    # y is inverted below, so len(labels)-0.4 is the BOTTOM of the panel --
    # keeps this label clear of the panel title
    ax.text(10.5, len(labels) - 0.4, "10%", fontsize=8, color=C["ctrl"],
            va="bottom")
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
    nreg = max(summary[k]["nreg"] for k in summary)
    ax.set_xlabel("equation parameters within 10%:  hybrid $-$ mechanistic "
                  f"control\n(summed over {nreg} regime"
                  f"{'s' if nreg > 1 else ''})")
    ax.set_title("B  What does hosting it cost the mechanism?", loc="left",
                 fontsize=11)

    for a in axes:
        a.set_yticks(y)
        a.set_yticklabels([f"{t}  ({CLASS[term_class(t)]})" for t in labels],
                          fontsize=9)
        a.grid(axis="x", alpha=0.25, lw=0.6, zorder=0)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    # ONCE, not per-axis: sharey=True means inverting both cancels out and the
    # rows come back in bottom-up order with the 10% label thrown to the top.
    axes[0].invert_yaxis()

    fig.suptitle("Neural-mechanistic edge atlas — every regulatory "
                 "relationship in the WNT-RA-HOX model, learned one at a time",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(dirs[0], "edge_atlas.png")
    fig.savefig(out, dpi=170)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
