"""Post-hoc analysis of hybrid_fim.py output (no new FIM computations).

  * information-matched vs depletion: S(design)/S(base10) per edge x regime;
  * dose-response: within-arm monotonicity of S in rho, and the "collapse onto
    one curve against rho": the net's information-loss factor
    L = S_net / S_mech (S of the learned-term model relative to the
    mechanistic control on the SAME design) pooled across arms/regimes/edges;
  * robustness of the measured-error correlations to the SVD tolerance rtol.

usage:  python3 hybrid_fim_analysis.py runs/20261017_hybrid_fim
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

D = sys.argv[1]
R = json.load(open(os.path.join(D, "results.json")))["results"]
C = json.load(open(os.path.join(D, "correlations.json")))
idx = {(r["regime"], r["design"], r["edge"]): r for r in R}
out, L = {}, []


def gm(x):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(x))))


# 1. info vs depletion factors
L.append("1. S(design)/S(base10), geometric mean [min, max] over 7 edges x 4 "
         "regimes (S = gated net, rtol 1e-8)")
fac = {}
for fam in ("gated", "sc", "lin_3", "lin_6", "mech"):
    fac[fam] = {}
    for d in ("info", "dep", "depb"):
        v = [idx[(rg, d, e)]["S"][fam] / idx[(rg, "base10", e)]["S"][fam]
             for (rg, dd, e) in idx if dd == "base10"]
        fac[fam][d] = dict(gmean=gm(v), min=float(min(v)), max=float(max(v)))
        L.append(f"   {fam:<6s} {d:<5s} x{gm(v):9.3g}  "
                 f"[{min(v):.3g}, {max(v):.3g}]")
# per-edge for anchor-reaching edges only
L.append("   dep/base10 restricted to cells where dep reaches the anchor "
         "(rho_dep < 1e-3):")
v = [idx[(rg, "dep", e)]["S"]["gated"] / idx[(rg, "base10", e)]["S"]["gated"]
     for (rg, dd, e) in idx if dd == "base10"
     and idx[(rg, "dep", e)]["rho"] < 1e-3]
L.append(f"      n={len(v)}  x{gm(v):.3g} [{min(v):.3g}, {max(v):.3g}]")
out["design_factors"] = fac

# rtol robustness of the factors
L.append("   rtol sensitivity of the gated dep/base10 and info/base10 factors:")
out["rtol_factors"] = {}
for rt in ("1e-06", "1e-08", "1e-10"):
    k = f"gated@{rt}"
    a = [idx[(rg, "info", e)]["S"][k] / idx[(rg, "base10", e)]["S"][k]
         for (rg, dd, e) in idx if dd == "base10"]
    b = [idx[(rg, "depb", e)]["S"][k] / idx[(rg, "base10", e)]["S"][k]
         for (rg, dd, e) in idx if dd == "base10"]
    out["rtol_factors"][rt] = dict(info=gm(a), depb=gm(b))
    L.append(f"      rtol {rt}: info x{gm(a):.3g}  depb x{gm(b):.3g}")

# 2. dose-response
L.append("\n2. dose-response")
dose = [r for r in R if r["design"].startswith("dose_")]
for r in dose:
    r["arm"], r["k"] = r["design"].split("_")[1], float(r["design"].split("_k")[1])
out["dose"] = {}
for fam in ("gated", "sc", "lin_3", "lin_6"):
    res = {}
    for arm in ("ra", "wnt", "bcat"):
        for rg in ("Normal", "Severe APC Loss"):
            rr = sorted([r for r in dose if r["arm"] == arm
                         and r["regime"] == rg], key=lambda r: r["k"])
            s = spearmanr([r["rho"] for r in rr], [r["S"][fam] for r in rr])
            res[f"{arm}/{rg}"] = float(s.statistic)
    # pooled loss factor vs rho, excluding rho == 0 exactly (log scale)
    pts = [(r["rho"], r["S"][fam] / r["S"]["mech"]) for r in dose]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    s_all = spearmanr(x, y)
    pos = x > 0
    slope = np.polyfit(np.log(x[pos]), np.log(y[pos]), 1)
    resid = np.log(y[pos]) - np.polyval(slope, np.log(x[pos]))
    out["dose"][fam] = dict(within_arm_spearman=res,
                            pooled_lossfactor_vs_rho_spearman=float(
                                s_all.statistic),
                            pooled_n=int(len(x)),
                            loglog_slope=float(slope[0]),
                            loglog_resid_sd=float(resid.std(ddof=2)))
    L.append(f"   [{fam}] within-arm Spearman(S, rho): " +
             "  ".join(f"{k}={v:+.2f}" for k, v in res.items()))
    L.append(f"   [{fam}] pooled (3 arms x 2 regimes x 6 doses) Spearman("
             f"S_net/S_mech, rho) = {s_all.statistic:+.2f}; log-log slope "
             f"{slope[0]:.2f}, resid sd {resid.std(ddof=2):.2f} (ln units)")
# matched-ratio comparison (P6 analogue)
L.append("   matched-rho pairs (S_gated/S_mech):")
for rg in ("Normal", "Severe APC Loss"):
    w0 = idx[(rg, "dose_wnt_k0", "bm_myc")]
    for k in ("0.1", "0.03"):
        b = idx[(rg, f"dose_bcat_k{k}", "bm_myc")]
        L.append(f"     {rg:<16s} wnt k=0 rho={w0['rho']:.4f} "
                 f"L={w0['S']['gated']/w0['S']['mech']:.3g}   bcat k={k} "
                 f"rho={b['rho']:.4f} L={b['S']['gated']/b['S']['mech']:.3g}")
    for k in ("0.1", "0.03"):
        a = idx[(rg, f"dose_ra_k{k}", "ra_h5")]
        b = idx[(rg, f"dose_bcat_k{k}", "bm_myc")]
        L.append(f"     {rg:<16s} ra k={k} rho={a['rho']:.4f} "
                 f"L={a['S']['gated']/a['S']['mech']:.3g}   bcat k={k} "
                 f"rho={b['rho']:.4f} L={b['S']['gated']/b['S']['mech']:.3g}")

# 3. rtol robustness of the measured-error correlation
L.append("\n3. Spearman(measured hybrid basal err, S_net) vs rtol "
         "(pooled_dedup / atlas / dose)")
out["corr_rtol"] = {}
pairs = C["pairs"]
dedup = {}
for p in pairs:
    k = (p["edge"], p["regime"], p["design"], p["form"])
    if k not in dedup or p["source"] == "dose5":
        dedup[k] = p
for rt in ("1e-06", "1e-08", "1e-10"):
    row = {}
    for name, pp in (("pooled_dedup", list(dedup.values())),
                     ("atlas", [p for p in pairs if p["source"] == "atlas"]),
                     ("dose", [p for p in pairs if p["source"] == "dose"])):
        s = [idx[(p["regime"], p["design"], p["edge"])]["S"]
             [f"{p['form']}@{rt}"] for p in pp]
        e = [p["hybrid_basal_err"] for p in pp]
        row[name] = float(spearmanr(s, e).statistic)
    out["corr_rtol"][rt] = row
    L.append(f"   rtol {rt}: " + "  ".join(f"{k}={v:+.2f}"
                                             for k, v in row.items()))
# within-base10 only (does the bound rank edges/regimes without a design change?)
b10 = [p for p in dedup.values() if p["design"] == "base10"]
s = spearmanr([p["S_form"] for p in b10], [p["hybrid_basal_err"] for p in b10])
L.append(f"   base10-only cells (no design change), n={len(b10)}: "
         f"Spearman = {s.statistic:+.2f} (p={s.pvalue:.2g})")
out["base10_only_spearman"] = dict(rho=float(s.statistic),
                                   p=float(s.pvalue), n=len(b10))
# excess (hybrid - control) vs log loss factor
ex = [(p["hybrid_basal_err"] - p["ctrl_basal_err"],
       p["S_form"] / p["S_mech"]) for p in dedup.values()
      if p["ctrl_basal_err"] is not None]
s = spearmanr([a for a, _ in ex], [b for _, b in ex])
L.append(f"   excess basal err (hyb - ctrl) vs S_net/S_mech, n={len(ex)}: "
         f"Spearman = {s.statistic:+.2f} (p={s.pvalue:.2g})")
out["excess_vs_lossfactor"] = dict(rho=float(s.statistic), p=float(s.pvalue),
                                   n=len(ex))

txt = "\n".join(L)
print(txt)
with open(os.path.join(D, "analysis.txt"), "w") as fh:
    fh.write(txt + "\n")
with open(os.path.join(D, "analysis.json"), "w") as fh:
    json.dump(out, fh, indent=1)
