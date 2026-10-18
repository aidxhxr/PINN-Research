"""Which edges' anchors can ANY available protocol reach? (no training)

`anchor_report.py` says whether an edge's `f(0)=0` anchor is visited by the
condition set you happen to have. This asks the design question instead: over a
ladder of depletion protocols, **what is the cheapest one that puts each edge's
anchor into the data**, and which anchors no protocol reaches at all.

The 2026-08-01 A/B established that a learned term's basal-production cost
collapses when its regulator reaches zero and does not when it merely
approaches. That makes this table a prediction sheet: an edge whose anchor a
protocol reaches should learn cleanly under it, and an edge no protocol reaches
should keep paying 10-80% of its basal parameter no matter what is run.

The ladder is ordered by how invasive the experiment is, because the cheapest
protocol that works is the answer an experimentalist wants:

  raKO      retinoid-free medium                      (a diet/medium change)
  wntKO     WNT ligand/pathway knockdown              (one siRNA)
  bcatKO    + knock down the HOXA13 feedback arm      (two siRNAs)
  hox5KO    raKO + silence HOXA5 basal production     (siRNA + promoter edit)
  mycKO     bcatKO + silence MYC basal production     (2 siRNA + promoter edit)
  hox13KO   bcatKO + mycKO + silence HOXA13 basal     (3 siRNA + 2 edits)
  cypKO     raKO + bcatKO + silence CYP26A1 basal     (3 siRNA + 1 edit)

Knocking down a *signalling* input (`kW`, retinoid) is an ordinary siRNA or
medium change. Knocking down a *basal production* term (`kaM`, `ka13`, `kaC`,
`ka5`) means editing that gene's constitutive promoter, which is a materially
harder experiment -- so the ladder is not a free menu, and the last four rows
should only be proposed for edges that need them.

usage:  python3 anchor_reach.py
"""
from __future__ import annotations

import numpy as np

from anchor_doses import anchor_ratio, solve_conditions
from config import CONDITIONS, HYBRID_TERMS, REGIMES

# name -> forcing of the ONE extra condition appended to the base ten.
# Ordered cheapest-first; `none` is the existing 10-condition set.
PROTOCOLS = [
    ("none",    None),
    ("raKO",    {"mu0": 0.0, "AR": 0.0, "DR": 0.0}),
    ("wntKO",   {"kW": 0.0}),
    ("bcatKO",  {"kW": 0.0, "k13": 0.0}),
    ("hox5KO",  {"mu0": 0.0, "AR": 0.0, "DR": 0.0, "ka5": 0.0}),
    ("mycKO",   {"kW": 0.0, "k13": 0.0, "kaM": 0.0}),
    ("hox13KO", {"kW": 0.0, "k13": 0.0, "kaM": 0.0, "ka13": 0.0}),
    ("cypKO",   {"mu0": 0.0, "AR": 0.0, "DR": 0.0,
                 "kW": 0.0, "k13": 0.0, "kaC": 0.0}),
]
REACHED = 0.01      # the threshold the A/B outcomes separate at

# Which experimenter-set knob scales each basal-production parameter. Needed for
# the SECOND clause of the design rule, which the 2026-08-02 prospective test
# discovered the hard way: reaching the anchor is not sufficient if the very
# condition that reaches it also deletes the basal parameter from the equation.
#
# `h13_b` is the case. Driving h13 to its anchor requires kW = 0 (h13 cannot
# reach zero unless b does first), but h13_b's basal parameter IS W -- so the
# one condition that observes the network at its anchor carries no information
# about the parameter the anchor exists to protect. Measured: 16.2% basal error
# under all three protocols, identical to three significant figures.
#
# Contrast the successful cases: under bcatKO the MYC equation collapses to
# `eps*dm/dt = aM - m`, which pins aM directly.
BASAL_KNOB = {"W": "kW", "a5": "ka5", "a13": "ka13",
              "aM": "kaM", "aC": "kaC"}


def informative_about_basal(forcing, basal):
    """Does a protocol leave the edge's basal parameter observable?

    False when the protocol scales that parameter to zero -- the condition then
    constrains the network at its anchor but says nothing about the constant the
    network could absorb.
    """
    knob = BASAL_KNOB.get(basal)
    if forcing is None or knob is None:
        return True
    return float(forcing.get(knob, 1.0)) > 0.0


def edge_regulators():
    """(edge, regulator, class) for every state-driven term with an anchor."""
    out = []
    for term, spec in HYBRID_TERMS.items():
        if spec.get("input_kind") == "parameter" or spec.get("anchor") is None:
            continue
        if len(spec["inputs"]) != 1:
            continue                      # apc_prod is multivariate; skip
        cls = "modulator" if spec.get("factor") else "production"
        out.append((term, spec["inputs"][0], cls, spec.get("basal")))
    return out


def main():
    regimes = list(REGIMES)
    edges = edge_regulators()
    regs = sorted({var for _t, var, _c, _b in edges})

    # ratios[protocol][regime][var]
    ratios = {}
    for name, forcing in PROTOCOLS:
        conds = CONDITIONS if forcing is None else (
            CONDITIONS + [{"name": name, "forcing": forcing}])
        ratios[name] = {}
        for regime in regimes:
            refs = solve_conditions(regime, conds)
            ratios[name][regime] = {v: anchor_ratio(refs, v)[0] for v in regs}
        print(f"  solved {name}", flush=True)

    print("\n\nanchor ratio x_lo/x_hi per regulator, WORST over the 4 regimes")
    print("(0.000 = the anchor is in the data; the A/B outcomes separate "
          f"at {REACHED})\n")
    print(f"  {'protocol':<10s} " + " ".join(f"{v:>9s}" for v in regs))
    for name, _f in PROTOCOLS:
        worst = [max(ratios[name][r][v] for r in regimes) for v in regs]
        print(f"  {name:<10s} " + " ".join(f"{w:9.4f}" for w in worst))

    print("\n\nCHEAPEST PROTOCOL THAT REACHES EACH EDGE'S ANCHOR\n")
    print(f"  {'edge':<10s} {'class':<11s} {'regulator':<10s} "
          f"{'basal':<7s} {'10-cond':>8s}  {'cheapest that works':<12s} "
          f"{'ratio':>8s}")
    unreachable, blinded = [], []
    for term, var, cls, basal in sorted(edges, key=lambda e: (e[2], e[0])):
        base = max(ratios["none"][r][var] for r in regimes)
        hit = None
        for name, forcing in PROTOCOLS[1:]:
            w = max(ratios[name][r][var] for r in regimes)
            if w < REACHED:
                if not informative_about_basal(forcing, basal):
                    # reaches the anchor but deletes the basal parameter with it
                    blinded.append((term, name, basal))
                    continue
                hit = (name, w)
                break
        if hit is None:
            best = min(((max(ratios[n][r][var] for r in regimes), n)
                        for n, _f in PROTOCOLS[1:]))
            unreachable.append((term, var, cls, best))
            print(f"  {term:<10s} {cls:<11s} {var:<10s} "
                  f"{(basal or '-'):<7s} {base:8.4f}  {'NONE':<12s} "
                  f"{best[0]:8.4f}  (best: {best[1]})")
        else:
            print(f"  {term:<10s} {cls:<11s} {var:<10s} "
                  f"{(basal or '-'):<7s} {base:8.4f}  {hit[0]:<12s} "
                  f"{hit[1]:8.4f}")

    print("\n  Reading it: a `production` edge whose anchor a protocol reaches "
          "should\n  recover its basal parameter to ~1% under that protocol "
          "(measured for\n  ra_h5, rc_cyp, bm_myc). A `modulator` edge has no "
          "basal parameter to\n  lose and is already free -- the anchor column "
          "is informational only.")
    if unreachable:
        print("\n  No USABLE protocol in the ladder (unreached, or reached only\n  by one that deletes the basal parameter — see below):")
        for term, var, cls, (w, n) in unreachable:
            print(f"    {term:<10s} ({cls}, regulator {var}) — best is "
                  f"{n} at {w:.4f}")
    if blinded:
        print("\n  REACHES the anchor but DELETES the basal parameter with it "
              "(rejected — measured\n  to be useless on `h13_b`, 16.2% basal "
              "error under all three protocols):")
        for term, name, basal in blinded:
            print(f"    {term:<10s} {name:<9s} sets "
                  f"{BASAL_KNOB[basal]} = 0, and {basal} is that edge's own "
                  f"basal parameter")


if __name__ == "__main__":
    main()
