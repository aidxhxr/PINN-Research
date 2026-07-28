"""Summarise a hybrid (UDE) run dir, and -- when given a control run to compare
against -- test the COMPENSATION hypothesis per-parameter.

Scoring rules that keep this honest:

  * Recovery counts are computed on the INTERSECTION of the two runs' recovered
    parameters, so a hybrid (34 unknowns) is never compared against a control
    (36) on a different denominator.
  * H2 predicts the damage is LOCAL to the equation the learned term sits in.
    So deltas are grouped by equation, not just totalled -- a global drop and a
    dh5-localised drop are different results and the total alone hides that.

usage:
    python3 aggregate_hybrid.py <run_dir> [control_run_dir]
"""
import glob
import json
import os
import sys

ORDER = ["Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss"]

# which equation each unknown lives in -- for the localisation test (H2)
EQ = {
    "b":    ["W", "eta13", "kappa13", "lambdaP", "lambda5", "kappa5"],
    "apc":  ["epsP", "rho5", "rhoB", "rho13", "deltaP1", "thetaP"],
    "h5":   ["eps5", "a5", "etaR", "kappaR", "etaM", "kappaM"],
    "h13":  ["eps13", "a13", "etaB13", "kappaB13", "etaM13", "kappaM13"],
    "m":    ["epsM", "aM", "etaBM", "kappaBM"],
    "r":    ["epsR", "lambdaC"],
    "c":    ["epsC", "aC", "etaRC", "kappaRC", "etaBC", "kappaBC"],
}
PARAM_EQ = {p: eq for eq, ps in EQ.items() for p in ps}


def load(run_dir):
    safe2name = {r.replace(" ", "_").replace("/", "_"): r for r in ORDER}
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "*_recovered.json"))):
        base = os.path.basename(f).replace("_recovered.json", "")
        out[safe2name.get(base, base)] = json.load(open(f))
    return out


def rel_err(rec, tru, k):
    if k not in rec or k not in tru or tru[k] == 0:
        return None
    return abs(rec[k] - tru[k]) / abs(tru[k])


def count_under(d, keys, tol=0.10):
    n = 0
    for k in keys:
        e = rel_err(d["recovered"], d["true"], k)
        if e is not None and e < tol:
            n += 1
    return n


def main(run_dir, ctrl_dir=None):
    got = load(run_dir)
    if not got:
        print(f"no *_recovered.json in {run_dir}")
        return
    env = {}
    envf = os.path.join(run_dir, "variant.env")
    if os.path.exists(envf):
        for line in open(envf):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                env[k] = v

    any_d = next(iter(got.values()))
    term = any_d.get("hybrid_term")
    print(f"\nrun: {run_dir}")
    print(f"variant={env.get('variant', '?')}  term={term}  "
          f"constraint={any_d.get('hybrid_constraint')}  "
          f"wd={any_d.get('hybrid_wd')}")

    ctrl = load(ctrl_dir) if ctrl_dir else {}
    if ctrl:
        print(f"control: {ctrl_dir}")

    # ---- recovery table on the shared parameter set ----
    print(f"\n{'regime':<20s} {'under10%':>9s} {'of':>4s}", end="")
    if ctrl:
        print(f" {'control':>9s} {'delta':>6s}", end="")
    print(f" {'termNRMSE':>10s} {'gate_lo':>8s}")

    tot = ctot = 0
    skipped = []
    for r in ORDER:
        if r not in got:
            continue
        if ctrl and r not in ctrl:
            # never total a hybrid regime against a missing control regime --
            # that silently compares 4 regimes against 3
            skipped.append(r)
            continue
        d = got[r]
        keys = set(d["true"])
        if ctrl and r in ctrl:
            keys &= set(ctrl[r]["true"])          # honest shared denominator
        keys = sorted(keys)
        n = count_under(d, keys)
        tot += n
        line = f"{r:<20s} {n:>9d} {len(keys):>4d}"
        if ctrl and r in ctrl:
            cn = count_under(ctrl[r], keys)
            ctot += cn
            line += f" {cn:>9d} {n-cn:>+6d}"
        nr = d.get("term_nrmse")
        line += f" {(f'{nr:.1%}' if nr is not None else '-'):>10s}"
        tf = os.path.join(run_dir, r.replace(" ", "_") + "_term.json")
        if os.path.exists(tf):
            g = json.load(open(tf)).get("gate_lo")
            if g is not None:
                line += f" {g:>8.2f}" + ("  anchor BINDS" if g < 0.5
                                          else "  anchor weak")
        print(line)
    line = f"{'TOTAL':<20s} {tot:>9d} {'':>4s}"
    if ctrl:
        line += f" {ctot:>9d} {tot-ctot:>+6d}"
    print(line)
    if skipped:
        print(f"WARNING: excluded from TOTAL (no matching control regime): "
              f"{', '.join(skipped)}")

    if not ctrl:
        print("\n(pass a control run dir as argv[2] for the H2 localisation test)")
        return

    # ---- H2: is the damage LOCAL to the learned term's equation? ----
    term_eq = {"ra_h5": "h5", "bm_myc": "m", "b_h13": "h13",
               "bc_cyp": "c", "apc_mutation": "apc"}.get(term)
    print(f"\nH2 localisation test -- learned term sits in the d{term_eq} equation.")
    print("Prediction: recovery drops concentrate in that equation; other "
          "equations ~unchanged.\n")
    print(f"{'equation':<10s} {'hybrid':>7s} {'control':>8s} {'delta':>6s}   note")
    for eq in EQ:
        hn = cn = 0
        for r in ORDER:
            if r not in got or r not in ctrl:
                continue
            keys = [k for k in EQ[eq]
                    if k in got[r]["true"] and k in ctrl[r]["true"]]
            hn += count_under(got[r], keys)
            cn += count_under(ctrl[r], keys)
        mark = "  <-- learned term here" if eq == term_eq else ""
        print(f"d{eq:<9s} {hn:>7d} {cn:>8d} {hn-cn:>+6d}{mark}")

    # ---- the specific compensation victim: the basal-production param ----
    victim = {"h5": "a5", "h13": "a13", "m": "aM", "c": "aC"}.get(term_eq)
    if victim:
        print(f"\nH2/H3 sharp test -- '{victim}' is the basal-production term in "
              f"d{term_eq};\na constant offset trades freely between it and the "
              f"learned f_NN unless f(0)=0 is pinned.")
        print(f"\n{'regime':<20s} {'hybrid err':>11s} {'control err':>12s}")
        for r in ORDER:
            if r not in got or r not in ctrl:
                continue
            he = rel_err(got[r]["recovered"], got[r]["true"], victim)
            ce = rel_err(ctrl[r]["recovered"], ctrl[r]["true"], victim)
            fmt = lambda e: f"{e:.1%}" if e is not None else "-"
            print(f"{r:<20s} {fmt(he):>11s} {fmt(ce):>12s}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
