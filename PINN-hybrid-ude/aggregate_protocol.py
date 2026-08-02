"""Score the prospective protocol test against its pre-registered predictions.

Reads `protocol_<edge>.json` from `protocol_test.py` and scores Q1-Q5 from
`PREREGISTERED.md`. Runs on partial output.

usage:  python3 aggregate_protocol.py runs/20260802_protocol_test
"""
from __future__ import annotations

import glob
import json
import os
import sys

# baseline / near-miss / prescribed, per edge -- must match run_protocol_test.sh
LADDER = {
    "m_h13": ("none", "bcatKO", "mycKO"),
    "h13_b": ("none", "mycKO", "hox13KO"),
}
# the 10-condition atlas numbers these are measured against
ATLAS = {
    ("m_h13", "Normal"): (0.572, 0.128),
    ("m_h13", "Severe APC Loss"): (0.116, 0.956),
    ("h13_b", "Normal"): (0.193, 0.162),
    ("h13_b", "Severe APC Loss"): (1.714, 0.575),
}


def load(run_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "protocol_*.json"))):
        with open(path) as fh:
            rows.extend(json.load(fh))
    return rows


def get(rows, edge, regime, proto):
    for r in rows:
        if (r["edge"] == edge and r["regime"] == regime
                and r["protocol"] == proto):
            return r
    return None


def main():
    run_dir = (sys.argv[1] if len(sys.argv) > 1
               else "runs/20260802_protocol_test")
    rows = load(run_dir)
    if not rows:
        print(f"no rows in {run_dir} yet")
        return
    edges, regimes = [], []
    for r in rows:
        if r["edge"] not in edges:
            edges.append(r["edge"])
        if r["regime"] not in regimes:
            regimes.append(r["regime"])

    print(f"\nprospective protocol test — {len(rows)} cells from {run_dir}\n")
    for edge in edges:
        for regime in regimes:
            got = [get(rows, edge, regime, p)
                   for p in LADDER.get(edge, ())]
            if not any(got):
                continue
            atlas = ATLAS.get((edge, regime))
            print(f"  {edge}  {regime}"
                  + (f"   (atlas 10-cond: {atlas[0]:.1%} functional / "
                     f"{atlas[1]:.1%} basal)" if atlas else ""))
            print(f"    {'protocol':<10s} {'reaches':>8s} {'ratio':>8s} "
                  f"{'fNRMSE':>9s} {'basal':>8s} {'ctrl':>8s} "
                  f"{'eq params':>11s}")
            for r, label in zip(got, ("baseline", "NEAR-MISS", "prescribed")):
                if r is None:
                    continue
                reaches = "yes" if r["anchor_ratio"] < 0.01 else "no"
                hb = ("  n/a" if r["hybrid_basal_err"] is None
                      else f"{r['hybrid_basal_err']:7.1%}")
                cb = ("  n/a" if r["ctrl_basal_err"] is None
                      else f"{r['ctrl_basal_err']:7.1%}")
                print(f"    {r['protocol']:<10s} {reaches:>8s} "
                      f"{r['anchor_ratio']:8.4f} {r['term_nrmse']:9.1%} "
                      f"{hb:>8s} {cb:>8s} "
                      f"{r['hybrid_eq_under10']:>4d}/"
                      f"{r['hybrid_n_eq_params']:<6d}  ({label})")
            print()

    print("=" * 72)
    print("  PRE-REGISTERED PREDICTIONS  (see PREREGISTERED.md)")
    print("=" * 72)
    v = {}

    print("\n  Q1  `m_h13` basal `a13` < 3% under the prescribed `mycKO`")
    ok = []
    for regime in regimes:
        r = get(rows, "m_h13", regime, "mycKO")
        if r is None or r["hybrid_basal_err"] is None:
            continue
        good = r["hybrid_basal_err"] < 0.03
        ok.append(good)
        print(f"        {regime:<18s} {r['hybrid_basal_err']:6.1%}  "
              f"{'PASS' if good else 'FAIL'}")
    v["Q1"] = all(ok) if ok else None

    print("\n  Q2  `m_h13` under the NEAR-MISS `bcatKO` stays above 8% "
          "in >=1 regime")
    print("      (falsifier: if the double knockout fixes it too, the table "
          "is measuring")
    print("       perturbation size, not anchor visitation)")
    vals = []
    for regime in regimes:
        r = get(rows, "m_h13", regime, "bcatKO")
        b = get(rows, "m_h13", regime, "none")
        if r is None or r["hybrid_basal_err"] is None:
            continue
        vals.append(r["hybrid_basal_err"])
        base = "" if b is None or b["hybrid_basal_err"] is None else \
            f"  (baseline {b['hybrid_basal_err']:.1%})"
        print(f"        {regime:<18s} {r['hybrid_basal_err']:6.1%}{base}")
    if vals:
        good = max(vals) > 0.08
        v["Q2"] = good
        print(f"        max over regimes = {max(vals):.1%}  "
              f"{'PASS' if good else 'FAIL (falsified)'}")

    print("\n  Q3  `m_h13` equation-parameter count rises under `mycKO` and "
          "not under `bcatKO`")
    ok = []
    for regime in regimes:
        b = get(rows, "m_h13", regime, "none")
        n = get(rows, "m_h13", regime, "bcatKO")
        p = get(rows, "m_h13", regime, "mycKO")
        if not all((b, n, p)):
            continue
        good = (p["hybrid_eq_under10"] > b["hybrid_eq_under10"]
                and n["hybrid_eq_under10"] <= b["hybrid_eq_under10"])
        ok.append(good)
        print(f"        {regime:<18s} none={b['hybrid_eq_under10']}"
              f"/{b['hybrid_n_eq_params']}  "
              f"bcatKO={n['hybrid_eq_under10']}/{n['hybrid_n_eq_params']}  "
              f"mycKO={p['hybrid_eq_under10']}/{p['hybrid_n_eq_params']}  "
              f"{'PASS' if good else 'FAIL'}")
    v["Q3"] = all(ok) if ok else None

    print("\n  Q4  `m_h13` functional NRMSE < 20% in Normal under `mycKO` "
          "(from 57.2%)")
    r = get(rows, "m_h13", "Normal", "mycKO")
    if r is not None:
        good = r["term_nrmse"] < 0.20
        v["Q4"] = good
        print(f"        Normal             {r['term_nrmse']:6.1%}  "
              f"{'PASS' if good else 'FAIL'}")

    print("\n  Q5  `h13_b` basal `W` < 5% under `hox13KO`  "
          "(SECONDARY — confounded, see PREREGISTERED.md)")
    ok = []
    for regime in regimes:
        r = get(rows, "h13_b", regime, "hox13KO")
        if r is None or r["hybrid_basal_err"] is None:
            continue
        good = r["hybrid_basal_err"] < 0.05
        ok.append(good)
        print(f"        {regime:<18s} {r['hybrid_basal_err']:6.1%}  "
              f"{'PASS' if good else 'FAIL'}")
    v["Q5"] = all(ok) if ok else None

    print("\n  " + "-" * 68)
    for k in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        got = v.get(k)
        print(f"    {k}  " + ("PASS" if got is True else
                              "FAIL" if got is False else "not yet scorable")
              + ("   (secondary, cannot carry the claim)" if k == "Q5" else ""))
    print("  " + "-" * 68)


if __name__ == "__main__":
    main()
