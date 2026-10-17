"""Truth-free model selection for SGN-PINN runs: the shooting misfit.

For every <regime>_pinn_<tag>.json in a run dir, re-solve the ODE at the
estimated theta (Radau, rtol 1e-10) and compute the relative-weighted data
misfit 0.5 * sum |w (y(theta) - y_d) / sigma|^2 -- the classical objective,
which needs no trust in the network head. The reported estimate per regime is
the schedule with the lowest shooting misfit. (Selecting by the head's own data
misfit rewards trajectories that are not ODE solutions: in v2 Severe, schedule
S2 "won" on head misfit with a physics residual 1e5x the floor.)

Usage: python3 reselect.py <run_dir> [--seed 42] [--tags _S1,_S2]
"""
import argparse, glob, json, os
import numpy as np

import bench
from wnt_pinn.model.parameters import BASELINE, REGIMES


def shooting_cost(rec, regime, data):
    tot = 0.0
    for d in data:
        p = {**BASELINE, **rec, **d["forcing"]}
        y = bench.solve(p, d["t_d"])
        if y is None:
            return np.inf
        tot += 0.5 * float(np.sum((((y - d["y_d"]) * d["w"]) / bench.NOISE) ** 2))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tags", default="_S1,_S2")
    a = ap.parse_args()
    out = {}
    for regime in REGIMES:
        safe = regime.replace(" ", "_")
        data = bench.make_data(regime, a.seed)
        rows = []
        for tag in a.tags.split(","):
            fn = os.path.join(a.run_dir, f"{safe}_pinn{tag}.json")
            if not os.path.exists(fn):
                continue
            r = json.load(open(fn))
            c = shooting_cost(r["recovered"], regime, data)
            rows.append(dict(tag=tag, under10=r["under10"], head_data=r["data_cost"],
                             phys=r["phys_cost"], shooting=c,
                             median_err=float(np.median(list(r["errors"].values())))))
            print(f"{regime:18s} {tag:4s} under10={r['under10']:2d} head_data={r['data_cost']:.4e} "
                  f"phys={r['phys_cost']:.3e} shooting={c:.4e}", flush=True)
        if rows:
            best = min(rows, key=lambda z: z["shooting"])
            out[regime] = dict(rows=rows, chosen=best["tag"], under10=best["under10"])
            print(f"  -> chosen {best['tag']} under10={best['under10']}")
    print("TOTAL", sum(v["under10"] for v in out.values()))
    with open(os.path.join(a.run_dir, "reselect.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
