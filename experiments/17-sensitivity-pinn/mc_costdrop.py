"""2*(cost at truth - cost at the MLE) for every Monte Carlo fit in
runs/20261017_paper/mc (unit weights, deltaP1 known); should be ~chi^2_35 if the
fits reached the optimum. Cost at truth uses the exact (Radau) reference at the
observation times, i.e. the noise realisation itself."""
import glob, json, os
import numpy as np
import bench
from wnt_pinn.model.parameters import REGIMES

import sys
RUN = "runs/20261017_paper"
MCDIR = sys.argv[1] if len(sys.argv) > 1 else "mc"
out = {}
for r in REGIMES:
    out[r] = []
    for fn in sorted(glob.glob(f"{RUN}/{MCDIR}/{r.replace(' ', '_')}_truth_s*.json")):
        d = json.load(open(fn))
        seed = int(fn.rsplit("_s", 1)[1].split(".")[0])
        data = bench.make_data(r, seed)
        ct = 0.0
        for c in data:
            idx = np.searchsorted(c["t_ref"], c["t_d"])
            ct += 0.5 * float(np.sum(((c["y_d"] - c["y_ref"][idx]) / bench.NOISE) ** 2))
        out[r].append(2 * (ct - d["cost"]))
    print(r, len(out[r]), np.round(out[r], 1))
json.dump(out, open(f"{RUN}/{MCDIR}_costdrop.json", "w"), indent=1)
