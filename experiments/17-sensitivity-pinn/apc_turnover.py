"""APC level and the APC-dependent share of beta-catenin removal per regime
(results_c, 'time average 0.48 / 0.076'; '33% to 9% of turnover').
Control condition at the true parameters, time average over t in [10, 150]
(after the initial transient). Removal terms of db/dt: b, lambdaP*apc*b,
lambda5*h5*b/(kappa5+b). Writes runs/20261017_paper/apc_turnover.json."""
import json
import numpy as np

import bench

t = np.linspace(10.0, bench.T, 2801)
out = {}
for regime in bench.REGIMES:
    p = bench.true_params(regime)
    b, apc, h5 = bench.solve(p, t)[:, :3].T
    deg = b
    apc_term = p["lambdaP"] * apc * b
    h5_term = p["lambda5"] * h5 * b / (p["kappa5"] + b)
    out[regime] = dict(apc_mean=float(apc.mean()),
                       apc_share=float(apc_term.mean() / (deg + apc_term + h5_term).mean()))
    print(regime, {k: round(v, 4) for k, v in out[regime].items()})
json.dump(dict(window=[10.0, bench.T], regimes=out), open("runs/20261017_paper/apc_turnover.json", "w"), indent=1)
