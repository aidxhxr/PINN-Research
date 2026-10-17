"""Convergence check for a Monte Carlo MLE fit: restart least squares from the
fitted estimate with stricter stopping (xtol=ftol=1e-15, x_scale=1) and report
whether the cost keeps decreasing and how far the estimate moves.
Usage: python3 mc_restart_check.py <fit.json> [--workers 16]"""
import argparse, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares

import bench
import mle

ap = argparse.ArgumentParser()
ap.add_argument("fit"); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--method", default="trf")
a = ap.parse_args()
r = json.load(open(a.fit))
regime, seed = r["regime"], r["data_seed"]
fix = {"deltaP1": bench.BASELINE["deltaP1"]} if r.get("fix_deltaP1") else {}
nC = len(bench.make_data(regime, seed))
pool = ProcessPoolExecutor(a.workers, initializer=mle._init,
                           initargs=(regime, seed, "none", fix, "base", 1e-11))
free = [i for i, k in enumerate(bench.UNKNOWN) if k not in fix]
x_full = np.array(r["x"])


def full(z):
    x = x_full.copy(); x[free] = z; return x


def resid(z):
    return np.concatenate(list(pool.map(mle._resid_cond, [(full(z), ci) for ci in range(nC)])))


def jac(z):
    h = 1e-5
    r0 = resid(z)
    jobs = [(full(z + h * e), ci) for e in np.eye(len(z)) for ci in range(nC)]
    cols = list(pool.map(mle._resid_cond, jobs, chunksize=4))
    J = np.empty((r0.size, len(z)))
    for j in range(len(z)):
        J[:, j] = (np.concatenate(cols[j * nC:(j + 1) * nC]) - r0) / h
    return J


z0 = x_full[free]
c0 = 0.5 * float(resid(z0) @ resid(z0))
res = least_squares(resid, z0, jac=jac, method=a.method, x_scale=1.0,
                    xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200)
print(json.dumps(dict(fit=a.fit, cost_json=r["cost"], cost_start=c0, cost_restart=float(res.cost),
                      drop=c0 - float(res.cost), max_move=float(np.abs(res.x - z0).max()),
                      nfev=int(res.nfev), status=int(res.status))), flush=True)
pool.shutdown()
