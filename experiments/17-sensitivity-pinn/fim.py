"""Fisher information of the benchmark design (bench.py) at the true parameters.

F = J^T J / sigma^2 with J = d y(t_d) / d log(theta), all 7 states, the 150
observation times of each condition, central finite differences on a Radau
solve (rtol 1e-10). Two versions:

* ``none``: the iid N(0, sigma^2) likelihood -> the Cramer-Rao bound proper;
* ``rel``:  the relative-weighted objective every estimator minimises, for
  which the asymptotic covariance is the sandwich
  (J'WJ)^-1 J'W^2J (J'WJ)^-1 sigma^2.

Also stores the 35x35 bound with deltaP1 known (the 36-parameter problem has
an exact null direction: thetaP enters only via 1 + deltaP1 (1 - thetaP)).

Usage: python3 fim.py <out_dir> [--seed 42] [--workers 40]
"""
import argparse, json, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import bench
from wnt_pinn.model.parameters import BASELINE, REGIMES, UNKNOWN

H = 1e-4


def _sens(args):
    regime, seed, ci, j, design = args
    d = bench.make_data(regime, seed, design)[ci]
    x0 = bench.to_x(bench.true_params(regime))
    out = []
    for s in (+1, -1):
        x = x0.copy(); x[j] += s * H
        p = {**bench.true_params(regime), **bench.from_x(x), **d["forcing"]}
        out.append(bench.solve(p, d["t_d"]))
    return (out[0] - out[1]) / (2 * H)        # (n_t, 7)


def fim_for(regime, seed, pool, design="base"):
    data = bench.make_data(regime, seed, design)
    nC, P = len(data), len(UNKNOWN)
    jobs = [(regime, seed, ci, j, design) for ci in range(nC) for j in range(P)]
    S = list(pool.map(_sens, jobs))
    Jn, Jw = [], []
    for ci in range(nC):
        cols = np.stack([S[ci * P + j] for j in range(P)], axis=-1)  # (n_t,7,P)
        Jn.append(cols.reshape(-1, P))
        Jw.append((cols * data[ci]["w"][None, :, None]).reshape(-1, P))
    Jn, Jw = np.vstack(Jn), np.vstack(Jw)
    sig2 = bench.NOISE ** 2
    F = Jn.T @ Jn / sig2
    A = Jw.T @ Jw / sig2
    wrow = np.concatenate([np.repeat(d["w"][None, :], len(d["t_d"]), 0).ravel()
                           for d in data])
    B = (Jw * (wrow ** 2)[:, None]).T @ Jw / sig2
    return F, A, B, Jn, np.stack([np.stack([S[ci * P + j] for j in range(P)], axis=-1)
                                  for ci in range(nC)])  # (nC, n_t, 7, P)


def sds(F, keep):
    Fi = np.linalg.inv(F[np.ix_(keep, keep)])
    return np.sqrt(np.clip(np.diag(Fi), 0, None)), Fi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--design", default="base", choices=list(bench.EXTRA))
    ap.add_argument("--save-jac", action="store_true",
                    help="also save the per-condition sensitivity array (nC, n_t, 7, P)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keep = [i for i, k in enumerate(UNKNOWN) if k != "deltaP1"]
    summary = {}
    with ProcessPoolExecutor(a.workers) as pool:
        for regime in REGIMES:
            t0 = time.time()
            F, A, B, J, Jc = fim_for(regime, a.seed, pool, a.design)
            safe = regime.replace(" ", "_")
            if a.save_jac:
                np.save(os.path.join(a.out, f"{safe}_Jcond.npy"), Jc)
            np.save(os.path.join(a.out, f"{safe}_F.npy"), F)
            np.save(os.path.join(a.out, f"{safe}_Aw.npy"), A)
            np.save(os.path.join(a.out, f"{safe}_Bw.npy"), B)
            ev = np.linalg.eigvalsh(F)
            sd_n, cov_n = sds(F, keep)
            Ai = np.linalg.inv(A[np.ix_(keep, keep)])
            cov_w = Ai @ B[np.ix_(keep, keep)] @ Ai
            sd_w = np.sqrt(np.diag(cov_w))
            np.save(os.path.join(a.out, f"{safe}_cov35_none.npy"), cov_n)
            np.save(os.path.join(a.out, f"{safe}_cov35_rel.npy"), cov_w)
            names = [UNKNOWN[i] for i in keep]
            summary[regime] = dict(
                eig36=ev.tolist(),
                cond35=float(np.linalg.cond(F[np.ix_(keep, keep)])),
                sd35_none=dict(zip(names, sd_n.tolist())),
                sd35_rel=dict(zip(names, sd_w.tolist())),
                n5_none=int((sd_n < 0.05).sum()), n10_none=int((sd_n < 0.10).sum()),
                n5_rel=int((sd_w < 0.05).sum()), n10_rel=int((sd_w < 0.10).sum()),
                seconds=time.time() - t0, workers=a.workers, design=a.design)
            print(regime, "cond35=%.2e" % summary[regime]["cond35"],
                  "sd<5%% none=%d rel=%d" % (summary[regime]["n5_none"], summary[regime]["n5_rel"]),
                  "thetaP sd none=%.4f" % summary[regime]["sd35_none"]["thetaP"],
                  "min eig36=%.2e" % ev[0], flush=True)
    with open(os.path.join(a.out, "fim_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
