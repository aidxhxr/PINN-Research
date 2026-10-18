"""Classical maximum-likelihood (weighted least squares) reference estimator.

Trust-region least squares on the shooting objective with finite-difference
Jacobians computed in a process pool, run to convergence (not capped at 80
evaluations as the July ODE-fit was). Answers: is the Cramer-Rao bound
attainable from the +50% initial guess, and what does the estimator give when
started at the truth?

Usage: python3 mle.py <out_dir> <regime> [--start init|truth] [--workers N]
"""
import argparse, json, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares

import bench
from wnt_pinn.model.parameters import BASELINE, UNKNOWN

DATA = None
WEIGHTS = "rel"
FIX = {}
RTOL = 1e-8


def _init(regime, seed, weights="rel", fix=None, design="base", rtol=1e-8):
    global DATA, WEIGHTS, FIX, RTOL
    RTOL = rtol
    DATA = bench.make_data(regime, seed, design)
    WEIGHTS = weights
    FIX = fix or {}


class _SolveTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _SolveTimeout()


def _resid_cond(args):
    """One condition's residual. A solve that runs longer than 30 s (LSODA can
    stall indefinitely at stiff, far-from-truth jittered parameters) is
    treated as a failed solve and penalised, like a solver failure."""
    import signal
    x, ci = args
    p = {**BASELINE, **bench.from_x(x), **FIX, **DATA[ci]["forcing"]}
    d = DATA[ci]
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(30)
    try:
        y = bench.solve(p, d["t_d"], rtol=RTOL, atol=RTOL * 1e-2, method="LSODA")
    except _SolveTimeout:
        y = None
    finally:
        signal.alarm(0)
    if y is None:
        return np.full(d["y_d"].size, 1e3)
    w = d["w"] if WEIGHTS == "rel" else 1.0
    return (((y - d["y_d"]) * w) / bench.NOISE).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("regime")
    ap.add_argument("--start", default="init", choices=["init", "truth", "pinn"])
    ap.add_argument("--pinn-dir", default="../10-inverse-integral-residual/runs/20260711_203325_integral")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-nfev", type=int, default=400)
    ap.add_argument("--jitter-seed", type=int, default=None,
                    help="add N(0, 0.35^2) jitter in log space to the +50%% init")
    ap.add_argument("--weights", default="rel", choices=["rel", "none"],
                    help="rel = the PINN's 1/max|y| state weights (shared benchmark "
                         "objective); none = the true iid-Gaussian MLE")
    ap.add_argument("--fix-deltaP1", action="store_true",
                    help="hold deltaP1 at truth (removes the structural null "
                         "direction thetaP<->deltaP1)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--design", default="base", choices=list(bench.EXTRA))
    ap.add_argument("--rtol", type=float, default=1e-8,
                    help="LSODA relative tolerance (atol = rtol/100)")
    ap.add_argument("--fd-step", type=float, default=1e-6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = bench.make_data(a.regime, a.seed, a.design)
    nC = len(data)
    fix = {"deltaP1": BASELINE["deltaP1"]} if a.fix_deltaP1 else {}
    pool = ProcessPoolExecutor(a.workers, initializer=_init,
                               initargs=(a.regime, a.seed, a.weights, fix, a.design, a.rtol))

    def resid(x):
        return np.concatenate(list(pool.map(_resid_cond, [(x, ci) for ci in range(nC)])))

    def jac(x):
        h = a.fd_step
        jobs = [(x + h * e, ci) for e in np.eye(len(x)) for ci in range(nC)]
        r0 = resid(x)
        cols = list(pool.map(_resid_cond, jobs, chunksize=4))
        J = np.empty((r0.size, len(x)))
        for j in range(len(x)):
            J[:, j] = (np.concatenate(cols[j * nC:(j + 1) * nC]) - r0) / h
        return J

    if a.start == "truth":
        x0 = bench.to_x(bench.true_params(a.regime))
    elif a.start == "pinn":
        import torch
        raw = torch.load(os.path.join(a.pinn_dir, a.regime.replace(" ", "_") + "_params.pt"),
                         map_location="cpu")
        x0 = bench.to_x({k: (1 / (1 + np.exp(-float(raw[f"raw.{k}"]))) if k == "thetaP"
                             else BASELINE[k] * np.exp(float(raw[f"raw.{k}"]))) for k in UNKNOWN})
    else:
        x0 = bench.to_x(bench.INIT_GUESS)
    if a.jitter_seed is not None:
        x0 = x0 + np.random.default_rng(a.jitter_seed).normal(0, 0.35, x0.shape)
        a.start = f"{a.start}_j{a.jitter_seed}"
    t0 = time.time()
    it = [0]

    def cb_resid(x):
        r = resid(x)
        it[0] += 1
        rec = bench.from_x(x)
        print(f"  eval {it[0]:4d}  cost={0.5*r@r:.4e}  under10={bench.count_under(rec, a.regime)}"
              f"  [{time.time()-t0:.0f}s]", flush=True)
        return r

    res = least_squares(cb_resid, x0, jac=jac, method="trf", x_scale="jac",
                        ftol=1e-12, xtol=1e-12, gtol=1e-10, max_nfev=a.max_nfev)
    rec = {**bench.from_x(res.x), **fix}
    errs = bench.rel_errors(rec, a.regime)
    # Gauss-Newton covariance at the estimate (in x coordinates -> relative sd)
    J = res.jac
    cov = np.linalg.pinv(J.T @ J)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = dict(regime=a.regime, start=a.start, cost=float(res.cost),
               nfev=int(res.nfev), status=int(res.status),
               under10=int(bench.count_under(rec, a.regime)),
               recovered=rec, errors={k: float(v) for k, v in errs.items()},
               sd_x={k: float(s) for k, s in zip(UNKNOWN, sd)},
               seconds=time.time() - t0, weights=a.weights, data_seed=a.seed,
               fix_deltaP1=a.fix_deltaP1, x=res.x.tolist(), design=a.design)
    safe = a.regime.replace(" ", "_")
    with open(os.path.join(a.out, f"{safe}_{a.start}{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"DONE {a.regime} start={a.start} under10={out['under10']}/36 "
          f"cost={res.cost:.4e} nfev={res.nfev} status={res.status}", flush=True)
    pool.shutdown()


if __name__ == "__main__":
    main()
