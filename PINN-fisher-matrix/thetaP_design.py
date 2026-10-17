"""Training-free design study for thetaP (2026-09-21).

Question: is the high-WNT loss of thetaP a statement about WHERE THE DATA LIVE,
so that a condition which leaves the beta-catenin plateau restores it -- and a
same-sized condition that does not leave the plateau does not?

Method: Fisher information at the true parameters (log coordinates, all seven
states on 200 points, sigma = 0.002, identical to fisher_matrix.py), with
deltaP1 held KNOWN. thetaP and deltaP1 enter only through deltaP1*(1-thetaP),
so with both free there is an exact null direction in every design and no
design can be compared; fixing deltaP1 removes that one structural direction
and leaves the practical question. The statistic is the Cramer-Rao relative
standard deviation of thetaP with the other 34 parameters free,
sqrt((F^-1)_thetaP,thetaP), per regime and per design:

  ra6        baseline + the five retinoid protocols
  base10     + the four WNT/MYC-perturbing conditions (incl. wntInhib DW=-0.7)
  base10-X   leave-one-out of each of those four  (single-condition ablation)
  base10+info  + ONE information-matched condition (midATRA / wntPulse2):
               regulators move up, the plateau is not left
  base10+kW=k  + ONE graded WNT knockdown, k in {0.5, 0.25, 0.1, 0.02, 0}
  base10+bcatKO  kW = k13 = 0

Also reports min/max of b in the added condition (how far the plateau is left).
In Normal thetaP = 1 and deltaP1 drops out, so thetaP's information is that of
the APC turnover itself; it is reported for completeness.

usage: python3 thetaP_design.py <out_dir>
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from wnt_pinn.model.parameters import BASELINE, REGIMES, CONDITIONS, Y0, UNKNOWN  # noqa
from wnt_pinn.model.numpy_rhs import _ode_rhs                                       # noqa

T, N_TIME, SIGMA, REL_STEP = 150.0, 200, 0.002, 1e-3
FREE = [k for k in UNKNOWN if k != "deltaP1"]

EXTRA = {
    "midATRA":   {"DR": 1.20, "tau1": 60.0, "tau2": 100.0},
    "wntPulse2": {"DW": 0.50, "tauW1": 60.0, "tauW2": 110.0},
    "kW=0.5":    {"kW": 0.5},
    "kW=0.25":   {"kW": 0.25},
    "kW=0.1":    {"kW": 0.1},
    "kW=0.02":   {"kW": 0.02},
    "kW=0":      {"kW": 0.0},
    "bcatKO":    {"kW": 0.0, "k13": 0.0},
}
BASE = {c["name"]: c["forcing"] for c in CONDITIONS}
ALL = {**BASE, **EXTRA}


def solve(p, t_eval):
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0.0, T), Y0, t_eval=t_eval,
                    method="LSODA", rtol=1e-9, atol=1e-11)
    return sol.y.T if sol.success else None


def cond_block(args):
    regime, cname = args
    true_p = {**BASELINE, **REGIMES[regime]}
    forcing = ALL[cname]
    t_eval = np.linspace(0.0, T, N_TIME)
    y0 = solve({**true_p, **forcing}, t_eval)
    cols = []
    for name in FREE:
        p0 = float(true_p[name]); h = REL_STEP * abs(p0)
        yp = solve({**true_p, name: p0 + h, **forcing}, t_eval)
        ym = solve({**true_p, name: p0 - h, **forcing}, t_eval)
        if yp is None or ym is None:
            raise RuntimeError(f"{regime}/{cname}: solve failed for {name}")
        cols.append(p0 * (yp - ym).ravel() / (2.0 * h))
    S = np.column_stack(cols)
    F = S.T @ S / SIGMA ** 2
    return regime, cname, F, float(y0[:, 0].min()), float(y0[:, 0].max())


def crlb(F, name="thetaP"):
    w, V = np.linalg.eigh(F)
    w = np.maximum(w, 1e-300)
    j = FREE.index(name)
    return float(np.sqrt(np.sum(V[j, :] ** 2 / w))), float(w.max() / w.min())


def main(out):
    os.makedirs(out, exist_ok=True)
    jobs = [(r, c) for r in REGIMES for c in ALL]
    t0 = time.time()
    with Pool(32) as pool:
        res = pool.map(cond_block, jobs)
    Fc = {(r, c): F for r, c, F, _, _ in res}
    brange = {(r, c): (lo, hi) for r, c, _, lo, hi in res}
    print(f"sensitivities: {len(jobs)} blocks in {time.time()-t0:.0f}s", flush=True)

    base = list(BASE)
    ra6 = [c for c in base if c not in ("wntPulse", "wntInhib", "mycPulse", "wntRAcombo")]
    designs = {"ra6": ra6, "base10": base}
    for x in ("wntPulse", "wntInhib", "mycPulse", "wntRAcombo"):
        designs[f"base10-{x}"] = [c for c in base if c != x]
    for e in EXTRA:
        designs[f"base10+{e}"] = base + [e]

    rows = []
    for regime in REGIMES:
        print(f"\n{regime}  (W={REGIMES[regime]['W']}, thetaP={REGIMES[regime]['thetaP']})")
        print(f"  {'design':<22s} {'n':>3s} {'relsd(thetaP)':>14s} {'x vs base10':>12s} {'relsd(W)':>10s} {'cond':>10s}  b range of added cond")
        ref = None
        for dname, conds in designs.items():
            F = sum(Fc[(regime, c)] for c in conds)
            sd, cn = crlb(F, "thetaP"); sdW, _ = crlb(F, "W")
            if dname == "base10":
                ref = sd
            extra = dname.split("+")[1] if "+" in dname else None
            br = brange[(regime, extra)] if extra else brange[(regime, base[0])]
            rows.append(dict(regime=regime, design=dname, n_cond=len(conds),
                             relsd_thetaP=sd, relsd_W=sdW, cond=cn,
                             added=extra, b_min=br[0], b_max=br[1]))
        for r in [x for x in rows if x["regime"] == regime]:
            r["ratio_to_base10"] = r["relsd_thetaP"] / ref
            print(f"  {r['design']:<22s} {r['n_cond']:>3d} {r['relsd_thetaP']:>14.4g} "
                  f"{r['ratio_to_base10']:>12.3f} {r['relsd_W']:>10.4g} {r['cond']:>10.2e}  "
                  f"[{r['b_min']:.3f}, {r['b_max']:.3f}]")
    json.dump(rows, open(os.path.join(out, "thetaP_design.json"), "w"), indent=2)
    print(f"\nwrote {out}/thetaP_design.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
