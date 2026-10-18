"""Training-free Cramer-Rao bound for the BASAL parameter of an equation whose
regulator term is learned (the hybrid half of the paper's yardstick).

For a production edge hosted by equation k,

    eps_k dx_k/dt = kb*a + kf*f(u) + (other known/mechanistic terms) - x_k ,

with f learned under the anchor f(0)=0, this computes the Fisher information of
the equation-local screen's own data model and the CRLB of `a` after the
nuisance parameters (f's parameters and the equation's other mechanistic
parameters) are profiled out.  No training of the ODE problem is involved: the
FIM is evaluated at the truth (mechanistic parameters at their true values, f
at a least-squares fit of the family to the true Hill term).

DATA MODEL (identical to screen_terms.py / dose_response.py):
  * exact reference states (Radau, rtol 1e-10), grid t_ref[::stride], stride 8
    (the stride of the atlas, A/B, infoctl, dose and protocol runs);
  * residual per condition c and grid step i, equation row k:
        r_ci = [ (z_{i+1}-z_i) - dt/2 (F_k(i+1) + F_k(i)) ] / s_c ,
    s_c = max(max_t |x_k|, 0.05) over that condition (screen's `scale`);
  * statistical model r_ci ~ iid N(0, sigma_r^2)  -> FIM = J^T J / sigma_r^2.
    The screen's loss is the mean of r^2, i.e. exactly this Gaussian NLL up to
    a constant, so the CRLB is the bound for the screen's own estimator class.
    Reported: S = sd(log a) / sigma_r  (relative sd per unit residual noise;
    sigma-free, so ratios between designs and Spearman correlations do not
    depend on sigma_r), and sd_rel at sigma_r = 1e-4 (the RMS residual floor
    the screen's deterministic control fits stop at, physics MSE ~1e-8).

PARAMETERISATION: mechanistic parameters of equation k (screen's EQ_PARAMS
minus FIXED minus the two the edge replaces) enter as p = p_true*exp(theta),
so sd(theta_a) is the RELATIVE sd of a.  Families for f:
  * mech      the closed-form Hill with its (eta, kappa) unknown (the screen's
              mechanistic control);
  * lin_K     f(u) = sum_{k=1..K} c_k (u/u_max)^k, K=1..6 (anchor exact, linear
              in c, transparent);
  * gated/sc  the SAME 2x5 tanh nets the screen builds (hybrid.build_one_term),
              all ~46 weights as nuisance, fit to the true Hill on [0, u_max].
Profiling: I_a|nuis = J_a^T (I - P) J_a with P the projection onto range of
the nuisance columns.  The net Jacobian is rank-deficient (over-parameterised),
so its range is taken at SVD tolerance rtol (default 1e-8; 1e-6 and 1e-10
reported for sensitivity).

usage:
  python3 hybrid_fim.py --out runs/20261017_hybrid_fim             # main sweep
  python3 hybrid_fim.py --out runs/20261017_hybrid_fim --verify    # FD + MC
"""
from __future__ import annotations

import os
import sys

# CPU only; one BLAS thread per worker (<= 8 workers); full parameter set.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HYBRID_TERM"] = "none"
for _v in ("HYBRID_DEPLETION", "HYBRID_INFOCTL"):
    os.environ.pop(_v, None)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import (BASELINE, CONDITIONS, DEPLETION_CONDITIONS,  # noqa: E402
                    BCAT_DEPLETION_CONDITIONS, FIXED, HYBRID_TERMS,
                    INFO_CONDITIONS, REGIMES, Y0)
from hybrid import VAR_INDEX, build_one_term, true_term  # noqa: E402
from residual import physics_rhs  # noqa: E402
from odes import _ode_rhs  # noqa: E402
from anchor_doses import ARMS, DOSES  # noqa: E402
from anchor_reach import PROTOCOLS, BASAL_KNOB  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

EQ_ROW = {"db": 0, "dapc": 1, "dh5": 2, "dh13": 3, "dm": 4, "dr": 5, "dc": 6}
EQ_PARAMS = {   # copied from screen_terms.py (importing it pulls in torch nets)
    "db":   ["W", "eta13", "kappa13", "lambdaP", "lambda5", "kappa5"],
    "dapc": ["epsP", "rho5", "rhoB", "rho13", "deltaP1", "thetaP"],
    "dh5":  ["eps5", "a5", "etaR", "kappaR", "etaM", "kappaM"],
    "dh13": ["eps13", "a13", "etaB13", "kappaB13", "etaM13", "kappaM13"],
    "dm":   ["epsM", "aM", "etaBM", "kappaBM"],
    "dr":   ["epsR", "lambdaC"],
    "dc":   ["epsC", "aC", "etaRC", "kappaRC", "etaBC", "kappaBC"],
}
# the production edges (additive f, basal parameter in the same equation)
EDGES = [t for t, s in HYBRID_TERMS.items()
         if s.get("basal") and s.get("input_kind") != "parameter"]
REGIME_LIST = list(REGIMES)
DOSE_REGIMES = ["Normal", "Severe APC Loss"]
DOSE_EDGE = {"ra": "ra_h5", "wnt": "bm_myc", "bcat": "bm_myc"}
STRIDE = 8
SIGMA_REF = 1e-4
RTOLS = (1e-6, 1e-8, 1e-10)
RTOL_MAIN = 1e-8
NET_SEEDS = (0, 1, 2)
LIN_K = (1, 2, 3, 4, 5, 6)


# ---------------------------------------------------------------- designs
def designs():
    base = list(CONDITIONS)
    d = {"base10": base,
         "dep": base + list(DEPLETION_CONDITIONS),
         "depb": base + list(BCAT_DEPLETION_CONDITIONS),
         "info": base + list(INFO_CONDITIONS)}
    for name, forcing in PROTOCOLS[1:]:
        d[f"proto_{name}"] = base + [{"name": name, "forcing": forcing}]
    return d


def dose_designs():
    return {f"dose_{arm}_k{k:g}": list(CONDITIONS) +
            [{"name": f"{arm}Dep", "forcing": ARMS[arm](k)}]
            for arm in ARMS for k in DOSES}


def fkey(forcing):
    return tuple(sorted((k, float(v)) for k, v in forcing.items()))


def _solve(args):
    regime, forcing = args
    p = {**BASELINE, **REGIMES[regime], **dict(forcing)}
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, 150.0), Y0,
                    t_eval=np.linspace(0, 150.0, 5000), method="Radau",
                    rtol=1e-10, atol=1e-12)
    assert sol.success, sol.message
    return regime, forcing, sol.t, sol.y.T


def solve_all(out, workers):
    path = os.path.join(out, "refs_cache.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    jobs = set()
    for r in REGIME_LIST:
        for conds in list(designs().values()) + list(dose_designs().values()):
            for c in conds:
                jobs.add((r, fkey(c["forcing"])))
    jobs = sorted(jobs)
    print(f"solving {len(jobs)} reference trajectories", flush=True)
    refs = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, f, t, y in ex.map(_solve, jobs):
            refs[(r, f)] = (t, y)
    with open(path, "wb") as fh:
        pickle.dump(refs, fh)
    return refs


def design_refs(refs, regime, conds):
    """name -> (t, y) for one design, the dict build_one_term/anchor use."""
    return {c["name"]: refs[(regime, fkey(c["forcing"]))] for c in conds}


def anchor_ratio(dref, var):
    idx = VAR_INDEX[var]
    lo = min(float(y[:, idx].min()) for _t, y in dref.values())
    hi = max(float(y[:, idx].max()) for _t, y in dref.values())
    return max(lo, 0.0) / max(hi, 1e-12)


# ---------------------------------------------------------------- residual
def make_data(refs, regime, conds, row, stride=STRIDE):
    out = []
    for c in conds:
        t_ref, y_ref = refs[(regime, fkey(c["forcing"]))]
        scale = max(float(np.abs(y_ref[:, row]).max()), 0.05)
        out.append(dict(forcing=c["forcing"],
                        t=torch.tensor(t_ref[::stride]).reshape(-1, 1),
                        y=torch.tensor(y_ref[::stride]),
                        scale=scale))
    return out


def residual_vec(data, row, pdict_fn, nn_terms):
    rs = []
    for c in data:
        p = pdict_fn(c["forcing"])
        f = physics_rhs(c["t"], c["y"], p, nn_terms)[:, row]
        z = c["y"][:, row]
        dt = (c["t"][1:] - c["t"][:-1]).reshape(-1)
        rs.append(((z[1:] - z[:-1]) - 0.5 * dt * (f[1:] + f[:-1]))
                  / c["scale"])
    return torch.cat(rs)


class Problem:
    """theta = [log(p/p_true) for mech keys] + [family params]."""

    def __init__(self, edge, regime, data, family, net=None, lin=None):
        spec = HYBRID_TERMS[edge]
        self.edge, self.eq = edge, spec["eq"]
        self.row = EQ_ROW[self.eq]
        self.true = {**BASELINE, **REGIMES[regime]}
        replaced = set() if family == "mech" else set(spec["replaces"])
        self.keys = [k for k in EQ_PARAMS[self.eq]
                     if k not in FIXED and k not in replaced]
        self.basal = spec["basal"]
        self.data = data
        self.family = family
        self.net = net
        self.lin = lin            # (K, u_max)
        self.n_mech = len(self.keys)
        if family == "mech":
            self.theta0 = torch.zeros(self.n_mech)
        elif family == "lin":
            self.theta0 = torch.cat([torch.zeros(self.n_mech),
                                     torch.as_tensor(lin["c0"])])
        else:
            self.names = [n for n, _ in net.named_parameters()]
            self.shapes = [p.shape for _, p in net.named_parameters()]
            flat = torch.cat([p.detach().reshape(-1)
                              for _, p in net.named_parameters()])
            self.theta0 = torch.cat([torch.zeros(self.n_mech), flat])
        self.buffers = (dict(net.named_buffers()) if net is not None else {})

    def resid(self, theta):
        mech = {k: self.true[k] * torch.exp(theta[i])
                for i, k in enumerate(self.keys)}
        nn_terms = {}
        if self.family == "lin":
            c = theta[self.n_mech:]
            us = self.lin["u_max"]

            def flin(x, c=c, us=us):
                xp = torch.clamp(x, min=0.0) / us
                return sum(c[k] * xp ** (k + 1) for k in range(len(c)))
            nn_terms = {self.edge: flin}
        elif self.family in ("gated", "sc"):
            params, o = {}, self.n_mech
            for n, s in zip(self.names, self.shapes):
                m = int(np.prod(s)) if len(s) else 1
                params[n] = theta[o:o + m].reshape(s)
                o += m
            net, buf = self.net, self.buffers

            def fnet(x, params=params):
                return torch.func.functional_call(net, {**params, **buf},
                                                  (x,))
            nn_terms = {self.edge: fnet}

        def pdict(forcing):
            return {**self.true, **mech, **forcing}
        return residual_vec(self.data, self.row, pdict, nn_terms)

    def jac(self, theta=None):
        th = self.theta0 if theta is None else theta
        return torch.func.jacfwd(self.resid)(th).detach().numpy()

    @property
    def ia(self):
        return self.keys.index(self.basal)


def profiled_sd(J, n_mech, ia, rtol=RTOL_MAIN):
    """sd of theta_a (per unit sigma_r) with all other columns profiled."""
    Jm, Jp = J[:, :n_mech], J[:, n_mech:]
    rank = None
    if Jp.shape[1]:
        U, s, _ = np.linalg.svd(Jp, full_matrices=False)
        rank = int((s > rtol * s[0]).sum())
        Q = U[:, :rank]
        Jm = Jm - Q @ (Q.T @ Jm)
    _, s, Vt = np.linalg.svd(Jm, full_matrices=False)
    if s[-1] <= 1e-13 * s[0]:
        return float("inf"), rank
    cov = (Vt.T / s ** 2) @ Vt
    return float(np.sqrt(cov[ia, ia])), rank


# ---------------------------------------------------------------- families
_NET_CACHE = {}


def fit_net(edge, form, dref, seed):
    torch.manual_seed(seed)
    net = build_one_term(edge, dref, "cpu", param=form)
    xmax = float(np.asarray(net.x_scale).reshape(-1)[0])
    key = (edge, form, seed, round(xmax, 10))
    if key in _NET_CACHE:
        net.load_state_dict(_NET_CACHE[key][0])
        return net, _NET_CACHE[key][1]
    x = torch.linspace(0.0, xmax, 400).reshape(-1, 1)
    y = torch.as_tensor(true_term(edge, x.numpy().ravel(), dict(BASELINE)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(3000):
        opt.zero_grad()
        L = ((net(x).reshape(-1) - y) ** 2).mean()
        L.backward()
        opt.step()
    lb = torch.optim.LBFGS(net.parameters(), lr=0.5, max_iter=1000,
                           tolerance_grad=1e-14, tolerance_change=1e-18,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        L = ((net(x).reshape(-1) - y) ** 2).mean()
        L.backward()
        return L
    lb.step(closure)
    with torch.no_grad():
        nrmse = float(torch.sqrt(((net(x).reshape(-1) - y) ** 2).mean())
                      / torch.sqrt((y ** 2).mean()))
    _NET_CACHE[key] = ({k: v.clone() for k, v in net.state_dict().items()},
                       nrmse)
    return net, nrmse


def lin_init(edge, dref, K):
    var = HYBRID_TERMS[edge]["inputs"][0]
    idx = VAR_INDEX[var]
    u = np.concatenate([y[:, idx] for _t, y in dref.values()])
    u = np.maximum(u, 0.0)
    us = max(float(u.max()), 1e-3)
    A = np.stack([(u / us) ** (k + 1) for k in range(K)], axis=1)
    c0, *_ = np.linalg.lstsq(A, true_term(edge, u, dict(BASELINE)),
                             rcond=None)
    return dict(c0=c0, u_max=us)


def evaluate_case(args):
    """All families for one (regime, design, edge)."""
    regime, dname, conds, edge, refs_sub, families = args
    torch.set_num_threads(1)
    spec = HYBRID_TERMS[edge]
    row = EQ_ROW[spec["eq"]]
    dref = {c["name"]: refs_sub[fkey(c["forcing"])] for c in conds}
    data = make_data({(regime, k): v for k, v in refs_sub.items()},
                     regime, conds, row)
    var = spec["inputs"][0]
    out = dict(regime=regime, design=dname, edge=edge, eq=spec["eq"],
               basal=spec["basal"], regulator=var,
               n_conditions=len(conds),
               n_residuals=int(sum(len(c["t"]) - 1 for c in data)),
               rho=anchor_ratio(dref, var),
               basal_deleted_in=[c["name"] for c in conds
                                 if float(c["forcing"].get(
                                     BASAL_KNOB[spec["basal"]], 1.0)) == 0.0],
               S={})
    if "mech" in families:
        pr = Problem(edge, regime, data, "mech")
        out["S"]["mech"] = profiled_sd(pr.jac(), pr.n_mech, pr.ia)[0]
    if "lin" in families:
        for K in LIN_K:
            pr = Problem(edge, regime, data, "lin", lin=lin_init(edge, dref, K))
            out["S"][f"lin_{K}"] = profiled_sd(pr.jac(), pr.n_mech, pr.ia)[0]
    for form in ("gated", "sc"):
        if form not in families:
            continue
        per = []
        for seed in NET_SEEDS:
            net, nrmse = fit_net(edge, form, dref, seed)
            pr = Problem(edge, regime, data, form, net=net)
            J = pr.jac()
            sds = {}
            for rt in RTOLS:
                sd, rank = profiled_sd(J, pr.n_mech, pr.ia, rt)
                sds[f"{rt:g}"] = dict(sd=sd, rank=rank)
            per.append(dict(seed=seed, fit_nrmse=nrmse,
                            n_net_params=J.shape[1] - pr.n_mech, sds=sds))
        out[f"{form}_seeds"] = per
        main = [p["sds"][f"{RTOL_MAIN:g}"]["sd"] for p in per]
        out["S"][form] = float(np.median(main))
        for rt in RTOLS:
            out["S"][f"{form}@{rt:g}"] = float(np.median(
                [p["sds"][f"{rt:g}"]["sd"] for p in per]))
    out["sd_rel_at_sigma_ref"] = {k: v * SIGMA_REF for k, v in out["S"].items()}
    return out


# ---------------------------------------------------------------- measured
def load_measured():
    R = os.path.join(HERE, "runs")
    m = []

    def add(src, edge, regime, design, form, hyb, ctrl, rho=None):
        if hyb is None or (isinstance(hyb, float) and np.isnan(hyb)):
            return
        m.append(dict(source=src, edge=edge, regime=regime, design=design,
                      form=form, hybrid_basal_err=float(hyb),
                      ctrl_basal_err=(None if ctrl is None else float(ctrl)),
                      measured_rho=rho))
    # atlas, Normal (log only)
    pat = re.compile(r"^\s+(Normal)\s+(\w+)\s+(gated|sc)\s+fNRMSE=.*"
                     r"basal_err=([\d.]+|nan)% \(ctrl ([\d.]+|nan)%\)")
    with open(os.path.join(R, "20260801_204046_screen_atlas",
                           "screen.log")) as fh:
        for line in fh:
            g = pat.match(line)
            if g and g.group(4) != "nan":
                add("atlas", g.group(2), g.group(1), "base10", g.group(3),
                    float(g.group(4)) / 100, float(g.group(5)) / 100)
    for row in json.load(open(os.path.join(
            R, "20260802_screen_atlas_rest", "screen.json"))):
        if "basal_err" in row:
            add("atlas", row["term"], row["regime"], "base10", row["param"],
                row["basal_err"], row.get("ctrl_basal_err"),
                row["x_lo"] / row["x_hi"])
    for sub, design in (("screen_nodep", "base10"), ("screen_dep", "dep")):
        for row in json.load(open(os.path.join(
                R, "20260801_depletion_ab", sub, "screen.json"))):
            if "basal_err" in row:
                add("AB", row["term"], row["regime"], design, row["param"],
                    row["basal_err"], row.get("ctrl_basal_err"),
                    row["x_lo"] / row["x_hi"])
    for row in json.load(open(os.path.join(R, "20260801_infoctl",
                                           "screen.json"))):
        if "basal_err" in row:
            add("infoctl", row["term"], row["regime"], "info", row["param"],
                row["basal_err"], row.get("ctrl_basal_err"),
                row["x_lo"] / row["x_hi"])
    for d, src in (("20260802_dose_response", "dose"),
                   ("20260802_dose_starts5", "dose5")):
        for arm in ARMS:
            f = os.path.join(R, d, f"dose_{arm}_{DOSE_EDGE[arm]}.json")
            if not os.path.exists(f):
                continue
            for row in json.load(open(f)):
                add(src, row["edge"], row["regime"],
                    f"dose_{arm}_k{row['dose']:g}", row["param"],
                    row["hybrid_basal_err"], row["ctrl_basal_err"],
                    row["anchor_ratio"])
    for e in ("m_h13", "h13_b"):
        for row in json.load(open(os.path.join(
                R, "20260802_protocol_test", f"protocol_{e}.json"))):
            design = ("base10" if row["protocol"] == "none"
                      else f"proto_{row['protocol']}")
            add("protocol", e, row["regime"], design, "gated",
                row["hybrid_basal_err"], row["ctrl_basal_err"],
                row["anchor_ratio"])
    return m


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return dict(rho=None, p=None, n=int(ok.sum()))
    r = spearmanr(x[ok], y[ok])
    return dict(rho=float(r.statistic), p=float(r.pvalue), n=int(ok.sum()))


def correlate(results, measured):
    idx = {(r["regime"], r["design"], r["edge"]): r for r in results}
    rows = []
    for m in measured:
        r = idx.get((m["regime"], m["design"], m["edge"]))
        if r is None:
            continue
        rows.append(dict(m, rho_design=r["rho"],
                         S_form=r["S"].get(m["form"]),
                         S_mech=r["S"].get("mech"),
                         **{f"S_lin_{K}": r["S"].get(f"lin_{K}")
                            for K in LIN_K}))
    out = {"pairs": rows, "spearman": {}}
    groups = {"all": rows}
    for src in sorted({r["source"] for r in rows}):
        groups[src] = [r for r in rows if r["source"] == src]
    groups["all_gated"] = [r for r in rows if r["form"] == "gated"]
    # one point per (edge, regime, design, form): dose5 replaces dose where
    # both exist; this is the de-duplicated pooled set.
    dedup = {}
    for r in rows:
        k = (r["edge"], r["regime"], r["design"], r["form"])
        if k not in dedup or r["source"] == "dose5":
            dedup[k] = r
    groups["pooled_dedup"] = list(dedup.values())
    for g, rr in groups.items():
        hyb = [r["hybrid_basal_err"] for r in rr]
        ctl = [r["ctrl_basal_err"] if r["ctrl_basal_err"] is not None
               else np.nan for r in rr]
        out["spearman"][g] = dict(
            hyb_vs_S_net=sp([r["S_form"] for r in rr], hyb),
            hyb_vs_rho=sp([r["rho_design"] for r in rr], hyb),
            ctrl_vs_S_mech=sp([r["S_mech"] for r in rr], ctl),
            **{f"hyb_vs_S_lin_{K}": sp([r[f"S_lin_{K}"] for r in rr], hyb)
               for K in LIN_K})
    return out


# ---------------------------------------------------------------- summary
def fmt(x):
    if x is None or not np.isfinite(x):
        return "     inf"
    return f"{x:8.3g}"


def write_summary(out, results, corr):
    L = []
    L.append("HYBRID CRLB (training-free) -- basal parameter of a learned-"
             "regulator equation")
    L.append("S = sd(log basal)/sigma_r  (relative sd per unit residual noise;"
             f" multiply by sigma_r, e.g. {SIGMA_REF:g} -> sd_rel)")
    L.append("families: mech = Hill control; lin_K = K-term polynomial "
             "anchored at 0; gated/sc = screen's 2x5 nets (median of 3 fits,"
             f" rtol {RTOL_MAIN:g})\n")
    idx = {(r["regime"], r["design"], r["edge"]): r for r in results}
    for fam in ("gated", "sc", "lin_3", "mech"):
        L.append(f"--- S[{fam}] per edge x regime x design "
                 "(ratio to base10 in brackets) ---")
        des = ["base10", "info", "dep", "depb"]
        L.append(f"{'edge':<8s} {'regime':<17s} {'rho10':>7s} " +
                 " ".join(f"{d:>18s}" for d in des))
        for e in EDGES:
            for rg in REGIME_LIST:
                b = idx[(rg, "base10", e)]
                cells = []
                for d in des:
                    r = idx[(rg, d, e)]
                    v = r["S"].get(fam)
                    rat = v / b["S"][fam] if b["S"].get(fam) else np.nan
                    cells.append(f"{fmt(v)} [x{rat:7.3g}]")
                L.append(f"{e:<8s} {rg:<17s} {b['rho']:7.4f} " +
                         " ".join(f"{c:>18s}" for c in cells))
        L.append("")
    L.append("--- anchor ratio rho per design (regulator min/max) ---")
    for e in EDGES:
        for rg in REGIME_LIST:
            L.append(f"{e:<8s} {rg:<17s} " + " ".join(
                f"{d}={idx[(rg, d, e)]['rho']:.4f}"
                for d in ("base10", "info", "dep", "depb")))
    L.append("\n--- protocol ladder (base10 + one condition): S[gated] "
             "(rho) ---")
    protos = [f"proto_{n}" for n, _ in PROTOCOLS[1:]]
    L.append(f"{'edge':<8s} {'regime':<17s} {'base10':>16s} " +
             " ".join(f"{p[6:]:>16s}" for p in protos))
    for e in EDGES:
        for rg in REGIME_LIST:
            cells = []
            for d in ["base10"] + protos:
                r = idx[(rg, d, e)]
                mark = "*" if r["basal_deleted_in"] else " "
                cells.append(f"{fmt(r['S']['gated'])}{mark}({r['rho']:.3f})")
            L.append(f"{e:<8s} {rg:<17s} " +
                     " ".join(f"{c:>16s}" for c in cells))
    L.append("  * = the added condition sets the basal parameter's knob to 0")
    L.append("\n--- dose-response (11 conditions at every dose) ---")
    for arm in ARMS:
        e = DOSE_EDGE[arm]
        for rg in DOSE_REGIMES:
            L.append(f"arm {arm:<5s} edge {e:<7s} {rg}")
            L.append(f"   {'k':>6s} {'rho':>8s} {'S_gated':>9s} {'S_sc':>9s}"
                     f" {'S_lin3':>9s} {'S_lin6':>9s} {'S_mech':>9s}")
            for k in DOSES:
                r = idx[(rg, f"dose_{arm}_k{k:g}", e)]
                L.append(f"   {k:6g} {r['rho']:8.4f} {fmt(r['S']['gated'])}"
                         f" {fmt(r['S']['sc'])} {fmt(r['S']['lin_3'])}"
                         f" {fmt(r['S']['lin_6'])} {fmt(r['S']['mech'])}")
    L.append("\n--- Spearman: measured basal error vs CRLB S ---")
    L.append(f"{'set':<14s} {'n':>3s} {'S_net':>14s} {'rho':>14s} "
             f"{'S_lin3':>14s} {'S_lin6':>14s} {'ctrl~S_mech':>14s}")
    for g, s in corr["spearman"].items():
        def c(d):
            return ("      -       " if d["rho"] is None else
                    f"{d['rho']:+.2f} (p={d['p']:.1g})".rjust(14))
        L.append(f"{g:<14s} {s['hyb_vs_S_net']['n']:3d} "
                 f"{c(s['hyb_vs_S_net'])} {c(s['hyb_vs_rho'])} "
                 f"{c(s['hyb_vs_S_lin_3'])} {c(s['hyb_vs_S_lin_6'])} "
                 f"{c(s['ctrl_vs_S_mech'])}")
    txt = "\n".join(L)
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------- verify
def verify(out, refs):
    """(1) autodiff Jacobian vs central finite differences;
    (2) Monte Carlo: empirical sd of the basal estimate vs the CRLB."""
    rep = {}
    regime, edge = "Normal", "bm_myc"
    D = designs()
    for dname in ("base10", "depb"):
        conds = D[dname]
        dref = design_refs(refs, regime, conds)
        row = EQ_ROW[HYBRID_TERMS[edge]["eq"]]
        data = make_data(refs, regime, conds, row)
        for fam in ("lin_3", "gated", "mech"):
            if fam == "lin_3":
                pr = Problem(edge, regime, data, "lin", lin=lin_init(edge, dref, 3))
            elif fam == "gated":
                net, _ = fit_net(edge, "gated", dref, 0)
                pr = Problem(edge, regime, data, "gated", net=net)
            else:
                pr = Problem(edge, regime, data, "mech")
            J = pr.jac()
            # --- finite-difference check
            h = 1e-6
            Jfd = np.zeros_like(J)
            with torch.no_grad():
                for j in range(J.shape[1]):
                    e = torch.zeros_like(pr.theta0)
                    e[j] = h
                    Jfd[:, j] = ((pr.resid(pr.theta0 + e)
                                  - pr.resid(pr.theta0 - e)) / (2 * h)).numpy()
            colerr = (np.linalg.norm(J - Jfd, axis=0)
                      / np.maximum(np.linalg.norm(J, axis=0), 1e-300))
            sd_ad = profiled_sd(J, pr.n_mech, pr.ia)[0]
            sd_fd = profiled_sd(Jfd, pr.n_mech, pr.ia)[0]
            key = f"{dname}/{fam}"
            rep[key] = dict(fd_max_col_relerr=float(colerr.max()),
                            fd_median_col_relerr=float(np.median(colerr)),
                            S_autodiff=sd_ad, S_fd=sd_fd)
            print(f"[verify FD] {key:<14s} max col rel err "
                  f"{colerr.max():.2e}  S ad={sd_ad:.4g} fd={sd_fd:.4g}",
                  flush=True)
            # --- Monte Carlo (lin_3 and gated on both designs)
            if fam == "mech":
                continue
            rep[key]["mc"] = monte_carlo(pr, n_rep=200)
            mc = rep[key]["mc"]
            print(f"[verify MC] {key:<14s} sigma_r={mc['sigma']:.3g} "
                  f"CRLB sd={mc['crlb_sd']:.3g} | tGN emp sd="
                  f"{mc['emp_sd_tgn']:.3g} [{mc['emp_sd_tgn_ci95'][0]:.3g},"
                  f"{mc['emp_sd_tgn_ci95'][1]:.3g}] bias "
                  f"{mc['emp_mean_tgn']:+.2g} | LM emp sd={mc['emp_sd_lm']:.3g}"
                  f" [{mc['emp_sd_lm_ci95'][0]:.3g},"
                  f"{mc['emp_sd_lm_ci95'][1]:.3g}] | CRLB by rtol "
                  + " ".join(f"{k}:{v:.3g}" for k, v in
                             mc["crlb_sd_by_rtol"].items()), flush=True)
    with open(os.path.join(out, "verify.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    return rep


def lm_fit(pr, r_obs_fn, theta0, iters=60):
    """Levenberg-Marquardt on ||r(theta) - offset||^2, numpy linear algebra."""
    th = theta0.clone()
    lam = 1e-6
    cur = r_obs_fn(th)
    c2 = float((cur ** 2).sum())
    for _ in range(iters):
        J = torch.func.jacfwd(r_obs_fn)(th).detach().numpy()
        r = cur.detach().numpy()
        JtJ = J.T @ J
        g = J.T @ r
        d = np.diag(JtJ).copy()
        d[d <= 0] = 1e-300
        improved = False
        for _ in range(12):
            step = np.linalg.lstsq(JtJ + lam * np.diag(d), -g, rcond=None)[0]
            th_new = th + torch.as_tensor(step)
            new = r_obs_fn(th_new)
            c_new = float((new ** 2).sum())
            if c_new < c2:
                th, cur, lam = th_new, new, max(lam / 5, 1e-12)
                improved = (c2 - c_new < 1e-9 * max(c2, 1e-300)
                            or float(np.abs(step).max()) < 1e-9)
                c2 = c_new
                break
            lam *= 10
        else:
            break
        if improved:
            break
    return th, c2


def tgn_fit(pr, r_obs_fn, theta0, rtol=RTOL_MAIN, iters=30):
    """Gauss-Newton with the SAME truncation the bound uses: the nuisance
    (net) block is solved in the span of its singular vectors above
    rtol*s_max, the mechanistic block against the projected residual. This is
    the estimator the profiled CRLB is the bound OF (backtracking line search)."""
    th = theta0.clone()
    cur = r_obs_fn(th)
    c2 = float((cur ** 2).sum())
    n = pr.n_mech
    for _ in range(iters):
        J = torch.func.jacfwd(r_obs_fn)(th).detach().numpy()
        r = cur.detach().numpy()
        Jm, Jp = J[:, :n], J[:, n:]
        if Jp.shape[1]:
            U, sv, Vt = np.linalg.svd(Jp, full_matrices=False)
            k = int((sv > rtol * sv[0]).sum())
            Q = U[:, :k]
            dm = np.linalg.lstsq(Jm - Q @ (Q.T @ Jm), -(r - Q @ (Q.T @ r)),
                                 rcond=None)[0]
            rhs = -(r + Jm @ dm)
            dp = Vt[:k].T @ ((Q.T @ rhs) / sv[:k])
            step = np.concatenate([dm, dp])
        else:
            step = np.linalg.lstsq(Jm, -r, rcond=None)[0]
        t, done = 1.0, True
        for _ in range(20):
            th_new = th + torch.as_tensor(t * step)
            new = r_obs_fn(th_new)
            c_new = float((new ** 2).sum())
            if c_new <= c2:
                done = (c2 - c_new < 1e-12 * max(c2, 1e-300))
                th, cur, c2 = th_new, new, c_new
                break
            t *= 0.5
        if done:
            break
    return th, c2


_MC_CTX = None
MC_RTOL = RTOL_MAIN


def _mc_one(noise):
    torch.set_num_threads(1)
    pr, r_hat0, th_hat0 = _MC_CTX
    off = r_hat0 + torch.as_tensor(noise)

    def robs(th):
        return pr.resid(th) - off
    th_lm, _ = lm_fit(pr, robs, th_hat0)
    th_gn, _ = tgn_fit(pr, robs, th_hat0, rtol=MC_RTOL)
    return (float(th_lm[pr.ia] - th_hat0[pr.ia]),
            float(th_gn[pr.ia] - th_hat0[pr.ia]))


def _sd_ci(est):
    from scipy.stats import chi2
    n = len(est)
    sd = float(np.std(est, ddof=1))
    return sd, [float(sd * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1))),
                float(sd * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1)))]


def monte_carlo(pr, n_rep=200, target_sd=0.01, sigma_cap=1e-3, seed=0):
    """Data generated FROM the family at its best fit theta_hat0 to the exact
    data (correctly specified model), plus iid N(0, sigma^2) on every
    residual. sigma = min(target_sd/S, sigma_cap) keeps the noise small
    against the residual signal (linear regime). Two estimators per replicate:
    truncated Gauss-Newton (the bound's own projection rule) and damped LM
    (the practical one). An rtol scan of the bound at theta_hat0 shows which
    truncation the LM estimator effectively realises."""
    th_hat0, _ = tgn_fit(pr, pr.resid, pr.theta0, iters=100)
    r_hat0 = pr.resid(th_hat0).detach()
    J = pr.jac(th_hat0)
    S, _ = profiled_sd(J, pr.n_mech, pr.ia)
    sigma = min(target_sd / S, sigma_cap)
    scan = {f"{rt:g}": profiled_sd(J, pr.n_mech, pr.ia, rt)[0] * sigma
            for rt in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                       1e-10)}
    rng = np.random.default_rng(seed)
    noises = [rng.normal(0.0, sigma, size=tuple(r_hat0.shape))
              for _ in range(n_rep)]
    global _MC_CTX
    _MC_CTX = (pr, r_hat0, th_hat0)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(8) as pool:     # 8 cores max
        est = np.asarray(pool.map(_mc_one, noises))
    sd_lm, ci_lm = _sd_ci(est[:, 0])
    sd_gn, ci_gn = _sd_ci(est[:, 1])
    return dict(n_rep=n_rep, sigma=float(sigma), S_at_hat0=float(S),
                crlb_sd=float(S * sigma),
                emp_sd_tgn=sd_gn, emp_sd_tgn_ci95=ci_gn,
                emp_mean_tgn=float(est[:, 1].mean()),
                emp_sd_lm=sd_lm, emp_sd_lm_ci95=ci_lm,
                emp_mean_lm=float(est[:, 0].mean()),
                crlb_sd_by_rtol=scan,
                theta_a_hat0=float(th_hat0[pr.ia]))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="one regime, base10/dep/info only (smoke test)")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    workers = min(args.workers, 8)
    t0 = time.time()
    refs = solve_all(out, workers)
    print(f"refs ready [{time.time()-t0:.0f}s]", flush=True)

    if args.verify:
        verify(out, refs)
        print(f"verify done [{time.time()-t0:.0f}s]")
        return

    fams = ("mech", "lin", "gated", "sc")
    tasks = []
    D = designs()
    regimes = REGIME_LIST[:1] if args.quick else REGIME_LIST
    for rg in regimes:
        for dname, conds in D.items():
            if args.quick and dname not in ("base10", "dep", "info"):
                continue
            for e in EDGES:
                sub = {fkey(c["forcing"]): refs[(rg, fkey(c["forcing"]))]
                       for c in conds}
                tasks.append((rg, dname, conds, e, sub, fams))
    if not args.quick:
        for rg in DOSE_REGIMES:
            for dname, conds in dose_designs().items():
                arm = dname.split("_")[1]
                sub = {fkey(c["forcing"]): refs[(rg, fkey(c["forcing"]))]
                       for c in conds}
                tasks.append((rg, dname, conds, DOSE_EDGE[arm], sub, fams))
    print(f"{len(tasks)} FIM cases on {workers} workers", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(evaluate_case, tasks, chunksize=1)):
            results.append(r)
            if i % 20 == 0 or i == len(tasks) - 1:
                print(f"  [{i+1}/{len(tasks)}] {r['regime']:<17s} "
                      f"{r['design']:<16s} {r['edge']:<7s} rho={r['rho']:.4f}"
                      f" S_gated={r['S'].get('gated', np.nan):.3g} "
                      f"S_mech={r['S'].get('mech', np.nan):.3g}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
            if i % 50 == 0:
                with open(os.path.join(out, "results.json"), "w") as fh:
                    json.dump(results, fh, indent=1)
    meta = dict(stride=STRIDE, sigma_ref=SIGMA_REF, rtols=RTOLS,
                rtol_main=RTOL_MAIN, net_seeds=NET_SEEDS, lin_K=LIN_K,
                edges=EDGES, regimes=REGIME_LIST, dose_regimes=DOSE_REGIMES,
                residual="r=[(z_{i+1}-z_i) - dt/2 (F(i+1)+F(i))]/s_c, "
                         "s_c=max(max|x_k|,0.05) per condition; "
                         "r ~ iid N(0, sigma_r^2)",
                S="sd(log basal)/sigma_r after profiling all other "
                  "parameters of the equation and of f")
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(dict(meta=meta, results=results), fh, indent=1)
    if args.quick:
        for r in results:
            print(r["regime"], r["design"], r["edge"], f"{r['rho']:.4f}",
                  {k: f"{v:.3g}" for k, v in r["S"].items()})
        return
    corr = correlate(results, load_measured())
    with open(os.path.join(out, "correlations.json"), "w") as fh:
        json.dump(corr, fh, indent=1)
    write_summary(out, results, corr)
    print(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
