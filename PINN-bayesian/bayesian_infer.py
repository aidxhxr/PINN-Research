"""Bayesian inverse PINN (Tier 1): posterior over the 36 ODE parameters given
the FROZEN pinn-boost state surrogates and the derivative-free integral residual.

Motivation (see notes/2026-07-07-bayesian-pinn.md)
--------------------------------------------------
The `pinn-boost` integral-residual run is the best-working inverse PINN we have
(17/16/10/7 under 10%). Its Stage-3 "parameter refinement on FROZEN nets"
(training.py) is a *point* MAP estimate: it freezes the data-pinned state nets
and minimises ONLY the trapezoidal integral residual over the 36 shared
parameters. This module is the BAYESIAN version of exactly that stage — it
samples the posterior

    P(lambda | frozen nets, physics)  proportional to  exp(-U(lambda))

with the SAME integral residual as the negative log-likelihood, following the
B-PINN framework of Yang, Meng & Karniadakis 2020 (arXiv:2003.06097). Because
the state nets are frozen, the data term is constant in lambda and drops out;
the only lambda-dependent term is the ODE (integral) residual, treated as a
Gaussian physics likelihood with scale sigma_res (the "f-sensor" noise in the
B-PINN paper). Sampling is by a self-contained Hamiltonian Monte Carlo with a
DENSE mass matrix — necessary because the model is sloppy (the companion FIM
run found cond ~ 1e7-1e17), so a diagonal metric would mix terribly along the
correlated eta/kappa directions.

Key deliverable: the marginal posterior of each parameter. A parameter whose
posterior reverts to the prior is (practically) NON-identifiable; a tight
posterior is identifiable. The expected headline is that thetaP's marginal
collapses toward the prior as WNT drive increases (Normal -> Strong APC-mutant)
— the posterior-width restatement of the high-WNT thetaP wall that the profile
likelihood and the two FIM runs already found by other routes.

The integral residual uses net VALUES only (physics_rhs), never the biased
autodiff dz/dt, so this posterior inherits the accuracy lever that broke the
PINN derivative-bias ceiling in the first place — a derivative-free Bayesian
inverse solve.

Usage (one regime; run_bayes.sh fans the 4 regimes out in parallel):
    python3 bayesian_infer.py --regime "Normal" \
        --seed-run /abs/path/to/runs/20260702_160102_integral \
        --out <run_dir> --warmup 800 --draws 1200 --colloc 4000
"""
import argparse
import json
import math
import os
import time as wall

import numpy as np
import torch

from config import (BASELINE, REGIMES, CONDITIONS, DEVICE, VAR_NAMES,
                    VAR_LABELS, UNKNOWN, PARAM_RANGE, NOMINAL)
from model import ForwardPINN
from reference import generate_references
from residual import physics_rhs


# ----------------------------------------------------------------------
# Parameter transform: flat raw vector s (R^36) <-> constrained values.
# Mirrors model.InverseParams.dict():
#   positive params  value = nominal * exp(s)          (log/relative)
#   thetaP in (0,1)  value = sigmoid(s)                (bounded)
# Working in s-space makes the prior a clean isotropic-ish Gaussian and the
# HMC geometry scale-invariant across the 0.08..3.5 span of the unknowns.
# ----------------------------------------------------------------------
# NOTE: kept on CPU at import so that merely importing this module does not
# initialise a CUDA context. generate_references() forks a scipy process pool,
# and a fork AFTER CUDA init can deadlock — so CUDA must first be touched only
# later (in load_conditions, after the references are built). _ensure_device()
# moves these onto DEVICE at that point.
_LOG_MASK = torch.tensor(
    [PARAM_RANGE[k][1] is None for k in UNKNOWN])                  # True => log
_NOMINAL = torch.tensor([float(NOMINAL[k]) for k in UNKNOWN])


def _ensure_device(device):
    global _LOG_MASK, _NOMINAL
    if _LOG_MASK.device != device:
        _LOG_MASK = _LOG_MASK.to(device)
        _NOMINAL = _NOMINAL.to(device)


def raw_to_values(s):
    """s: (...,36) tensor -> (...,36) tensor of constrained parameter values."""
    log_vals = _NOMINAL * torch.exp(s)
    bnd_vals = torch.sigmoid(s)                 # thetaP: lo=0, hi=1
    return torch.where(_LOG_MASK, log_vals, bnd_vals)


def raw_to_dict(s):
    """s: (36,) -> {param_name: scalar tensor} ready to splice into BASELINE."""
    v = raw_to_values(s)
    return {k: v[i] for i, k in enumerate(UNKNOWN)}


def values_to_raw(values):
    """Inverse map (numpy in, numpy out), for seeding the chain at the MAP."""
    s = np.zeros(len(UNKNOWN))
    for i, k in enumerate(UNKNOWN):
        if PARAM_RANGE[k][1] is None:
            s[i] = np.log(values[k] / NOMINAL[k])
        else:
            y = np.clip(values[k], 1e-6, 1 - 1e-6)
            s[i] = np.log(y / (1.0 - y))
    return s


# ----------------------------------------------------------------------
# Load the frozen state surrogates + build per-condition collocation tensors.
# ----------------------------------------------------------------------
def load_conditions(regime, seed_run, T, n_colloc):
    """Reconstruct + load the frozen ForwardPINN for each of the 10 conditions,
    and precompute the (frozen) state values z, dt and relative weights on a
    fixed collocation grid — everything the integral residual needs that does
    NOT depend on lambda."""
    safe = regime.replace(" ", "_").replace("/", "_")
    refs = generate_references(T=T, n_pts=5000)[regime]     # fork pool: pre-CUDA
    _ensure_device(DEVICE)                                  # now safe to touch CUDA

    t_fixed = torch.linspace(0, T, n_colloc, device=DEVICE).reshape(-1, 1)
    conds = []
    for c in CONDITIONS:
        t_ref, y_ref = refs[c["name"]]
        scale = np.maximum(np.abs(y_ref).max(axis=0), 0.05)          # (7,)
        net = ForwardPINN(T_max=T, width=256, depth=4, n_fourier=16,
                          fourier_sigma=4.0, out_scale=scale,
                          activation="gelu").to(DEVICE)
        sd = torch.load(os.path.join(seed_run, f"{safe}_{c['name']}_net.pt"),
                        map_location=DEVICE)
        net.load_state_dict(sd)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            z = net(t_fixed).detach()                                # (N,7) frozen
        w = torch.tensor((1.0 / scale), device=DEVICE).reshape(1, 7)  # rel_weight
        conds.append(dict(name=c["name"], forcing=c["forcing"],
                          t=t_fixed, z=z,
                          dt=(t_fixed[1:] - t_fixed[:-1]),
                          w=w))
    n_resid = sum((c["z"].shape[0] - 1) * 7 for c in conds)
    return conds, n_resid, safe


# ----------------------------------------------------------------------
# Potential energy  U(s) = phys neg-log-lik + neg-log-prior.
# ----------------------------------------------------------------------
def make_potential(conds, sigma_res, sigma_prior_vec):
    inv2_res = 1.0 / (2.0 * sigma_res ** 2)
    inv2_pri = 1.0 / (2.0 * sigma_prior_vec ** 2)

    def sum_sq_residual(s):
        params = raw_to_dict(s)
        ssr = s.new_zeros(())
        for c in conds:
            pp = {**BASELINE, **params, **c["forcing"]}
            f = physics_rhs(c["t"], c["z"], pp)
            r = ((c["z"][1:] - c["z"][:-1]) - 0.5 * c["dt"] * (f[1:] + f[:-1]))
            r = r * c["w"]
            ssr = ssr + (r ** 2).sum()
        return ssr

    def potential(s):
        ssr = sum_sq_residual(s)
        nll = ssr * inv2_res
        nlp = (s ** 2 * inv2_pri).sum()
        return nll + nlp

    return potential, sum_sq_residual


def u_and_grad(potential, q):
    q = q.detach().requires_grad_(True)
    u = potential(q)
    g, = torch.autograd.grad(u, q)
    return u.detach(), g.detach()


# ----------------------------------------------------------------------
# Self-contained HMC with a DENSE mass matrix and dual-averaging step size.
#
#   metric  M^{-1} = Sigma  (estimated posterior covariance, Stan convention)
#   momentum p ~ N(0, M),  kinetic K = 0.5 p^T Sigma p
#   leapfrog q += eps * Sigma @ p
# A dense metric is essential for this sloppy problem: it absorbs the strong
# linear eta/kappa correlations so the effective conditioning HMC sees is far
# below the raw 1e7-1e17, which a diagonal metric could not do.
# ----------------------------------------------------------------------
class DenseMetric:
    def __init__(self, dim):
        self.set_cov(torch.eye(dim, device=DEVICE))

    def set_cov(self, cov, floor_ratio=1e-10):
        # cov = Sigma = M^{-1}. Eigen-floor so BOTH Sigma and M = Sigma^{-1}
        # are positive-definite regardless of rank-deficiency / indefiniteness
        # in the supplied matrix (sample covariances and the full Hessian can
        # be indefinite for this sloppy model).
        cov = 0.5 * (cov + cov.T)
        w, V = torch.linalg.eigh(cov)
        wmax = float(w.max())
        if not (wmax > 0):
            wmax = 1.0
        w = torch.clamp(w, min=wmax * floor_ratio)
        self.cov = (V * w) @ V.T
        self.M = (V * (1.0 / w)) @ V.T
        self.M = 0.5 * (self.M + self.M.T)
        self.L_M = torch.linalg.cholesky(self.M)          # for momentum sampling

    def sample_p(self):
        z = torch.randn(self.cov.shape[0], device=DEVICE)
        return self.L_M @ z

    def kinetic(self, p):
        return 0.5 * (p @ (self.cov @ p))

    def dq(self, p):                                       # Sigma @ p
        return self.cov @ p


def leapfrog(potential, metric, q, p, eps, L):
    _, g = u_and_grad(potential, q)
    for _ in range(L):
        p = p - 0.5 * eps * g
        q = q + eps * metric.dq(p)
        _, g = u_and_grad(potential, q)
        p = p - 0.5 * eps * g
    return q, p


def laplace_cov(potential, q0, curv_floor=0.25):
    """Hessian of U at the MAP -> Laplace posterior covariance Sigma = H^{-1}.
    Used as the INITIAL dense HMC metric (the posterior is sloppy, cond up to
    1e17, so identity-metric HMC diverges instantly; the Laplace metric starts
    the sampler in the right geometry). Also a standalone Gaussian cross-check
    of the marginals, echoing the companion FIM run.

    The FULL Hessian is indefinite here (the residual is not exactly zero at
    the MAP, so the second-derivative terms can dominate in sloppy directions).
    We eigen-floor the curvature at `curv_floor` (~ the prior curvature
    1/sigma_prior^2) before inverting, which caps a sloppy direction's Laplace
    variance at the prior scale — exactly the "reverts to prior" behaviour we
    want the metric to reflect."""
    dim = q0.shape[0]
    q = q0.detach().clone().requires_grad_(True)
    u = potential(q)
    g, = torch.autograd.grad(u, q, create_graph=True)
    H = torch.zeros(dim, dim, device=q0.device, dtype=q0.dtype)
    for i in range(dim):
        Hi, = torch.autograd.grad(g[i], q, retain_graph=True)
        H[i] = Hi.detach()
    H = 0.5 * (H + H.T)
    w, V = torch.linalg.eigh(H)
    w_floored = torch.clamp(w, min=curv_floor)
    Sigma = (V * (1.0 / w_floored)) @ V.T
    return 0.5 * (Sigma + Sigma.T), H.detach()


def find_reasonable_eps(potential, metric, q, eps=0.1):
    """NUTS Alg. 4: scale eps until a single leapfrog step has ~0.5 accept.
    Robust to divergence (nonfinite -> treat as reject and shrink)."""
    p = metric.sample_p()
    u0, _ = u_and_grad(potential, q)
    H0 = u0 + metric.kinetic(p)

    def acc(eps):
        q1, p1 = leapfrog(potential, metric, q, p, eps, 1)
        u1, _ = u_and_grad(potential, q1)
        H1 = u1 + metric.kinetic(p1)
        lr = float(H0 - H1)
        if not math.isfinite(lr):
            return 0.0
        return math.exp(min(0.0, lr))

    a0 = acc(eps)
    direction = 1.0 if a0 > 0.5 else -1.0
    for _ in range(60):
        a = acc(eps)
        if (direction > 0 and a <= 0.5) or (direction < 0 and a > 0.5):
            break
        eps *= 2.0 ** direction
        if eps < 1e-9 or eps > 1e3:
            break
    return float(eps)


def hmc_run(potential, q0, *, warmup, draws, L, target_accept,
            log_every, tag, rng_seed):
    torch.manual_seed(rng_seed)
    dim = q0.shape[0]
    metric = DenseMetric(dim)
    q = q0.clone()

    # Laplace metric at the MAP: essential for this sloppy posterior so the
    # sampler starts in the right (heavily anisotropic) geometry.
    Sigma0, H_map = laplace_cov(potential, q0)
    metric.set_cov(Sigma0)
    eps = find_reasonable_eps(potential, metric, q, eps=0.5)
    print(f"    [{tag}] Laplace metric set; eps0={eps:.4g}", flush=True)

    class DualAvg:
        def __init__(self, eps0):
            self.mu = np.log(10 * eps0)
            self.logeps_bar = 0.0
            self.H_bar = 0.0
            self.gamma, self.t0, self.kappa = 0.05, 10.0, 0.75
            self.m = 0
        def update(self, accept):
            self.m += 1
            w = 1.0 / (self.m + self.t0)
            self.H_bar = (1 - w) * self.H_bar + w * (target_accept - accept)
            logeps = self.mu - np.sqrt(self.m) / self.gamma * self.H_bar
            eta = self.m ** (-self.kappa)
            self.logeps_bar = eta * logeps + (1 - eta) * self.logeps_bar
            return float(np.exp(logeps))
        def finalize(self):
            return float(np.exp(self.logeps_bar))

    # warmup schedule: 20% init step-size, then two metric-adaptation windows
    # (50%), then a final step-size-only window (30%).
    w_init = max(25, int(0.20 * warmup))
    w_adaptA = max(25, int(0.30 * warmup))
    w_adaptB = max(25, int(0.20 * warmup))
    w_final = max(25, warmup - w_init - w_adaptA - w_adaptB)

    def one_step(q, eps):
        p0 = metric.sample_p()
        u0, _ = u_and_grad(potential, q)
        H0 = u0 + metric.kinetic(p0)
        Lj = int(L + torch.randint(-3, 4, (1,)).item())
        Lj = max(3, Lj)
        q_new, p_new = leapfrog(potential, metric, q, p0, eps, Lj)
        u1, _ = u_and_grad(potential, q_new)
        H1 = u1 + metric.kinetic(p_new)
        if not (torch.isfinite(u1) and torch.isfinite(H1)):
            return q, 0.0, float(u0)                       # divergence -> reject
        a = float(torch.exp(torch.clamp(H0 - H1, max=0.0)))
        if float(torch.rand(1)) < a:
            return q_new, a, float(u1)
        return q, a, float(u0)

    t0 = wall.perf_counter()
    accepts = []

    def warm_window(q, n, eps, adapt_metric, phase):
        da = DualAvg(eps)
        buf = []
        for it in range(1, n + 1):
            q, a, u = one_step(q, eps)
            eps = da.update(a)
            buf.append(q.detach().clone())
            accepts.append(a)
            if it % log_every == 0:
                dt = wall.perf_counter() - t0
                print(f"    [{tag}] warm-{phase} {it:>4}/{n}  eps={eps:.4f}  "
                      f"acc={np.mean(accepts[-log_every:]):.2f}  U={u:.1f} "
                      f"[{dt:.0f}s]", flush=True)
        eps = da.finalize()
        if adapt_metric and len(buf) >= dim + 5:
            S = torch.stack(buf[len(buf) // 2:])            # 2nd half of window
            cov = torch.cov(S.T)
            # blend with the Laplace metric for robustness when a window's
            # sample covariance is rank-deficient / noisy.
            try:
                metric.set_cov(0.7 * cov + 0.3 * Sigma0)
                eps = find_reasonable_eps(potential, metric, q, eps=eps)
            except Exception as e:
                print(f"    [{tag}] metric update skipped ({e})", flush=True)
        return q, eps

    q, eps = warm_window(q, w_init, eps, adapt_metric=False, phase="init")
    q, eps = warm_window(q, w_adaptA, eps, adapt_metric=True, phase="covA")
    q, eps = warm_window(q, w_adaptB, eps, adapt_metric=True, phase="covB")
    q, eps = warm_window(q, w_final, eps, adapt_metric=False, phase="fine")
    print(f"    [{tag}] warmup done: eps={eps:.4f}  "
          f"mean-acc={np.mean(accepts):.2f}", flush=True)

    # ---- sampling (fixed eps + metric) ----
    samples = []
    us = []
    sacc = []
    for it in range(1, draws + 1):
        q, a, u = one_step(q, eps)
        samples.append(q.detach().clone())
        us.append(u)
        sacc.append(a)
        if it % log_every == 0:
            dt = wall.perf_counter() - t0
            print(f"    [{tag}] draw {it:>5}/{draws}  acc={np.mean(sacc[-log_every:]):.2f}"
                  f"  U={u:.1f}  [{dt:.0f}s]", flush=True)
    S = torch.stack(samples)                                # (draws, 36) raw
    return S, dict(eps=eps, accept=float(np.mean(sacc)),
                   metric_cov=metric.cov.detach().cpu().numpy())


# ----------------------------------------------------------------------
# ESS (Geyer initial monotone sequence, per dimension) — a cheap diagnostic.
# ----------------------------------------------------------------------
def ess_1d(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    v = np.var(x)
    if v <= 0:
        return float(n)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (v * n)
    s = 1.0
    for k in range(1, n):
        if acf[k] < 0.05:
            break
        s += 2 * acf[k]
    return float(n / max(s, 1.0))


# ----------------------------------------------------------------------
# Diagnostics + persistence.
# ----------------------------------------------------------------------
def summarise(S_raw, sigma_prior_vec, true_vals, out_dir, safe, meta):
    S_raw_np = S_raw.cpu().numpy()
    V = raw_to_values(S_raw).cpu().numpy()                  # constrained values
    sp = sigma_prior_vec.cpu().numpy()

    summary = {}
    for i, k in enumerate(UNKNOWN):
        raw_i = S_raw_np[:, i]
        val_i = V[:, i]
        post_std_raw = float(raw_i.std())
        # identifiability = how much the posterior shrank vs the prior, in the
        # scale-invariant raw (log/logit) space.
        shrink = post_std_raw / float(sp[i])
        verdict = ("IDENT" if shrink < 0.30 else
                   "WEAK" if shrink < 0.70 else "NON-IDENT")
        tv = true_vals[k]
        lo, med, hi = np.percentile(val_i, [2.5, 50, 97.5])
        covered = bool(lo <= tv <= hi)
        z = float((val_i.mean() - tv) / (val_i.std() + 1e-12))
        summary[k] = dict(true=tv, post_mean=float(val_i.mean()),
                          post_std=float(val_i.std()),
                          ci95=[float(lo), float(hi)], median=float(med),
                          shrink_vs_prior=shrink, verdict=verdict,
                          covers_true=covered, z_truth=z,
                          ess=ess_1d(raw_i))
    with open(os.path.join(out_dir, f"{safe}_posterior_summary.json"), "w") as fh:
        json.dump(dict(regime=safe, meta=meta, params=summary), fh, indent=2)
    np.savez_compressed(os.path.join(out_dir, f"{safe}_posterior_samples.npz"),
                        raw=S_raw_np, values=V, names=np.array(UNKNOWN),
                        true=np.array([true_vals[k] for k in UNKNOWN]),
                        sigma_prior=sp, **{f"meta_{a}": b for a, b in
                                           meta.items() if np.isscalar(b)})
    return summary, V


def plot_marginals(V, summary, true_vals, sigma_prior_vec, out_dir, safe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = sigma_prior_vec.cpu().numpy()
    ncol, nrow = 6, 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 18))
    for i, k in enumerate(UNKNOWN):
        ax = axes.flat[i]
        vals = V[:, i]
        ax.hist(vals, bins=40, density=True, color="#4c72b0", alpha=0.75)
        tv = true_vals[k]
        ax.axvline(tv, color="crimson", lw=2, label="true")
        ax.axvline(summary[k]["post_mean"], color="k", ls="--", lw=1.2,
                   label="post mean")
        col = {"IDENT": "#2ca02c", "WEAK": "#ff7f0e",
               "NON-IDENT": "#d62728"}[summary[k]["verdict"]]
        ax.set_title(f"{k}  [{summary[k]['verdict']}]  "
                     f"shrink={summary[k]['shrink_vs_prior']:.2f}",
                     fontsize=9, color=col)
        ax.tick_params(labelsize=7)
    for j in range(len(UNKNOWN), nrow * ncol):
        axes.flat[j].axis("off")
    fig.suptitle(f"{safe} — posterior marginals (36 params)  "
                 f"[green=identifiable, red=reverts to prior]", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(out_dir, f"{safe}_marginals.png"), dpi=110)
    plt.close(fig)

    # focused W / thetaP panel (the headline pair)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, k in zip(axes, ["W", "thetaP"]):
        i = UNKNOWN.index(k)
        ax.hist(V[:, i], bins=45, density=True, color="#4c72b0", alpha=0.8)
        ax.axvline(true_vals[k], color="crimson", lw=2, label="true")
        ax.set_title(f"{safe}: {k}  [{summary[k]['verdict']}]  "
                     f"post={summary[k]['post_mean']:.3f}"
                     f"±{summary[k]['post_std']:.3f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{safe}_W_thetaP.png"), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=list(REGIMES))
    ap.add_argument("--seed-run", required=True,
                    help="pinn-boost integral run dir with the frozen *_net.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--draws", type=int, default=1200)
    ap.add_argument("--leapfrog", type=int, default=18)
    ap.add_argument("--colloc", type=int, default=4000)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--sigma-prior-log", type=float, default=1.0)
    ap.add_argument("--sigma-prior-theta", type=float, default=2.0)
    ap.add_argument("--sigma-res", type=float, default=None,
                    help="physics-noise scale; default = MAP residual RMS "
                         "(reduced chi^2 = 1 at the MAP)")
    ap.add_argument("--T", type=float, default=150.0)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    safe = args.regime.replace(" ", "_").replace("/", "_")
    print(f"=== Bayesian inverse PINN — regime={args.regime} "
          f"device={DEVICE} ===", flush=True)

    conds, n_resid, safe = load_conditions(args.regime, args.seed_run,
                                           args.T, args.colloc)
    true_p = {**BASELINE, **REGIMES[args.regime]}
    true_vals = {k: true_p[k] for k in UNKNOWN}

    # seed the chain at the pinn-boost MAP lambda
    map_sd = torch.load(os.path.join(args.seed_run, f"{safe}_params.pt"),
                        map_location=DEVICE)
    map_raw = torch.tensor([float(map_sd[f"raw.{k}"]) for k in UNKNOWN],
                           device=DEVICE)

    sigma_prior_vec = torch.tensor(
        [args.sigma_prior_theta if k == "thetaP" else args.sigma_prior_log
         for k in UNKNOWN], device=DEVICE)

    # calibrate sigma_res so reduced chi^2 = 1 at the MAP (unless overridden)
    _, ssr_fn = make_potential(conds, sigma_res=1.0,
                               sigma_prior_vec=sigma_prior_vec)
    with torch.no_grad():
        ssr_map = float(ssr_fn(map_raw))
    sigma_res = args.sigma_res or float(np.sqrt(ssr_map / n_resid))
    print(f"  n_resid={n_resid:,}  SSR@MAP={ssr_map:.3e}  "
          f"sigma_res={sigma_res:.4e}", flush=True)

    potential, _ = make_potential(conds, sigma_res, sigma_prior_vec)

    S_raw, info = hmc_run(potential, map_raw, warmup=args.warmup,
                          draws=args.draws, L=args.leapfrog,
                          target_accept=args.target_accept,
                          log_every=100, tag=safe[:4], rng_seed=args.seed + 7)

    meta = dict(regime=safe, sigma_res=sigma_res, n_resid=n_resid,
                eps=info["eps"], accept=info["accept"],
                warmup=args.warmup, draws=args.draws, leapfrog=args.leapfrog,
                colloc=args.colloc, sigma_prior_log=args.sigma_prior_log,
                sigma_prior_theta=args.sigma_prior_theta)
    summary, V = summarise(S_raw, sigma_prior_vec, true_vals,
                           args.out, safe, meta)
    plot_marginals(V, summary, true_vals, sigma_prior_vec, args.out, safe)

    # console verdict table (W/thetaP + counts)
    nid = sum(1 for k in summary if summary[k]["verdict"] == "IDENT")
    nweak = sum(1 for k in summary if summary[k]["verdict"] == "WEAK")
    nnon = sum(1 for k in summary if summary[k]["verdict"] == "NON-IDENT")
    cov = sum(1 for k in summary if summary[k]["covers_true"])
    print(f"\n  [{safe}] accept={info['accept']:.2f}  "
          f"IDENT={nid} WEAK={nweak} NON-IDENT={nnon} /36  "
          f"CI covers truth: {cov}/36", flush=True)
    for k in ["W", "thetaP"]:
        s = summary[k]
        print(f"    {k:>7}: post={s['post_mean']:.3f}±{s['post_std']:.3f} "
              f"true={s['true']:.3f}  CI95={s['ci95'][0]:.3f}..{s['ci95'][1]:.3f}"
              f"  [{s['verdict']}]  ess={s['ess']:.0f}", flush=True)
    print(f"  [{safe}] DONE", flush=True)


if __name__ == "__main__":
    main()
