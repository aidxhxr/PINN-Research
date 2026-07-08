"""Bayesian FORWARD PINN: posterior over the NEURAL-NETWORK WEIGHTS given sparse
noisy observations + the known physics, i.e. uncertainty quantification on the
reconstructed trajectory.

Relationship to PINN-bayesian (the inverse B-PINN)
--------------------------------------------------
`PINN-bayesian/bayesian_infer.py` is the INVERSE B-PINN: the state nets are
frozen and HMC samples the posterior over the 36 unknown ODE PARAMETERS; the
deliverable is a per-parameter marginal (a wide marginal = non-identifiable).

This module is the FORWARD twin, following the same B-PINN framework (Yang,
Meng & Karniadakis 2020, arXiv:2003.06097) but flipping what is known and what
is sampled:

  * the ODE PARAMETERS are KNOWN (the forward problem);
  * the NETWORK WEIGHTS are the unknown, and HMC samples their posterior
        P(w | sparse data, physics)  proportional to  exp(-U(w))
  * the deliverable is the POSTERIOR-PREDICTIVE TRAJECTORY: for each state, a
    mean and a 95% credible band. Where data is sparse and the physics weakly
    constrains a state, the band widens — an honest, calibrated statement of how
    well the sparse-data forward solve is pinned down.

It is the Bayesian version of the honest sparse-data forward solve in
`PINN-smaller/forward-pinn-train-hybrid` (the point estimate): same ~40
scattered noisy observations + IC anchor + physics residual, now returning a
posterior instead of a single fit.

Design choices carried over from the rest of the study
------------------------------------------------------
  * DERIVATIVE-FREE trapezoidal integral residual (residual.physics_rhs, net
    VALUES only). Avoids the biased autodiff dz/dt (the PINN accuracy ceiling)
    AND makes U a single-backward function of the weights, so HMC over the
    ~10^4 weights is affordable.
  * A deliberately SMALL net (width 64 x depth 3 ~ 11k weights) so the weight
    posterior is samplable; the fixed Fourier features carry the fast forcing
    without spending sampled dofs.
  * CALIBRATED noise scales. The inverse run's first attempt miscalibrated
    sigma_res against a 210k-term residual sum and collapsed the posterior to a
    stuck pinpoint (see notes/2026-07-07-bayesian-pinn.md). Here every noise
    scale is set to the residual RMS at the MAP (reduced chi^2 = 1 per term) on
    a MODEST fixed collocation grid, so the physics likelihood cannot blow up
    with collocation count.
  * DIAGONAL-metric HMC (Stan-style diag mass adaptation) + dual-averaging step
    size. A dense metric (used by the 36-param inverse) is infeasible at ~10^4
    dims; the diagonal metric is the standard high-dimensional choice, and the
    PREDICTIVE functionals mix far faster than the raw redundant weights.

Usage (one regime; run_bayes_forward.sh fans the 4 regimes out in parallel):
    python3 bayesian_forward.py --regime "Normal" --out <run_dir> \
        --n-data 40 --noise-std 0.02 --warmup 400 --draws 600
"""
import argparse
import json
import math
import os
import time as wall

import numpy as np
import torch
from torch.func import functional_call

from config import BASELINE, REGIMES, DEVICE, VAR_NAMES, VAR_LABELS, Y0
from model import ForwardPINN
from reference import generate_references
from residual import physics_rhs


# ----------------------------------------------------------------------
# Sparse noisy observations (mirrors the forward-hybrid data model).
# ----------------------------------------------------------------------
def sample_sparse(t_ref, y_ref, n_data, noise_std, seed):
    """`n_data` scattered observation times (t=0 always included) with optional
    Gaussian measurement noise. The physics residual fills the long gaps."""
    rng = np.random.default_rng(seed)
    pool = np.arange(1, len(t_ref))
    pick = np.sort(rng.choice(pool, size=n_data - 1, replace=False))
    idx = np.concatenate([[0], pick])
    t_obs = t_ref[idx]
    y_obs = y_ref[idx].copy()
    if noise_std > 0:
        y_obs = y_obs + rng.normal(0.0, noise_std, size=y_obs.shape)
    return t_obs, y_obs


# ----------------------------------------------------------------------
# Flat-weight <-> functional network forward (so U is differentiable in w).
# ----------------------------------------------------------------------
class FlatNet:
    """Wraps a ForwardPINN so it can be evaluated from a FLAT weight vector w,
    differentiably. The Fourier B and out_scale buffers are fixed and carried
    along; only the Linear weights/biases are sampled."""

    def __init__(self, net):
        self.net = net
        self.pnames = [k for k, _ in net.named_parameters()]
        self.shapes = [tuple(v.shape) for _, v in net.named_parameters()]
        self.numels = [v.numel() for _, v in net.named_parameters()]
        self.buffers = {k: v for k, v in net.named_buffers()}
        self.dim = int(sum(self.numels))

    def flat0(self):
        with torch.no_grad():
            return torch.cat([v.reshape(-1)
                              for _, v in self.net.named_parameters()]).clone()

    def unflatten(self, w):
        out, i = {}, 0
        for name, sh, n in zip(self.pnames, self.shapes, self.numels):
            out[name] = w[i:i + n].reshape(sh)
            i += n
        return out

    def forward(self, w, t):
        params = {**self.unflatten(w), **self.buffers}
        return functional_call(self.net, params, (t,))


# ----------------------------------------------------------------------
# Build the per-regime data / collocation tensors.
# ----------------------------------------------------------------------
def load_regime(regime, T, n_colloc, n_data, noise_std, seed,
                width, depth, n_fourier, fourier_sigma):
    refs = generate_references(T=T, n_pts=5000)          # fork pool: pre-CUDA
    t_ref, y_ref = refs[regime]
    p = {**BASELINE, **REGIMES[regime]}

    scale = np.maximum(np.abs(y_ref).max(axis=0), 0.05)  # (7,) per-state scale
    w_state = torch.tensor((1.0 / scale), device=DEVICE).reshape(1, 7)

    t_obs, y_obs = sample_sparse(t_ref, y_ref, n_data, noise_std, seed)
    t_d = torch.tensor(t_obs[:, None], device=DEVICE)
    y_d = torch.tensor(y_obs, device=DEVICE)

    t_ic = torch.zeros(1, 1, device=DEVICE)
    y_ic = torch.tensor(Y0[None, :], device=DEVICE)

    t_col = torch.linspace(0, T, n_colloc, device=DEVICE).reshape(-1, 1)
    dt = (t_col[1:] - t_col[:-1])

    net = ForwardPINN(T_max=T, width=width, depth=depth, n_fourier=n_fourier,
                      fourier_sigma=fourier_sigma,
                      out_scale=torch.tensor(scale)).to(DEVICE)

    data = dict(p=p, w=w_state, t_d=t_d, y_d=y_d, t_ic=t_ic, y_ic=y_ic,
                t_col=t_col, dt=dt, t_obs=t_obs, y_obs=y_obs,
                t_ref=t_ref, y_ref=y_ref, scale=scale)
    n_terms = dict(data=t_d.shape[0] * 7, ic=7, phys=(n_colloc - 1) * 7)
    return net, data, n_terms


# ----------------------------------------------------------------------
# Weighted sum-of-squares for each likelihood term (differentiable in w).
# ----------------------------------------------------------------------
def make_terms(flatnet, data):
    p, w = data["p"], data["w"]
    t_d, y_d = data["t_d"], data["y_d"]
    t_ic, y_ic = data["t_ic"], data["y_ic"]
    t_col, dt = data["t_col"], data["dt"]

    def ssr_data(wv):
        z = flatnet.forward(wv, t_d)
        return (((z - y_d) * w) ** 2).sum()

    def ssr_ic(wv):
        z = flatnet.forward(wv, t_ic)
        return (((z - y_ic) * w) ** 2).sum()

    def ssr_phys(wv):
        z = flatnet.forward(wv, t_col)
        f = physics_rhs(t_col, z, p)
        r = ((z[1:] - z[:-1]) - 0.5 * dt * (f[1:] + f[:-1])) * w
        return (r ** 2).sum()

    return ssr_data, ssr_ic, ssr_phys


# ----------------------------------------------------------------------
# MAP pretraining (Adam) — good HMC init + the anchor for noise calibration.
# ----------------------------------------------------------------------
def pretrain_map(net, flatnet, data, terms, *, epochs, lr, lam_data, lam_ic,
                 lam_phys, tag, log_every=500):
    ssr_data, ssr_ic, ssr_phys = terms
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)
    t0 = wall.perf_counter()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        wv = torch.cat([q.reshape(-1) for q in net.parameters()])
        L = lam_data * ssr_data(wv) + lam_ic * ssr_ic(wv) + lam_phys * ssr_phys(wv)
        L.backward()
        opt.step()
        sched.step()
        if ep % log_every == 0 or ep == 1:
            dt = wall.perf_counter() - t0
            print(f"    [{tag}] MAP {ep:>5}/{epochs}  L={float(L.detach()):.3e}  "
                  f"lr={sched.get_last_lr()[0]:.1e}  [{dt:.0f}s]", flush=True)
    return flatnet.flat0().detach()


# ----------------------------------------------------------------------
# Potential energy U(w) = weighted physics/data/IC neg-log-lik + weight prior.
# ----------------------------------------------------------------------
def make_potential(terms, sig, sigma_w):
    ssr_data, ssr_ic, ssr_phys = terms
    inv2_d = 1.0 / (2.0 * sig["data"] ** 2)
    inv2_i = 1.0 / (2.0 * sig["ic"] ** 2)
    inv2_p = 1.0 / (2.0 * sig["phys"] ** 2)
    inv2_w = 1.0 / (2.0 * sigma_w ** 2)

    def potential(w):
        return (ssr_data(w) * inv2_d + ssr_ic(w) * inv2_i
                + ssr_phys(w) * inv2_p + (w ** 2).sum() * inv2_w)

    return potential


def u_and_grad(potential, q):
    q = q.detach().requires_grad_(True)
    u = potential(q)
    g, = torch.autograd.grad(u, q)
    return u.detach(), g.detach()


# ----------------------------------------------------------------------
# Diagonal-metric HMC (Stan-style diag mass adaptation) + dual averaging.
#   metric  M^{-1} = diag(var)   (estimated posterior variance)
#   momentum p ~ N(0, M),  kinetic 0.5 sum(p^2 * var),  dq = var * p
# A dense metric is infeasible at ~10^4 dims; the diagonal metric is the
# standard high-dimensional choice.
# ----------------------------------------------------------------------
class DiagMetric:
    def __init__(self, dim):
        self.var = torch.ones(dim, device=DEVICE)      # M^{-1}

    def set_var(self, var, floor=1e-8):
        self.var = torch.clamp(var, min=floor)

    def sample_p(self):
        return torch.randn(self.var.shape[0], device=DEVICE) / self.var.sqrt()

    def kinetic(self, p):
        return 0.5 * (p * p * self.var).sum()

    def dq(self, p):
        return self.var * p


def leapfrog(potential, metric, q, p, eps, L):
    _, g = u_and_grad(potential, q)
    for _ in range(L):
        p = p - 0.5 * eps * g
        q = q + eps * metric.dq(p)
        _, g = u_and_grad(potential, q)
        p = p - 0.5 * eps * g
    return q, p


def find_reasonable_eps(potential, metric, q, eps=0.1):
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


def hmc_run(potential, q0, *, warmup, draws, L, target_accept, log_every,
            tag, rng_seed):
    torch.manual_seed(rng_seed)
    dim = q0.shape[0]
    metric = DiagMetric(dim)
    q = q0.clone()
    eps = find_reasonable_eps(potential, metric, q, eps=0.1)
    print(f"    [{tag}] dim={dim}  eps0={eps:.4g}", flush=True)

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

    def one_step(q, eps):
        p0 = metric.sample_p()
        u0, _ = u_and_grad(potential, q)
        H0 = u0 + metric.kinetic(p0)
        Lj = max(3, int(L + torch.randint(-3, 4, (1,)).item()))
        q_new, p_new = leapfrog(potential, metric, q, p0, eps, Lj)
        u1, _ = u_and_grad(potential, q_new)
        H1 = u1 + metric.kinetic(p_new)
        if not (torch.isfinite(u1) and torch.isfinite(H1)):
            return q, 0.0, float(u0)
        a = float(torch.exp(torch.clamp(H0 - H1, max=0.0)))
        if float(torch.rand(1)) < a:
            return q_new, a, float(u1)
        return q, a, float(u0)

    # warmup: 20% step-size only, 50% (two windows) diag-variance adaptation,
    # 30% step-size-only finalise.
    w_init = max(25, int(0.20 * warmup))
    w_adaptA = max(25, int(0.30 * warmup))
    w_adaptB = max(25, int(0.20 * warmup))
    w_final = max(25, warmup - w_init - w_adaptA - w_adaptB)
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
        if adapt_metric and len(buf) >= 10:
            S = torch.stack(buf[len(buf) // 2:])
            var = S.var(dim=0)
            # blend with identity for robustness on short windows
            metric.set_var(0.9 * var + 0.1 * metric.var.mean())
            eps = find_reasonable_eps(potential, metric, q, eps=eps)
        return q, eps

    q, eps = warm_window(q, w_init, eps, False, "init")
    q, eps = warm_window(q, w_adaptA, eps, True, "varA")
    q, eps = warm_window(q, w_adaptB, eps, True, "varB")
    q, eps = warm_window(q, w_final, eps, False, "fine")
    print(f"    [{tag}] warmup done: eps={eps:.4f}  "
          f"mean-acc={np.mean(accepts):.2f}", flush=True)

    samples, us, sacc = [], [], []
    for it in range(1, draws + 1):
        q, a, u = one_step(q, eps)
        samples.append(q.detach().clone())
        us.append(u)
        sacc.append(a)
        if it % log_every == 0:
            dt = wall.perf_counter() - t0
            print(f"    [{tag}] draw {it:>5}/{draws}  "
                  f"acc={np.mean(sacc[-log_every:]):.2f}  U={u:.1f}  "
                  f"[{dt:.0f}s]", flush=True)
    S = torch.stack(samples)
    return S, dict(eps=eps, accept=float(np.mean(sacc)), U=np.array(us))


# ----------------------------------------------------------------------
# ESS (Geyer initial monotone sequence) — cheap 1-D diagnostic.
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
# Posterior predictive trajectory band + diagnostics.
# ----------------------------------------------------------------------
def predictive(flatnet, S, data, T, n_eval, max_draws):
    t_eval = torch.linspace(0, T, n_eval, device=DEVICE).reshape(-1, 1)
    idx = np.linspace(0, S.shape[0] - 1,
                      min(max_draws, S.shape[0])).astype(int)
    preds = np.empty((len(idx), n_eval, 7))
    with torch.no_grad():
        for j, k in enumerate(idx):
            preds[j] = flatnet.forward(S[k], t_eval).cpu().numpy()
    te = t_eval.cpu().numpy().ravel()
    mean = preds.mean(0)
    lo = np.percentile(preds, 2.5, axis=0)
    hi = np.percentile(preds, 97.5, axis=0)
    # reference interpolated onto the eval grid
    ref = np.stack([np.interp(te, data["t_ref"], data["y_ref"][:, i])
                    for i in range(7)], axis=1)
    return dict(t=te, preds=preds, mean=mean, lo=lo, hi=hi, ref=ref)


def summarise(pred, info, meta, out_dir, safe):
    te, mean, lo, hi, ref = (pred["t"], pred["mean"], pred["lo"],
                             pred["hi"], pred["ref"])
    scale = np.maximum(np.abs(ref).max(axis=0), 0.05)
    per_state = {}
    for i, vn in enumerate(VAR_NAMES):
        rmse = float(np.sqrt(np.mean((mean[:, i] - ref[:, i]) ** 2)))
        rel_rmse = rmse / scale[i]
        cover = float(np.mean((ref[:, i] >= lo[:, i]) & (ref[:, i] <= hi[:, i])))
        width = float(np.mean(hi[:, i] - lo[:, i]))
        rel_width = width / scale[i]
        per_state[vn] = dict(rmse=rmse, rel_rmse=rel_rmse,
                             coverage95=cover, band_width=width,
                             rel_band_width=rel_width)
    overall = dict(
        mean_rel_rmse=float(np.mean([per_state[v]["rel_rmse"] for v in VAR_NAMES])),
        mean_coverage95=float(np.mean([per_state[v]["coverage95"] for v in VAR_NAMES])),
        mean_rel_band_width=float(np.mean([per_state[v]["rel_band_width"] for v in VAR_NAMES])))

    # predictive-functional ESS (mean of beta-catenin over time, per draw) — a
    # meaningful mixing check (redundant weights mix slower than functionals).
    f_beta = pred["preds"][:, :, 0].mean(axis=1)
    ess_pred = ess_1d(f_beta)
    ess_U = ess_1d(info["U"])

    meta_json = {k: v for k, v in meta.items() if not k.startswith("_")}
    summary = dict(regime=safe, meta=meta_json, per_state=per_state,
                   overall=overall, ess_pred_beta=ess_pred, ess_U=ess_U,
                   accept=info["accept"], eps=info["eps"])
    with open(os.path.join(out_dir, f"{safe}_predictive_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, f"{safe}_predictive.npz"),
        t=te, mean=mean, lo=lo, hi=hi, ref=ref,
        t_obs=meta["_t_obs"], y_obs=meta["_y_obs"],
        var_names=np.array(VAR_NAMES))
    return summary


def plot_predictive(pred, data, summary, out_dir, safe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    te, mean, lo, hi, ref = (pred["t"], pred["mean"], pred["lo"],
                             pred["hi"], pred["ref"])
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for i, (vn, lab) in enumerate(zip(VAR_NAMES, VAR_LABELS)):
        ax = axes.flat[i]
        ax.fill_between(te, lo[:, i], hi[:, i], color="#4c72b0", alpha=0.30,
                        label="95% credible band")
        ax.plot(te, mean[:, i], color="#1f4e9c", lw=1.6, label="post. mean")
        ax.plot(te, ref[:, i], color="crimson", lw=1.4, ls="--", label="truth")
        ax.scatter(data["t_obs"], data["y_obs"][:, i], s=16, color="k",
                   zorder=5, label="noisy obs")
        ps = summary["per_state"][vn]
        ax.set_title(f"{lab}   cov95={ps['coverage95']:.2f}  "
                     f"relRMSE={ps['rel_rmse']:.2f}", fontsize=10)
        ax.set_xlabel("t")
        if i == 0:
            ax.legend(fontsize=7, loc="best")
    for j in range(7, 9):
        axes.flat[j].axis("off")
    ov = summary["overall"]
    fig.suptitle(f"{safe} — Bayesian forward PINN posterior predictive   "
                 f"[mean cov95={ov['mean_coverage95']:.2f}, "
                 f"mean relRMSE={ov['mean_rel_rmse']:.2f}]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(out_dir, f"{safe}_predictive.png"), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=list(REGIMES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=float, default=150.0)
    ap.add_argument("--n-data", type=int, default=40)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--colloc", type=int, default=800)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-fourier", type=int, default=16)
    ap.add_argument("--fourier-sigma", type=float, default=4.0)
    ap.add_argument("--map-epochs", type=int, default=4000)
    ap.add_argument("--map-lr", type=float, default=2e-3)
    ap.add_argument("--lam-data", type=float, default=1.0)
    ap.add_argument("--lam-ic", type=float, default=20.0)
    ap.add_argument("--lam-phys", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--leapfrog", type=int, default=25)
    ap.add_argument("--target-accept", type=float, default=0.75)
    ap.add_argument("--sigma-w", type=float, default=3.0,
                    help="Gaussian weight-prior scale (B-PINN prior on weights)")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--max-pred-draws", type=int, default=400)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    safe = args.regime.replace(" ", "_").replace("/", "_")
    print(f"=== Bayesian FORWARD PINN — regime={args.regime}  "
          f"device={DEVICE} ===", flush=True)

    net, data, n_terms = load_regime(
        args.regime, args.T, args.colloc, args.n_data, args.noise_std,
        args.seed, args.width, args.depth, args.n_fourier, args.fourier_sigma)
    flatnet = FlatNet(net)
    terms = make_terms(flatnet, data)
    print(f"  net dim={flatnet.dim:,}  obs={args.n_data} "
          f"(noise={args.noise_std})  colloc={args.colloc}", flush=True)

    # --- MAP pretraining: HMC init + calibration anchor ---
    q_map = pretrain_map(net, flatnet, data, terms,
                         epochs=args.map_epochs, lr=args.map_lr,
                         lam_data=args.lam_data, lam_ic=args.lam_ic,
                         lam_phys=args.lam_phys, tag=safe[:4])

    # --- calibrate the three noise scales to reduced chi^2 = 1 at the MAP ---
    # (each sigma = RMS of that term's weighted residual; floored so clean-data
    # terms don't collapse to a delta). This is the fix for the inverse run's
    # collocation-count miscalibration: sigma tracks the residual, never fights
    # a growing sum.
    with torch.no_grad():
        ssr_data, ssr_ic, ssr_phys = terms
        sig = dict(
            data=max(float(np.sqrt(float(ssr_data(q_map)) / n_terms["data"])), 1e-3),
            ic=max(float(np.sqrt(float(ssr_ic(q_map)) / n_terms["ic"])), 1e-3),
            phys=max(float(np.sqrt(float(ssr_phys(q_map)) / n_terms["phys"])), 1e-4))
    print(f"  calibrated sigma: data={sig['data']:.3e}  ic={sig['ic']:.3e}  "
          f"phys={sig['phys']:.3e}", flush=True)

    potential = make_potential(terms, sig, args.sigma_w)

    S, info = hmc_run(potential, q_map, warmup=args.warmup, draws=args.draws,
                      L=args.leapfrog, target_accept=args.target_accept,
                      log_every=100, tag=safe[:4], rng_seed=args.seed + 7)

    pred = predictive(flatnet, S, data, args.T, args.n_eval, args.max_pred_draws)
    meta = dict(regime=safe, n_data=args.n_data, noise_std=args.noise_std,
                colloc=args.colloc, net_dim=flatnet.dim, width=args.width,
                depth=args.depth, warmup=args.warmup, draws=args.draws,
                leapfrog=args.leapfrog, sigma_w=args.sigma_w,
                sigma_data=sig["data"], sigma_ic=sig["ic"], sigma_phys=sig["phys"],
                _t_obs=data["t_obs"], _y_obs=data["y_obs"])
    summary = summarise(pred, info, meta, args.out, safe)
    plot_predictive(pred, data, summary, args.out, safe)

    ov = summary["overall"]
    print(f"\n  [{safe}] accept={info['accept']:.2f}  eps={info['eps']:.4f}  "
          f"ess(U)={summary['ess_U']:.0f}  ess(pred)={summary['ess_pred_beta']:.0f}",
          flush=True)
    print(f"  [{safe}] mean cov95={ov['mean_coverage95']:.2f}  "
          f"mean relRMSE={ov['mean_rel_rmse']:.3f}  "
          f"mean rel band width={ov['mean_rel_band_width']:.3f}", flush=True)
    for vn in VAR_NAMES:
        ps = summary["per_state"][vn]
        print(f"    {vn:>4}: cov95={ps['coverage95']:.2f}  "
              f"relRMSE={ps['rel_rmse']:.3f}  relBand={ps['rel_band_width']:.3f}",
              flush=True)
    print(f"  [{safe}] DONE", flush=True)


if __name__ == "__main__":
    main()
