"""Tangent PINN (T-PINN): a state network that also carries its parameter
sensitivities, for ODE parameter estimation.

Architecture (one network per experimental condition c):

    N_c(t) -> [ Z(t) in R^7 ,  S(t) in R^{7 x P} ]          (P = 36)
    z(t; x) = Z(t) + S(t) (x - x_ref)                     (tangent coupling)

with x = log(theta / nominal) and hard initial conditions
Z(0) = y0, S(0) = 0 (y0 is known and theta-independent).

Losses (all residuals in derivative-free integral/trapezoid form):

    L_data  = mean_w | z(t_d; x) - y_d |^2
    L_state = mean_w | (z_{k+1}-z_k)/dt - (f_k + f_{k+1})/2 |^2
    L_sens  = mean_w | (S_{k+1}-S_k)/dt - (G_k + G_{k+1})/2 |^2,
              G = (df/dz) S + df/dx       (forward-sensitivity equations,
                                           computed with batched forward-mode
                                           JVPs; f and G at z(t; x), x)

Why: in a standard inverse PINN the state network is free of theta, so it can
fit the data at any theta and the physics residual is the only route by which
theta feels the data ("gradient starvation"). Here a change dx in theta moves
the state by S dx, where S is constrained to be the true sensitivity, so the
data gradient reaches theta as S^T W r -- the Gauss-Newton gradient of the
maximum-likelihood objective. The fixed point at hard physics is the MLE, and
S^T S / sigma^2 is the Fisher information: per-parameter Cramer-Rao standard
deviations come out of the trained network.

Stages
  1. joint  : Adam on (nets, x) over L_data + lam (L_state + L_sens), with a
              geometric continuation lam: lam0 -> lam1 (smoothing -> shooting).
  2. gn     : profiled Levenberg-Marquardt. Nets are trained physics-only at
              fixed x; then x <- x + delta with
              delta = argmin |W(y - z - S delta)|^2 + mu |D delta|^2;
              the tangent coupling warm-starts the state at the new x.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import jvp, vmap

import bench
from wnt_pinn.model.parameters import BASELINE, UNKNOWN
from wnt_pinn.model.integral_rhs import physics_rhs
from wnt_pinn.networks.state import ForwardPINN

torch.set_default_dtype(torch.float64)
P = len(UNKNOWN)
NOM = torch.tensor([BASELINE[k] for k in UNKNOWN])


def make_params(x, forcing):
    th = NOM.to(x.device) * torch.exp(x)
    p = {**BASELINE, **forcing}
    for i, k in enumerate(UNKNOWN):
        p[k] = th[i]
    return p


class TangentNet(torch.nn.Module):
    """Fourier-feature MLP with a (7 + 7P)-wide head and hard ICs."""

    def __init__(self, y0, scale, T, width, depth, n_fourier, sigma):
        super().__init__()
        self.T = T
        self.core = ForwardPINN(T_max=T, n_vars=7 * (1 + P), width=width,
                                depth=depth, n_fourier=n_fourier,
                                fourier_sigma=sigma)
        self.register_buffer("y0", torch.as_tensor(y0).reshape(1, 7))
        self.register_buffer("scale", torch.as_tensor(scale).reshape(1, 7, 1))

    def forward(self, t):
        out = self.core(t).reshape(-1, 7, 1 + P) * self.scale
        phi = (1.0 - torch.exp(-t / 2.0)).reshape(-1, 1, 1)
        Z = self.y0 + phi[:, :, 0] * out[:, :, 0]
        S = phi * out[:, :, 1:]
        return Z, S


def trapezoid(t, u, g):
    """Integral residual (u_{k+1}-u_k)/dt - (g_k+g_{k+1})/2, any trailing dims."""
    dt = (t[1:] - t[:-1]).reshape(-1, *([1] * (u.dim() - 1)))
    return (u[1:] - u[:-1]) / dt - 0.5 * (g[1:] + g[:-1])


class Problem:
    def __init__(self, regime, data_seed, device, n_colloc, width, depth,
                 n_fourier, sigma, net_seed):
        self.regime = regime
        self.device = device
        self.data = bench.make_data(regime, data_seed)
        torch.manual_seed(net_seed)
        self.conds = []
        for d in self.data:
            scale = np.maximum(np.abs(d["y_ref"]).max(axis=0), 0.05)
            net = TangentNet(bench.Y0, scale, bench.T, width, depth,
                             n_fourier, sigma).to(device)
            self.conds.append(dict(
                net=net, forcing=d["forcing"],
                t_d=torch.tensor(d["t_d"], device=device).reshape(-1, 1),
                y_d=torch.tensor(d["y_d"], device=device),
                w=torch.tensor(d["w"], device=device).reshape(1, 7),
                t_c=torch.linspace(0, bench.T, n_colloc,
                                   device=device).reshape(-1, 1)))
        self.eye = torch.eye(P, device=device)

    def nets_params(self):
        return [p for c in self.conds for p in c["net"].parameters()]

    # ---- per-condition quantities -------------------------------------
    def state(self, c, t, x, x_ref):
        Z, S = c["net"](t)
        z = Z + torch.einsum("nij,j->ni", S, x - x_ref)
        return z, S

    def residuals(self, c, x, x_ref, with_sens=True):
        t = c["t_c"]
        z, S = self.state(c, t, x, x_ref)
        forcing = c["forcing"]

        def f(zz, xx):
            return physics_rhs(t, zz, make_params(xx, forcing))

        w = c["w"]
        fz = f(z, x)
        Ls = (trapezoid(t, z, fz) * w).pow(2).mean()
        if not with_sens:
            return Ls, torch.zeros((), device=x.device)
        # G[:, :, j] = (df/dz) S[:, :, j] + df/dx_j   via forward-mode JVPs
        G = vmap(lambda sd, xd: jvp(f, (z, x), (sd, xd))[1],
                 in_dims=(2, 0), out_dims=2)(S, self.eye)
        Lsen = (trapezoid(t, S, G) * w.reshape(1, 7, 1)).pow(2).mean()
        return Ls, Lsen

    def data_loss(self, c, x, x_ref):
        z, _ = self.state(c, c["t_d"], x, x_ref)
        return ((z - c["y_d"]) * c["w"]).pow(2).mean()

    # ---- whole-problem helpers --------------------------------------------
    def losses(self, x, x_ref, with_sens=True):
        Ld = Ls = Lsen = 0.0
        for c in self.conds:
            Ld = Ld + self.data_loss(c, x, x_ref)
            a, b = self.residuals(c, x, x_ref, with_sens)
            Ls, Lsen = Ls + a, Lsen + b
        n = len(self.conds)
        return Ld / n, Ls / n, Lsen / n

    @torch.no_grad()
    def gn_system(self, x, x_ref):
        """Weighted J^T J, J^T r at the data, plus the Fisher matrix."""
        JtJ = torch.zeros(P, P, device=x.device)
        Jtr = torch.zeros(P, device=x.device)
        F = torch.zeros(P, P, device=x.device)
        cost = 0.0
        for c in self.conds:
            z, S = self.state(c, c["t_d"], x, x_ref)
            r = (c["y_d"] - z)
            w = c["w"]
            Sw = S * w.reshape(1, 7, 1)
            Jw = Sw.reshape(-1, P)
            JtJ += Jw.T @ Jw
            Jtr += Jw.T @ (r * w).reshape(-1)
            Jf = S.reshape(-1, P)
            F += Jf.T @ Jf / bench.NOISE ** 2
            cost += float(((r * w) ** 2).sum())
        return JtJ, Jtr, F, cost


def report(prob, x):
    rec = bench.from_x(x.detach().cpu().numpy())
    errs = bench.rel_errors(rec, prob.regime)
    return rec, errs, int(sum(e < 0.1 for e in errs.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("regime")
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--net-seed", type=int, default=0)
    ap.add_argument("--start", default="init", choices=["init", "truth"])
    ap.add_argument("--n-colloc", type=int, default=3000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-fourier", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--pre", type=int, default=2000,
                    help="physics-only pretraining epochs at the initial x")
    ap.add_argument("--joint", type=int, default=6000)
    ap.add_argument("--lam0", type=float, default=1e-2)
    ap.add_argument("--lam1", type=float, default=1e2)
    ap.add_argument("--lam-sens", type=float, default=1.0,
                    help="sensitivity-residual weight relative to lam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-x", type=float, default=5e-3)
    ap.add_argument("--gn-iters", type=int, default=15)
    ap.add_argument("--gn-inner", type=int, default=800)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    safe = a.regime.replace(" ", "_")
    tag = f"{safe}_s{a.net_seed}_{a.start}"

    prob = Problem(a.regime, a.data_seed, dev, a.n_colloc, a.width, a.depth,
                   a.n_fourier, a.sigma, a.net_seed)
    x0 = bench.to_x(bench.true_params(a.regime) if a.start == "truth"
                    else bench.INIT_GUESS)
    x = torch.tensor(x0, device=dev, requires_grad=True)
    x_ref = x.detach().clone()
    hist = []
    t0 = time.time()

    def log(stage, it, Ld, Ls, Lsen, lam=None, extra=""):
        rec, errs, n10 = report(prob, x)
        med = 100 * float(np.median(list(errs.values())))
        hist.append(dict(stage=stage, it=it, Ld=float(Ld), Ls=float(Ls),
                         Lsen=float(Lsen), lam=lam, under10=n10, med=med,
                         x=x.detach().cpu().tolist()))
        print(f"[{tag}] {stage:5s} {it:6d} Ld={float(Ld):.3e} Ls={float(Ls):.3e} "
              f"Lsen={float(Lsen):.3e} lam={lam if lam is None else f'{lam:.2e}'} "
              f"| under10={n10:2d} med={med:5.1f}% W={rec['W']:.3f} "
              f"thP={rec['thetaP']:.3f} {extra}[{time.time()-t0:.0f}s]",
              flush=True)

    netp = prob.nets_params()

    # ---- Stage 0: physics-only pretraining at the initial x ---------------
    opt = torch.optim.Adam(netp, lr=a.lr)
    for ep in range(1, a.pre + 1):
        opt.zero_grad()
        Ld, Ls, Lsen = prob.losses(x.detach(), x_ref)
        (Ls + a.lam_sens * Lsen).backward()
        opt.step()
        if ep % a.log_every == 0 or ep == 1:
            log("pre", ep, Ld, Ls, Lsen)

    # ---- Stage 1: joint Adam with lambda continuation ---------------------
    opt = torch.optim.Adam([{"params": netp, "lr": a.lr},
                            {"params": [x], "lr": a.lr_x}])
    for ep in range(1, a.joint + 1):
        lam = a.lam0 * (a.lam1 / a.lam0) ** (ep / max(a.joint, 1))
        opt.zero_grad()
        Ld, Ls, Lsen = prob.losses(x, x_ref)
        loss = Ld + lam * Ls + lam * a.lam_sens * Lsen
        loss.backward()
        opt.step()
        if ep % a.log_every == 0 or ep == 1:
            log("joint", ep, Ld, Ls, Lsen, lam)

    # ---- Stage 2: profiled Levenberg-Marquardt ---------------------------
    mu = 1e-3
    opt = torch.optim.Adam(netp, lr=a.lr * 0.3)

    def relax(n):
        for _ in range(n):
            opt.zero_grad()
            Ld, Ls, Lsen = prob.losses(x.detach(), x_ref)
            (Ls + a.lam_sens * Lsen).backward()
            opt.step()
        return Ld, Ls, Lsen

    Ld, Ls, Lsen = relax(a.gn_inner)
    JtJ, Jtr, F, cost = prob.gn_system(x.detach(), x_ref)
    for it in range(1, a.gn_iters + 1):
        D = torch.diag(torch.diag(JtJ)).clamp_min(1e-12)
        delta = torch.linalg.solve(JtJ + mu * D, Jtr)
        pred = float(2 * delta @ Jtr - delta @ JtJ @ delta)
        state = [ {k: v.clone() for k, v in c["net"].state_dict().items()}
                  for c in prob.conds ]
        x_old = x.detach().clone()
        with torch.no_grad():
            x += delta
        Ld, Ls, Lsen = relax(a.gn_inner)
        JtJ_n, Jtr_n, F_n, cost_n = prob.gn_system(x.detach(), x_ref)
        rho = (cost - cost_n) / max(pred, 1e-30)
        if cost_n < cost:
            JtJ, Jtr, F, cost = JtJ_n, Jtr_n, F_n, cost_n
            mu = max(mu / 3.0, 1e-7) if rho > 0.25 else mu
            verdict = "acc"
        else:
            with torch.no_grad():
                x.copy_(x_old)
            for c, sd in zip(prob.conds, state):
                c["net"].load_state_dict(sd)
            mu *= 4.0
            verdict = "rej"
        log("gn", it, Ld, Ls, Lsen,
            extra=f"cost={cost:.4e} rho={rho:.2f} mu={mu:.1e} {verdict} ")

    rec, errs, n10 = report(prob, x)
    cov = torch.linalg.pinv(F)
    crlb = torch.sqrt(torch.clamp(torch.diag(cov), min=0)).cpu().numpy()
    out = dict(regime=a.regime, start=a.start, net_seed=a.net_seed,
               args=vars(a), under10=n10, recovered=rec,
               errors={k: float(v) for k, v in errs.items()},
               crlb_rel_sd={k: float(s) for k, s in zip(UNKNOWN, crlb)},
               fisher=F.cpu().tolist(), history=hist,
               seconds=time.time() - t0)
    with open(os.path.join(a.out, f"{tag}.json"), "w") as fh:
        json.dump(out, fh)
    print(f"DONE {tag} under10={n10}/36 "
          f"crlb<10%={int((crlb < 0.1).sum())} [{time.time()-t0:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main()
