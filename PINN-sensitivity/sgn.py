"""Schur-complement Gauss-Newton PINN (SGN-PINN) for multi-condition ODE
parameter estimation.

State model, one per experimental condition c: a physics-informed network
whose output is LINEAR in its head,

    z_c(t) = (H_c(t) A_c) * scale_c ,

H_c = [ PINN last-hidden features (optional) | random Fourier features | 1 ].
With the features fixed, the inverse problem in (A_1..A_C, theta) is a
structured nonlinear least-squares problem

    sum_c |W (z_c(t_d) - y_c) / sigma|^2                         (data)
  + lam * sum_c |W R_c(A_c, theta)|^2 dt / sigma^2               (physics)
  + lam_ic * sum_c |W (z_c(0) - y0) / sigma|^2                   (IC)

with R the derivative-free trapezoidal residual
R_k = (z_{k+1} - z_k)/dt - (f(z_k) + f(z_{k+1}))/2.

Its Gauss-Newton matrix is block-arrowhead: C dense head blocks coupled only
through the 36 physical parameters, so each Levenberg-Marquardt step in all
~7C(h+1)+36 unknowns costs C Cholesky factorisations plus one 36x36 solve on
the Schur complement

    S = sum_c [ X_c - B_c^T (A_c + D_c)^{-1} B_c ].

lam is continued upward (smoothing -> shooting). At large lam the estimate is
the constrained maximum-likelihood estimate within the head span, and the
undamped S is the profiled Fisher information of log(theta): sqrt(diag S^-1)
are Cramer-Rao standard deviations, computed by the training itself.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.func import jacfwd, vmap

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


class Features:
    """Head basis [PINN last-hidden (optional) | random Fourier | 1]."""

    def __init__(self, net=None, n_rff=512, f_max=0.5, seed=0, n_spline=0,
                 n_exp=0, tau_min=0.05, tau_max=20.0):
        self.net = net
        # decaying exponentials exp(-t/tau): the initial-transient modes that
        # global Fourier features cannot represent (where ~85% of the
        # physics-residual floor sits without them)
        self.taus = torch.logspace(np.log10(tau_min), np.log10(tau_max), n_exp) \
            if n_exp else torch.zeros(0)
        g = torch.Generator().manual_seed(seed)
        self.om = torch.rand(n_rff // 2, generator=g) * 2 * np.pi * f_max
        self.n_spline = n_spline
        if n_spline:
            h = bench.T / (n_spline - 3)
            self.knots = torch.arange(-3, n_spline - 2) * h     # n_spline+1 left knots
            self.h = h

    def splines(self, t):
        """Uniform cubic B-splines covering [0, T] (local support 4h)."""
        u = (t - self.knots[None, :]) / self.h                # (n, n_spline+1)
        u = u[:, :self.n_spline]

        def b3(v):
            out = torch.zeros_like(v)
            m = (v >= 0) & (v < 1); out[m] = v[m] ** 3 / 6
            m = (v >= 1) & (v < 2); w = v[m] - 1
            out[m] = (-3 * w ** 3 + 3 * w ** 2 + 3 * w + 1) / 6
            m = (v >= 2) & (v < 3); w = v[m] - 2
            out[m] = (3 * w ** 3 - 6 * w ** 2 + 4) / 6
            m = (v >= 3) & (v < 4); w = 4 - v[m]
            out[m] = w ** 3 / 6
            return out
        return b3(u)

    def __call__(self, t):
        cols = []
        if self.n_spline:
            cols.append(self.splines(t))
        if self.net is not None:
            with torch.no_grad():
                cols.append(self.net.net[:-1](self.net._embed(t)))
        if self.om.numel():
            a = t * self.om[None, :]
            cols += [torch.sin(a), torch.cos(a)]
        if self.taus.numel():
            cols.append(torch.exp(-t / self.taus[None, :]))
        cols.append(torch.ones(t.shape[0], 1))
        return torch.cat(cols, dim=1)


class Cond:
    def __init__(self, feats, d, n_grid, lam_ic, dev, net=None, weights="rel"):
        self.forcing = d["forcing"]
        self.dev = dev
        scale = np.maximum(np.abs(d["y_ref"]).max(axis=0), 0.05) if net is None \
            else net.out_scale.reshape(7).numpy()
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64).to(dev)
        self.scale = T(scale)
        t = torch.linspace(0, bench.T, n_grid).reshape(-1, 1)
        self.dt = float(t[1] - t[0])
        td = torch.tensor(d["t_d"]).reshape(-1, 1)
        self.Hg = feats(t).to(dev)
        self.Hd = feats(td).to(dev)
        self.H0 = feats(torch.zeros(1, 1)).to(dev)
        self.t = t.to(dev)
        self.y = T(d["y_d"])
        self.w = T(d["w"] if weights == "rel" else np.ones(7))
        self.y0 = T(bench.Y0)
        self.lam_ic = lam_ic
        self.nh = self.Hg.shape[1]
        # lambda-independent scale for the ridge: mean data-block curvature
        cw = self.w * self.scale / bench.NOISE
        self.data_diag = float((cw ** 2).mean() * torch.diag(self.Hd.T @ self.Hd).mean())
        self.A = torch.zeros(self.nh, 7, device=dev)
        if net is not None:
            last = net.net[-1]
            h = last.weight.shape[1]
            self.A[:h] = last.weight.detach().T.to(dev)
            self.A[-1] = last.bias.detach().to(dev)

    def z(self, H, A):
        return (H @ A) * self.scale

    def residual(self, A, x, lam):
        s = bench.NOISE
        rd = ((self.z(self.Hd, A) - self.y) * self.w / s).reshape(-1)
        zg = self.z(self.Hg, A)
        fg = physics_rhs(self.t, zg, make_params(x, self.forcing))
        R = (zg[1:] - zg[:-1]) / self.dt - 0.5 * (fg[1:] + fg[:-1])
        rp = (np.sqrt(lam * self.dt) * R * self.w / s).reshape(-1)
        ric = (np.sqrt(self.lam_ic) * (self.z(self.H0, A) - self.y0) * self.w / s).reshape(-1)
        return rd, rp, ric

    def smooth_fit_head(self, feats, power=2.0, omega_c=0.3):
        """Smoothing initialisation without a PINN: per-state generalised
        ridge with penalty (1 + (omega/omega_c)^2)^power on each random
        Fourier coefficient (a Sobolev-type prior), strength chosen by the
        discrepancy principle (weighted residual sum = number of data)."""
        s = bench.NOISE
        om = feats.om.to(self.dev)
        pen = torch.cat([(1 + (om / omega_c) ** 2) ** power] * 2
                        + [torch.ones(feats.taus.numel(), device=self.dev)]
                        + [torch.full((1,), 1e-6, device=self.dev)])
        if pen.numel() != self.nh:              # PINN / spline features present
            pen = torch.cat([torch.ones(self.nh - pen.numel(), device=self.dev), pen])
        ci = np.sqrt(self.lam_ic)
        H = torch.cat([self.Hd, ci * self.H0], 0)
        G = H.T @ H
        A = torch.zeros(self.nh, 7, device=self.dev)
        n = self.Hd.shape[0]
        for i in range(7):
            yi = torch.cat([self.y[:, i], ci * self.y0[i:i + 1]]) / self.scale[i]
            b = H.T @ yi
            best = None
            for la in torch.logspace(-12, 2, 57):
                a = torch.linalg.solve(G + la * torch.diag(pen), b)
                r = (self.Hd @ a * self.scale[i] - self.y[:, i]) / s
                chi = float(r @ r) / n
                if best is None or abs(np.log(chi)) < best[0]:
                    best = (abs(np.log(chi)), a)
            A[:, i] = best[1]
        self.A = A

    def data_fit_head(self, ridge):
        """lam = 0 initialisation: per-state ridge regression of the data."""
        s = bench.NOISE
        H = torch.cat([self.Hd, np.sqrt(self.lam_ic) * self.H0], 0)
        Y = torch.cat([self.y, np.sqrt(self.lam_ic) * self.y0[None]], 0) / self.scale
        G = H.T @ H
        G = G + ridge * torch.diag(G).mean() * torch.eye(self.nh, device=self.dev)
        self.A = torch.linalg.solve(G, H.T @ Y)

    def normal_eq(self, A, x, lam, chunk=400):
        """Gauss-Newton normal-equation blocks for this condition,
        accumulated over row chunks (the Jacobian is never materialised):
        returns J_a^T J_a, J_a^T J_x, J_a^T r, J_x^T J_x, J_x^T r with
        column block i of J_a = head of output i."""
        s, nh, dev = bench.NOISE, self.nh, self.dev
        rd, rp, ric = self.residual(A, x, lam)
        n7 = 7 * nh
        Aa = torch.zeros(n7, n7, device=dev)
        Bx = torch.zeros(n7, P, device=dev)
        ga = torch.zeros(n7, device=dev)
        Xx = torch.zeros(P, P, device=dev)
        gx = torch.zeros(P, device=dev)
        # data + IC rows: block-diagonal in the output index
        cw = self.w * self.scale / s
        ci = np.sqrt(self.lam_ic)
        Gd = self.Hd.T @ self.Hd + ci ** 2 * self.H0.T @ self.H0
        rdm = rd.reshape(-1, 7)
        ricm = ric.reshape(1, 7)
        for i in range(7):
            sl = slice(i * nh, (i + 1) * nh)
            Aa[sl, sl] += cw[i] ** 2 * Gd
            ga[sl] += cw[i] * (self.Hd.T @ rdm[:, i] + ci * self.H0[0] * ricm[0, i])
        # physics rows, chunked over trapezoid intervals
        zg = self.z(self.Hg, A)
        forcing = self.forcing

        def f1(zz, tt, xx):
            return physics_rhs(tt.reshape(1, 1), zz.reshape(1, 7),
                               make_params(xx, forcing)).reshape(7)
        Jz = vmap(jacfwd(f1, argnums=0), in_dims=(0, 0, None))(zg, self.t, x)
        Jx = vmap(jacfwd(f1, argnums=2), in_dims=(0, 0, None))(zg, self.t, x)
        eye = torch.eye(7, device=dev)
        c = np.sqrt(lam * self.dt) / s
        wc = (c * self.w)[None, :, None]
        Mk1 = (eye / self.dt - 0.5 * Jz[1:]) * self.scale[None, None, :] * wc
        Mk = (-eye / self.dt - 0.5 * Jz[:-1]) * self.scale[None, None, :] * wc
        Jpx = -0.5 * (Jx[1:] + Jx[:-1]) * wc
        rpm = rp.reshape(-1, 7)
        n = Mk.shape[0]
        for a0 in range(0, n, chunk):
            b0 = min(a0 + chunk, n)
            J = (Mk1[a0:b0, :, :, None] * self.Hg[a0 + 1:b0 + 1, None, None, :]
                 + Mk[a0:b0, :, :, None] * self.Hg[a0:b0, None, None, :]).reshape(-1, n7)
            Jxc = Jpx[a0:b0].reshape(-1, P)
            rc = rpm[a0:b0].reshape(-1)
            Aa += J.T @ J
            Bx += J.T @ Jxc
            ga += J.T @ rc
            Xx += Jxc.T @ Jxc
            gx += Jxc.T @ rc
            del J
        return Aa, Bx, ga, Xx, gx


def costs(conds, As, x, lam):
    cd = cp = 0.0
    for c, A in zip(conds, As):
        rd, rp, ric = c.residual(A, x, lam)
        cd += float(rd @ rd + ric @ ric)
        cp += float(rp @ rp)
    return cd, cp


def normal_blocks(conds, As, x, lam):
    return [c.normal_eq(A, x, lam) + (c.data_diag,) for c, A in zip(conds, As)]


def lm_step(blocks, As, x, mu, ridge, dev, freeze_heads=False):
    S = torch.zeros(P, P, device=dev)
    s = torch.zeros(P, device=dev)
    cache = []
    if freeze_heads:                      # ablation: theta-only Gauss-Newton
        for Aa, Bx, ga, Xx, gx, dd in blocks:
            S += Xx
            s += gx
        Dx = torch.diag(S).clamp_min(1e-30) * mu
        dx = -torch.linalg.solve(S + torch.diag(Dx), s)
        return [A.clone() for A in As], x + dx, float(s @ dx)
    for Aa, Bx, ga, Xx, gx, dd in blocks:
        dA = torch.diag(Aa)
        Da = dA.clamp_min(1e-30) * mu + ridge * dd
        L, info = torch.linalg.cholesky_ex(Aa + torch.diag(Da))
        if int(info) != 0:
            return None
        AiB = torch.cholesky_solve(Bx, L)
        Aig = torch.cholesky_solve(ga[:, None], L)[:, 0]
        S += Xx - Bx.T @ AiB
        s += gx - Bx.T @ Aig
        cache.append((AiB, Aig))
    Dx = torch.diag(S).clamp_min(1e-30) * mu
    dx = -torch.linalg.solve(S + torch.diag(Dx), s)
    newAs = [A - (Aig + AiB @ dx).reshape(7, -1).T
             for A, (AiB, Aig) in zip(As, cache)]
    return newAs, x + dx, float(s @ dx)


def profiled_fisher(blocks, ridge, dev):
    S = torch.zeros(P, P, device=dev)
    for Aa, Bx, ga, Xx, gx, dd in blocks:
        Aa = Aa + ridge * dd * torch.eye(Aa.shape[0], device=dev)
        S += Xx - Bx.T @ torch.linalg.solve(Aa, Bx)
    return S


def crlb_sd(S):
    """Per-parameter Cramer-Rao sd of log(theta) from a (possibly singular)
    Fisher matrix: eigen-decomposition with eigenvalues floored at
    1e-14 * max, so parameters loading on null directions get huge sd
    instead of the spurious zeros a clamped pinv gives."""
    S = 0.5 * (S + S.T)
    ev, V = np.linalg.eigh(S)
    ev = np.maximum(ev, 1e-14 * ev.max())
    return np.sqrt((V ** 2 / ev[None, :]).sum(axis=1))


def run_adam(conds, x, lam_sched, steps, lr, lr_x, log, regime, dev):
    """Ablation: the SAME objective, heads and lambda schedule, optimised by
    Adam (what an inverse PINN uses) instead of Schur Gauss-Newton."""
    As = [c.A.clone().requires_grad_(True) for c in conds]
    x = x.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": As, "lr": lr}, {"params": [x], "lr": lr_x}])
    hist = []
    for lam in lam_sched:
        for it in range(steps):
            opt.zero_grad()
            tot = 0.0
            for c, A in zip(conds, As):
                rd, rp, ric = c.residual(A, x, lam)
                tot = tot + (rd @ rd + ric @ ric + rp @ rp)
            tot.backward()
            opt.step()
            if it % max(steps // 4, 1) == 0 or it == steps - 1:
                with torch.no_grad():
                    cd, cp = costs(conds, As, x, lam)
                rec = bench.from_x(x.detach().cpu().numpy())
                errs = bench.rel_errors(rec, regime)
                n10 = int(sum(e < 0.1 for e in errs.values()))
                med = 100 * float(np.median(list(errs.values())))
                hist.append(dict(lam=lam, it=it, data=cd, phys=cp, under10=n10,
                                 med=med, x=x.detach().cpu().tolist()))
                log(f"[adam] lam={lam:.1e} it={it:5d} data={cd:.5e} phys={cp:.3e} "
                    f"| under10={n10:2d} med={med:5.2f}% W={rec['W']:.4f} thP={rec['thetaP']:.4f}")
    As = [A.detach() for A in As]
    x = x.detach()
    blocks = normal_blocks(conds, As, x, lam_sched[-1])
    return As, x, profiled_fisher(blocks, 1e-10, dev), hist


def head_solve(c, A, x, lam, iters, ridge, tol=1e-9):
    """Gauss-Newton/LM on one condition's head at fixed theta."""
    mu = 1e-4
    rd, rp, ric = c.residual(A, x, lam)
    cost = float(rd @ rd + rp @ rp + ric @ ric)
    for it in range(iters):
        Aa, Bx, ga, Xx, gx = c.normal_eq(A, x, lam)
        M = Aa + torch.diag(torch.diag(Aa) * mu + ridge * c.data_diag)
        L, info = torch.linalg.cholesky_ex(M)
        if int(info) != 0:
            mu *= 5
            continue
        An = A - torch.cholesky_solve(ga[:, None], L)[:, 0].reshape(7, -1).T
        rd, rp, ric = c.residual(An, x, lam)
        cn = float(rd @ rd + rp @ rp + ric @ ric)
        if np.isfinite(cn) and cn < cost:
            done = (cost - cn) < tol * cost
            A, cost, mu = An, cn, max(mu / 3, 1e-12)
            if done:
                break
        else:
            mu *= 5
    return A, cost


def run_vp(conds, x, lam_sched, outer, inner, mu0, ridge, log, regime, dev, rtol):
    """Variable projection: theta is the only outer variable. For each theta
    the heads are solved to convergence (A*(theta): a data-anchored,
    physics-consistent trajectory per condition); the outer step is the
    profiled Gauss-Newton/LM step through the Schur complement, i.e. the
    exact Golub-Pereyra reduced Jacobian. Behaves like single shooting with
    the integrator replaced by the head solve."""
    As = [c.A.clone() for c in conds]
    x = x.clone()
    hist = []
    mu = mu0
    nu = 2.0
    for lam in lam_sched:
        res = [head_solve(c, A, x, lam, 3 * inner, ridge) for c, A in zip(conds, As)]
        As = [r[0] for r in res]
        cd, cp = costs(conds, As, x, lam)
        cost = cd + cp
        log(f"[vp] lam={lam:.1e} heads solved: data={cd:.5e} phys={cp:.3e}")
        n_rej = 0
        for it in range(outer):
            blocks = normal_blocks(conds, As, x, lam)
            step = lm_step(blocks, As, x, mu, ridge, dev)
            acc = False
            if step is not None:
                predAs, newx, pred = step
                res = [head_solve(c, A, newx, lam, inner, ridge) for c, A in zip(conds, predAs)]
                newAs = [r[0] for r in res]
                cd_n, cp_n = costs(conds, newAs, newx, lam)
                if np.isfinite(cd_n + cp_n) and cd_n + cp_n < cost:
                    rel = (cost - cd_n - cp_n) / cost
                    # Nielsen (1999) damping update from the gain ratio
                    rho = (cost - cd_n - cp_n) / max(-pred, 1e-30)
                    As, x, cost, cd, cp = newAs, newx, cd_n + cp_n, cd_n, cp_n
                    mu = max(mu * max(1 / 3, 1 - (2 * min(rho, 1.0) - 1) ** 3), 1e-12)
                    nu = 2.0
                    acc = True
            if not acc:
                mu *= nu
                nu *= 2
                n_rej += 1
            rec = bench.from_x(x.cpu().numpy())
            errs = bench.rel_errors(rec, regime)
            n10 = int(sum(e < 0.1 for e in errs.values()))
            med = 100 * float(np.median(list(errs.values())))
            hist.append(dict(lam=lam, it=it, data=cd, phys=cp, mu=mu, acc=acc,
                             under10=n10, med=med, x=x.cpu().tolist()))
            log(f"[vp] lam={lam:.1e} it={it:2d} data={cd:.5e} phys={cp:.3e} "
                f"mu={mu:.1e} {'acc' if acc else 'rej'} | under10={n10:2d} "
                f"med={med:5.2f}% W={rec['W']:.4f} thP={rec['thetaP']:.4f}")
            if acc and rel < rtol:
                break
            if n_rej >= 8:
                break
    blocks = normal_blocks(conds, As, x, lam_sched[-1])
    return As, x, profiled_fisher(blocks, ridge, dev), hist


def head_presolve(conds, As, x, lam, iters, ridge, log, dev):
    """Variable-projection start: Gauss-Newton on the heads ALONE at fixed
    theta (per condition, independent), so the joint steps begin from a
    trajectory that is physics-consistent at the current theta instead of
    letting theta absorb the inconsistency of the initial trajectory."""
    out = []
    for c, A in zip(conds, As):
        mu = 1e-3
        rd, rp, ric = c.residual(A, x, lam)
        cost = float(rd @ rd + rp @ rp + ric @ ric)
        for it in range(iters):
            Aa, Bx, ga, Xx, gx = c.normal_eq(A, x, lam)
            M = Aa + torch.diag(torch.diag(Aa) * mu + ridge * c.data_diag)
            L, info = torch.linalg.cholesky_ex(M)
            if int(info) != 0:
                mu *= 5
                continue
            An = A - torch.cholesky_solve(ga[:, None], L)[:, 0].reshape(7, -1).T
            rd, rp, ric = c.residual(An, x, lam)
            cn = float(rd @ rd + rp @ rp + ric @ ric)
            if cn < cost:
                A, cost, mu = An, cn, max(mu / 3, 1e-12)
            else:
                mu *= 5
        out.append(A)
    cd, cp = costs(conds, out, x, lam)
    log(f"head presolve lam={lam:.1e}: data={cd:.5e} phys={cp:.3e}")
    return out


def run(conds, x, lam_sched, max_iters, mu0, ridge, log, regime, dev, rtol,
        freeze_heads=False, presolve=0):
    As = [c.A.clone() for c in conds]
    x = x.clone()
    mu = mu0
    hist = []
    if presolve:
        As = head_presolve(conds, As, x, lam_sched[0], presolve, ridge, log, dev)
    for lam in lam_sched:
        cd, cp = costs(conds, As, x, lam)
        cost = cd + cp
        blocks = normal_blocks(conds, As, x, lam)
        n_rej = 0
        for it in range(max_iters):
            step = lm_step(blocks, As, x, mu, ridge, dev, freeze_heads)
            acc = False
            if step is not None:
                newAs, newx, pred = step
                cd_n, cp_n = costs(conds, newAs, newx, lam)
                if np.isfinite(cd_n + cp_n) and cd_n + cp_n < cost:
                    rel = (cost - cd_n - cp_n) / cost
                    As, x, cost, cd, cp = newAs, newx, cd_n + cp_n, cd_n, cp_n
                    mu = max(mu / 3, 1e-12)
                    acc = True
                    blocks = normal_blocks(conds, As, x, lam)
            if not acc:
                mu *= 5
                n_rej += 1
            rec = bench.from_x(x.cpu().numpy())
            errs = bench.rel_errors(rec, regime)
            n10 = int(sum(e < 0.1 for e in errs.values()))
            med = 100 * float(np.median(list(errs.values())))
            hist.append(dict(lam=lam, it=it, data=cd, phys=cp, mu=mu, acc=acc,
                             under10=n10, med=med, x=x.cpu().tolist()))
            log(f"lam={lam:.1e} it={it:2d} data={cd:.5e} phys={cp:.3e} "
                f"mu={mu:.1e} {'acc' if acc else 'rej'} | under10={n10:2d} "
                f"med={med:5.2f}% W={rec['W']:.4f} thP={rec['thetaP']:.4f}")
            if acc and rel < rtol:
                break
            if n_rej >= 8:
                break
    S = profiled_fisher(blocks, ridge, dev)
    return As, x, S, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("regime")
    ap.add_argument("--pinn-dir", default=None,
                    help="run dir with <regime>_<cond>_net.pt and _params.pt; "
                         "omit for a random-feature head only")
    ap.add_argument("--n-rff", type=int, default=512)
    ap.add_argument("--f-max", type=float, default=0.5)
    ap.add_argument("--feat-seed", type=int, default=0)
    ap.add_argument("--n-spline", type=int, default=0)
    ap.add_argument("--n-exp", type=int, default=0)
    ap.add_argument("--smooth-power", type=float, default=2.0)
    ap.add_argument("--omega-c", type=float, default=0.3)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--weights", default="rel", choices=["rel", "none"])
    ap.add_argument("--x-start", default="init", choices=["pinn", "init", "truth"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--jitter-seed", type=int, default=None,
                    help="add N(0, 0.35^2) log-jitter to the starting theta (as mle.py)")
    ap.add_argument("--n-grid", type=int, default=3000)
    ap.add_argument("--lam", default="1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100,300,1000")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rtol", type=float, default=1e-7)
    ap.add_argument("--mu0", type=float, default=1e-3)
    ap.add_argument("--ridge", type=float, default=1e-10)
    ap.add_argument("--lam-ic", type=float, default=100.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--opt", default="gn", choices=["gn", "adam", "vp"])
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--adam-steps", type=int, default=2000)
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--adam-lr-x", type=float, default=5e-3)
    ap.add_argument("--freeze-heads", action="store_true")
    ap.add_argument("--presolve", type=int, default=0,
                    help="head-only GN iterations at fixed theta before the joint steps")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    safe = a.regime.replace(" ", "_")
    tag = f"{safe}_{a.x_start}{a.tag}"
    t0 = time.time()
    logf = open(os.path.join(a.out, f"{tag}.log"), "w")

    def log(msg):
        line = f"[{tag}] {msg} [{time.time()-t0:.0f}s]"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    data = bench.make_data(a.regime, a.data_seed)
    conds = []
    for d in data:
        net = None
        if a.pinn_dir:
            sd = torch.load(os.path.join(a.pinn_dir, f"{safe}_{d['name']}_net.pt"),
                            map_location="cpu")
            net = ForwardPINN(T_max=bench.T, width=sd["net.0.weight"].shape[0],
                              depth=sum(1 for k in sd if k.endswith("weight")) - 1,
                              n_fourier=sd["B"].shape[0])
            net.load_state_dict(sd)
        feats = Features(net, a.n_rff, a.f_max, a.feat_seed, a.n_spline, a.n_exp)
        c = Cond(feats, d, a.n_grid, a.lam_ic, dev, net, a.weights)
        if net is None:
            c.smooth_fit_head(feats, a.smooth_power, a.omega_c)
        conds.append(c)
    if a.x_start == "pinn":
        raw = torch.load(os.path.join(a.pinn_dir, f"{safe}_params.pt"), map_location="cpu")
        rec0 = {k: (1 / (1 + np.exp(-float(raw[f"raw.{k}"]))) if k == "thetaP"
                    else BASELINE[k] * np.exp(float(raw[f"raw.{k}"]))) for k in UNKNOWN}
        x0 = bench.to_x(rec0)
    elif a.x_start == "truth":
        x0 = bench.to_x(bench.true_params(a.regime))
    else:
        x0 = bench.to_x(bench.INIT_GUESS)
    if a.jitter_seed is not None:
        x0 = x0 + np.random.default_rng(a.jitter_seed).normal(0, 0.35, x0.shape)
    x = torch.tensor(x0, device=dev)
    errs0 = bench.rel_errors(bench.from_x(x0), a.regime)
    log(f"start x ({a.x_start}): under10={int(sum(e < 0.1 for e in errs0.values()))} "
        f"nh={conds[0].nh} unknowns={7 * conds[0].nh * len(conds) + P}")
    lam_sched = [float(v) for v in a.lam.split(",")]
    if a.opt == "vp":
        As, x, S, hist = run_vp(conds, x, lam_sched, a.iters, a.inner, a.mu0,
                                a.ridge, log, a.regime, dev, a.rtol)
    elif a.opt == "adam":
        As, x, S, hist = run_adam(conds, x, lam_sched, a.adam_steps, a.adam_lr,
                                  a.adam_lr_x, log, a.regime, dev)
    else:
        As, x, S, hist = run(conds, x, lam_sched, a.iters, a.mu0, a.ridge, log,
                             a.regime, dev, a.rtol, a.freeze_heads, a.presolve)
    rec = bench.from_x(x.cpu().numpy())
    errs = bench.rel_errors(rec, a.regime)
    S = S.cpu()
    sd = crlb_sd(S.numpy())
    n10 = int(sum(e < 0.1 for e in errs.values()))
    cd, cp = costs(conds, As, x, lam_sched[-1])
    out = dict(regime=a.regime, x_start=a.x_start, args=vars(a), under10=n10,
               recovered=rec, errors={k: float(v) for k, v in errs.items()},
               start_under10=int(sum(e < 0.1 for e in errs0.values())),
               sd_logtheta={k: float(v) for k, v in zip(UNKNOWN, sd)},
               fisher=S.tolist(), data_cost=cd, phys_cost=cp, history=hist,
               seconds=time.time() - t0)
    with open(os.path.join(a.out, f"{tag}.json"), "w") as fh:
        json.dump(out, fh)
    log(f"DONE under10={n10}/36  sd<5%={int((sd < 0.05).sum())}  data={cd:.5e}")


if __name__ == "__main__":
    main()
