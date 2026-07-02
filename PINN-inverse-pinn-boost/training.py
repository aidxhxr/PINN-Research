import os
import json
import math
import time as wall

import numpy as np
import torch

from config import (BASELINE, REGIMES, CONDITIONS, DEVICE, VAR_NAMES,
                    UNKNOWN, INIT_GUESS, PARAM_RANGE, NOMINAL)
from model import ForwardPINN, InverseParams, time_derivatives
from residual import physics_residual, physics_rhs


def _fmt_est(cur, true_vals):
    """Compact W / thetaP + mean-relative-error string for the log lines (a
    mean relative error over all unknowns (a full 36-param line would be
    unreadable)."""
    errs = [abs(cur[k] - true_vals[k]) / abs(true_vals[k])
            for k in cur if true_vals[k] != 0]
    mean_err = 100.0 * sum(errs) / len(errs)
    return (f"W={cur['W']:.3f}(*{true_vals['W']:.2f})  "
            f"thetaP={cur['thetaP']:.3f}(*{true_vals['thetaP']:.2f})  "
            f"<rel.err>={mean_err:.0f}%")


def _count_under(cur, true_vals, tol=0.10):
    return sum(1 for k in cur if true_vals[k] != 0
               and abs(cur[k] - true_vals[k]) / abs(true_vals[k]) < tol)


def sample_sparse(t_ref, y_ref, n_data, noise_std, seed):
    """Pick n_data scattered observation times (random, + the t=0 anchor)
    and optionally corrupt them with Gaussian noise — a realistic inverse
    setting where we recover parameters from a few noisy measurements."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(1, len(t_ref)), size=n_data - 1, replace=False)
    idx = np.sort(np.concatenate([[0], idx]))
    t_d = t_ref[idx, None]
    y_d = y_ref[idx].copy()
    if noise_std > 0:
        y_d = y_d + rng.normal(0.0, noise_std, size=y_d.shape)
    return t_d, y_d, idx


def _grad_norm(loss, params):
    """L2 norm of d loss / d params, summed over the parameter list."""
    grads = torch.autograd.grad(loss, params, retain_graph=False,
                                create_graph=False, allow_unused=True)
    sq = 0.0
    for g in grads:
        if g is not None:
            sq = sq + (g**2).sum()
    return torch.sqrt(sq + 1e-30)


# ----------------------------------------------------------------------
# One full training run (one random start).  train_inverse() below wraps
# this in a multi-start loop and keeps the best by final physics residual.
# ----------------------------------------------------------------------
def _train_once(name, refs_for_regime, true_vals, *,
                T, width, depth, n_fourier, fourier_sigma,
                n_colloc, n_data, noise_std,
                adam_epochs, lbfgs_steps, param_refine_steps,
                param_refine_colloc, lr, lr_param,
                lam_data, lam_phys, lam_ic,
                adaptive_weights, weight_every, weight_beta,
                rel_weight, colloc_mode, residual_mode,
                activation, siren_w0,
                seed, init_jitter, log_every, tag):
    torch.manual_seed(seed)

    # ---- initial parameter guess (jittered for starts > anchor) ----
    init_guess = dict(INIT_GUESS)
    if init_jitter > 0:
        rng = np.random.default_rng(seed)
        for k in UNKNOWN:
            init_guess[k] = INIT_GUESS[k] * float(
                np.exp(rng.normal(0.0, init_jitter)))
        # keep the bounded thetaP inside (0,1)
        init_guess["thetaP"] = float(np.clip(init_guess["thetaP"], 0.05, 0.95))

    params = InverseParams(init_guess, PARAM_RANGE, NOMINAL).to(DEVICE)

    # ---- per-condition setup (own net + own sparse data + forcing) ----
    conds = []
    for ci, c in enumerate(CONDITIONS):
        t_ref, y_ref = refs_for_regime[c["name"]]
        scale = np.maximum(np.abs(y_ref).max(axis=0), 0.05)      # (7,)
        net = ForwardPINN(T_max=T, width=width, depth=depth,
                          n_fourier=n_fourier, fourier_sigma=fourier_sigma,
                          out_scale=scale, activation=activation,
                          siren_w0=siren_w0).to(DEVICE)
        t_arr, y_arr, idx = sample_sparse(t_ref, y_ref, n_data,
                                          noise_std, seed + ci)
        # per-variable relative weight (1/scale) applied to data + residual so
        # small-magnitude states (the "dark" b/h5/h13/m sub-network the excite
        # conditions illuminate) are not drowned out by beta-catenin / APC.
        w = (1.0 / scale) if rel_weight else np.ones_like(scale)
        w_t = torch.tensor(w, device=DEVICE).reshape(1, 7)
        # deterministic collocation grid (fixed => clean, non-stochastic
        # theta-gradient; also the sorted grid the integral residual needs).
        t_fixed = torch.linspace(0, T, n_colloc, device=DEVICE).reshape(-1, 1)
        conds.append(dict(
            name=c["name"], forcing=c["forcing"], net=net,
            t_d=torch.tensor(t_arr, device=DEVICE),
            y_d=torch.tensor(y_arr, device=DEVICE),
            y0=torch.tensor(y_ref[0], device=DEVICE),
            t_arr=t_arr, y_arr=y_arr, idx=idx,
            w_res=w_t, t_fixed=t_fixed,
        ))

    net_params = [p for c in conds for p in c["net"].parameters()]
    n_net = sum(p.numel() for p in conds[0]["net"].parameters())
    print(f"  [{tag}] arch {depth}x{width} act={activation} "
          f"net {n_net:,}x{len(conds)}cond  "
          f"rel_w={rel_weight} colloc={colloc_mode} res={residual_mode}")

    opt = torch.optim.Adam([
        {"params": net_params,          "lr": lr},
        {"params": params.parameters(), "lr": lr_param},
    ])

    # Decoupled cosine schedules: the net anneals to ~zero LR, but the
    # PHYSICAL parameters keep >=0.3x so they do not freeze before converging.
    def lam_net(ep):
        r = 1e-3
        return r + (1 - r) * 0.5 * (1 + math.cos(math.pi * ep / adam_epochs))

    def lam_par(ep):
        r = 0.3
        return r + (1 - r) * 0.5 * (1 + math.cos(math.pi * ep / adam_epochs))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam_net, lam_par])

    hist = dict(epoch=[], loss=[], Ld=[], Lp=[], Lic=[],
                lam_phys=[], **{k: [] for k in UNKNOWN})

    def physics_params(forcing):
        return {**BASELINE, **params.dict(), **forcing}

    def data_ic_loss(c):
        Ld = (((c["net"](c["t_d"]) - c["y_d"]) * c["w_res"])**2).mean()
        Lic = (((c["net"](torch.zeros(1, 1, device=DEVICE))[0] - c["y0"])
                * c["w_res"][0])**2).mean()
        return Ld, Lic

    def physics_loss(c, n_pts=None):
        """Physics residual for one condition, in the selected mode.

        deriv    : ||dz/dt_autodiff - f(z,theta)||   (classic PINN residual)
        integral : ||z_{k+1}-z_k - 0.5 dt (f_k+f_{k+1})||  (trapezoidal
                   multiple-shooting; uses net VALUES only, never the biased
                   autodiff derivative — the accuracy lever).
        """
        net, w = c["net"], c["w_res"]
        pp = physics_params(c["forcing"])
        if residual_mode == "integral":
            tc = c["t_fixed"]                       # sorted, fixed
            if n_pts is not None and n_pts < tc.shape[0]:
                tc = tc[:: tc.shape[0] // n_pts]    # strided => stays sorted
            z = net(tc)
            f = physics_rhs(tc, z, pp)
            dt = tc[1:] - tc[:-1]
            r = ((z[1:] - z[:-1]) - 0.5 * dt * (f[1:] + f[:-1])) * w
            return (r**2).mean()
        if colloc_mode == "fixed":
            tc = c["t_fixed"]
        else:
            tc = torch.rand(n_pts or n_colloc, 1, device=DEVICE) * T
        z, dz = time_derivatives(net, tc)
        r = physics_residual(tc, z, dz, pp) * w
        return (r**2).mean()

    def total_phys():
        with torch.no_grad():
            return float(sum(physics_loss(c) for c in conds) / len(conds))

    safe = name.replace(" ", "_").replace("/", "_")
    t0 = wall.perf_counter()
    lam_phys_cur = float(lam_phys)

    # ---- Stage 1: Adam ----
    for ep in range(1, adam_epochs + 1):
        if adaptive_weights and (ep % weight_every == 0 or ep == 1):
            Ld_s = sum(data_ic_loss(c)[0] for c in conds)
            Lp_s = sum(physics_loss(c, n_colloc // 4) for c in conds)
            g_d = _grad_norm(lam_data * Ld_s, net_params)
            g_p = _grad_norm(Lp_s, net_params)
            target = float((g_d / (g_p + 1e-30)).item())
            target = min(max(target, 1e-2), 1e3)
            lam_phys_cur = (1 - weight_beta) * lam_phys_cur + weight_beta * target

        opt.zero_grad()
        Ld_t = Lic_t = Lp_t = 0.0
        for c in conds:
            Ld, Lic = data_ic_loss(c)
            Lp = physics_loss(c, n_colloc)
            Ld_t = Ld_t + Ld
            Lic_t = Lic_t + Lic
            Lp_t = Lp_t + Lp
        nC = len(conds)
        Ld_t, Lic_t, Lp_t = Ld_t / nC, Lic_t / nC, Lp_t / nC

        loss = lam_data * Ld_t + lam_phys_cur * Lp_t + lam_ic * Lic_t
        loss.backward()
        opt.step()
        sched.step()

        if ep % log_every == 0 or ep == 1:
            cur = params.values()
            hist["epoch"].append(ep)
            hist["loss"].append(loss.item())
            hist["Ld"].append(Ld_t.item())
            hist["Lp"].append(Lp_t.item())
            hist["Lic"].append(Lic_t.item())
            hist["lam_phys"].append(lam_phys_cur)
            for k in UNKNOWN:
                hist[k].append(cur[k])
            dt = wall.perf_counter() - t0
            est = _fmt_est(cur, true_vals)
            print(f"    [{tag}] Adam {ep:>5}/{adam_epochs}  L={loss.item():.2e}"
                  f"  Ld={Ld_t.item():.2e}  Lp={Lp_t.item():.2e}  "
                  f"Lic={Lic_t.item():.2e}  lp={lam_phys_cur:.2f} | {est} "
                  f"[{dt:.0f}s]")

    # ---- Stage 2: L-BFGS joint polish ----
    if lbfgs_steps > 0:
        print(f"    [{tag}] L-BFGS ({lbfgs_steps}) ...")
        lbfgs = torch.optim.LBFGS(
            net_params + list(params.parameters()),
            lr=0.5, max_iter=20, history_size=50,
            line_search_fn="strong_wolfe")

        for step in range(1, lbfgs_steps + 1):
            def closure():
                lbfgs.zero_grad()
                Ld_t = Lic_t = Lp_t = 0.0
                for c in conds:
                    Ld, Lic = data_ic_loss(c)
                    Lp = physics_loss(c, n_colloc // 5)
                    Ld_t = Ld_t + Ld
                    Lic_t = Lic_t + Lic
                    Lp_t = Lp_t + Lp
                nC = len(conds)
                tot = (lam_data * Ld_t + lam_phys_cur * Lp_t
                       + lam_ic * Lic_t) / nC
                tot.backward()
                return tot
            loss = lbfgs.step(closure)
            if step % 50 == 0 or step == lbfgs_steps:
                cur = params.values()
                dt = wall.perf_counter() - t0
                print(f"    [{tag}] LBFGS {step:>4}/{lbfgs_steps}  "
                      f"L={float(loss.detach()):.2e} | {_fmt_est(cur, true_vals)}"
                      f" [{dt:.0f}s]")

    # ---- Stage 3: parameter refinement on FROZEN nets ----
    # The flexible state nets win the race to drive the residual down, starving
    # the parameter gradient. Freeze the (now data-pinned) nets and optimise
    # ONLY theta against the physics residual on a large fixed grid summed over
    # every condition. In INTEGRAL mode this is derivative-free gradient
    # matching — the frozen net's VALUES are exact, so the biased autodiff dz/dt
    # never enters and theta is recovered from the honest ODE constraint.
    if param_refine_steps > 0:
        print(f"    [{tag}] param-refine ({param_refine_steps}, frozen, "
              f"{residual_mode}) ...")
        for p in net_params:
            p.requires_grad_(False)
        colloc = []
        for c in conds:
            tc = torch.linspace(0, T, param_refine_colloc,
                                device=DEVICE).reshape(-1, 1)
            if residual_mode == "integral":
                with torch.no_grad():
                    z = c["net"](tc)
                dt = tc[1:] - tc[:-1]
                colloc.append(("int", tc, z.detach(), dt, c["w_res"],
                               c["forcing"]))
            else:
                tcg = tc.clone().requires_grad_(True)
                z, dz = time_derivatives(c["net"], tcg)
                colloc.append(("der", tc, z.detach(), dz.detach(), c["w_res"],
                               c["forcing"]))

        ref_opt = torch.optim.LBFGS(
            list(params.parameters()), lr=0.5, max_iter=20,
            history_size=50, line_search_fn="strong_wolfe")

        def ref_closure():
            ref_opt.zero_grad()
            Lp_t = 0.0
            for item in colloc:
                kind, tc, z, aux, w, forcing = item
                pp = physics_params(forcing)
                if kind == "int":
                    f = physics_rhs(tc, z, pp)
                    r = ((z[1:] - z[:-1]) - 0.5 * aux * (f[1:] + f[:-1])) * w
                else:
                    r = (aux - physics_rhs(tc, z, pp)) * w
                Lp_t = Lp_t + (r**2).mean()
            Lp_t = Lp_t / len(colloc)
            Lp_t.backward()
            return Lp_t

        for step in range(1, param_refine_steps + 1):
            loss = ref_opt.step(ref_closure)
            if step % 50 == 0 or step == param_refine_steps:
                cur = params.values()
                dt = wall.perf_counter() - t0
                nu = _count_under(cur, true_vals)
                print(f"    [{tag}] refine {step:>4}/{param_refine_steps}  "
                      f"Lp={float(loss.detach()):.2e}  under10%={nu}/36 | "
                      f"{_fmt_est(cur, true_vals)} [{dt:.0f}s]")

    # ---- evaluate ----
    sols = {}
    with torch.no_grad():
        t_eval = torch.linspace(0, T, 6000, device=DEVICE).reshape(-1, 1)
        for c in conds:
            z_eval = c["net"](t_eval).cpu().numpy()
            sol = {"t": t_eval.cpu().numpy().ravel()}
            for i, vn in enumerate(VAR_NAMES):
                sol[vn] = z_eval[:, i]
            sols[c["name"]] = sol

    final = params.values()
    fphys = total_phys()
    nu = _count_under(final, true_vals)
    print(f"  [{tag}] done  under10%={nu}/36  phys={fphys:.2e}  "
          f"W={final['W']:.3f} thetaP={final['thetaP']:.3f}  "
          f"({wall.perf_counter()-t0:.0f}s)")

    return dict(sols=sols, params=params, hist=hist, conds=conds,
                final=final, fphys=fphys, n_under=nu, safe=safe)


def train_inverse(name, refs_for_regime, *,
                  T=150.0,
                  width=256, depth=4,
                  n_fourier=16, fourier_sigma=4.0,
                  n_colloc=12_000,
                  n_data=60,
                  noise_std=0.005,
                  adam_epochs=3000,
                  lbfgs_steps=500,
                  param_refine_steps=400,
                  param_refine_colloc=8000,
                  lr=1e-3,
                  lr_param=5e-3,
                  lam_data=1.0,
                  lam_phys=1.0,
                  lam_ic=20.0,
                  adaptive_weights=True,
                  weight_every=200,
                  weight_beta=0.1,
                  rel_weight=False,
                  colloc_mode="random",
                  residual_mode="deriv",
                  activation="gelu",
                  siren_w0=30.0,
                  n_starts=1,
                  init_jitter=0.15,
                  seed=42,
                  log_every=100,
                  out_dir="."):
    """Multi-start inverse solve over MULTIPLE experimental conditions.

    One state network per condition (they see different trajectories under
    different RA forcing) but a SINGLE shared `InverseParams` — the biological
    parameters are common to every condition and must satisfy the ODE residual
    for all conditions at once, which restores identifiability relative to a
    single trajectory.

    Runs `n_starts` independent starts (start 0 = the deterministic +50%
    anchor init; the rest jitter the init lognormally) and keeps the one with
    the lowest final physics residual — the standard escape from the
    "low-loss-at-wrong-theta" basins the single-start PINN gets stuck in.
    """
    true_p = {**BASELINE, **REGIMES[name]}
    true_vals = {k: true_p[k] for k in UNKNOWN}

    best = None
    for s in range(n_starts):
        tag = f"{name[:4]}#{s}"
        res = _train_once(
            name, refs_for_regime, true_vals,
            T=T, width=width, depth=depth,
            n_fourier=n_fourier, fourier_sigma=fourier_sigma,
            n_colloc=n_colloc, n_data=n_data, noise_std=noise_std,
            adam_epochs=adam_epochs, lbfgs_steps=lbfgs_steps,
            param_refine_steps=param_refine_steps,
            param_refine_colloc=param_refine_colloc,
            lr=lr, lr_param=lr_param,
            lam_data=lam_data, lam_phys=lam_phys, lam_ic=lam_ic,
            adaptive_weights=adaptive_weights, weight_every=weight_every,
            weight_beta=weight_beta,
            rel_weight=rel_weight, colloc_mode=colloc_mode,
            residual_mode=residual_mode, activation=activation,
            siren_w0=siren_w0,
            seed=seed + 1000 * s, init_jitter=(0.0 if s == 0 else init_jitter),
            log_every=log_every, tag=tag)
        if best is None or res["fphys"] < best["fphys"]:
            best = res
        print(f"  >> {name} start {s}: under10%={res['n_under']}/36 "
              f"phys={res['fphys']:.2e}  (best so far "
              f"under10%={best['n_under']}/36 phys={best['fphys']:.2e})")

    # ---- persist the best start ----
    safe = best["safe"]
    for c in best["conds"]:
        torch.save(c["net"].state_dict(),
                   os.path.join(out_dir, f"{safe}_{c['name']}_net.pt"))
    torch.save(best["params"].state_dict(),
               os.path.join(out_dir, f"{safe}_params.pt"))
    with open(os.path.join(out_dir, f"{safe}_history.json"), "w") as fh:
        json.dump(best["hist"], fh)
    with open(os.path.join(out_dir, f"{safe}_recovered.json"), "w") as fh:
        json.dump({"recovered": best["final"], "true": true_vals,
                   "init": INIT_GUESS, "n_under10pct": best["n_under"],
                   "final_phys": best["fphys"]}, fh, indent=2)

    ctrl = CONDITIONS[0]["name"]
    obs = {"t": best["conds"][0]["t_arr"].ravel(),
           "y": best["conds"][0]["y_arr"], "idx": best["conds"][0]["idx"]}
    return best["sols"][ctrl], best["params"], best["hist"], obs, true_vals
