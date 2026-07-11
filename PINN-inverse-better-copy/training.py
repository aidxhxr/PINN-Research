import os
import json
import math
import time as wall

import numpy as np
import torch

from config import (BASELINE, REGIMES, CONDITIONS, DEVICE, VAR_NAMES,
                    UNKNOWN, INIT_GUESS, PARAM_RANGE, NOMINAL)
from model import ForwardPINN, InverseParams, time_derivatives
from residual import physics_residual


def _fmt_est(cur, true_vals):
    """Compact per-iteration log line: W and thetaP explicitly, plus the
    mean relative error over all unknowns (a full 36-param line would be
    unreadable). Every estimate is still stored in *_history.json."""
    errs = [abs(cur[k] - true_vals[k]) / abs(true_vals[k])
            for k in cur if true_vals[k] != 0]
    mean_err = 100.0 * sum(errs) / len(errs)
    return (f"W={cur['W']:.3f}(*{true_vals['W']:.2f})  "
            f"thetaP={cur['thetaP']:.3f}(*{true_vals['thetaP']:.2f})  "
            f"<rel.err>={mean_err:.0f}%")


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


def train_inverse(name, refs_for_regime, *,
                  T=150.0,
                  width=256, depth=4,
                  n_fourier=16, fourier_sigma=4.0,
                  n_colloc=12_000,
                  n_data=60,
                  noise_std=0.005,
                  adam_epochs=3000,
                  lbfgs_steps=500,
                  lr=1e-3,
                  lr_param=5e-3,
                  lam_data=1.0,
                  lam_phys=1.0,
                  lam_ic=20.0,
                  adaptive_weights=True,
                  weight_every=200,
                  weight_beta=0.1,
                  seed=42,
                  log_every=100,
                  out_dir="."):
    """Joint inverse solve over MULTIPLE experimental conditions.

    One state network per condition (they see different trajectories under
    different RA forcing) but a SINGLE shared `InverseParams` — the
    biological parameters are common to every condition. The shared
    parameters must satisfy the ODE residual for all conditions at once,
    which is what restores identifiability relative to a single trajectory.
    """
    torch.manual_seed(seed)
    true_p = {**BASELINE, **REGIMES[name]}
    true_vals = {k: true_p[k] for k in UNKNOWN}

    # one shared parameter set for all conditions
    params = InverseParams(INIT_GUESS, PARAM_RANGE, NOMINAL).to(DEVICE)

    # build a per-condition setup (own net + own sparse data + forcing)
    conds = []
    for ci, c in enumerate(CONDITIONS):
        t_ref, y_ref = refs_for_regime[c["name"]]
        # per-variable output scale (order-1 network outputs)
        scale = np.maximum(np.abs(y_ref).max(axis=0), 0.05)
        net = ForwardPINN(T_max=T, width=width, depth=depth,
                          n_fourier=n_fourier, fourier_sigma=fourier_sigma,
                          out_scale=scale).to(DEVICE)
        t_arr, y_arr, idx = sample_sparse(t_ref, y_ref, n_data,
                                          noise_std, seed + ci)
        conds.append(dict(
            name=c["name"],
            forcing=c["forcing"],
            net=net,
            t_d=torch.tensor(t_arr, device=DEVICE),
            y_d=torch.tensor(y_arr, device=DEVICE),
            y0=torch.tensor(y_ref[0], device=DEVICE),
            t_arr=t_arr, y_arr=y_arr, idx=idx,
        ))

    net_params = [p for c in conds for p in c["net"].parameters()]
    n_net = sum(p.numel() for p in conds[0]["net"].parameters())
    print(f"  arch {depth}×{width}  net params {n_net:,} × {len(conds)} cond  "
          f"unknowns {len(UNKNOWN)}")
    print(f"  conditions {[c['name'] for c in conds]}")
    print(f"  init guess W={INIT_GUESS['W']} thetaP={INIT_GUESS['thetaP']} "
          f"(+50% on the rest)  |  true W={true_vals['W']} "
          f"thetaP={true_vals['thetaP']}")

    opt = torch.optim.Adam([
        {"params": net_params,           "lr": lr},
        {"params": params.parameters(),  "lr": lr_param},
    ])

    # Decoupled schedules: the net cosine-anneals almost to zero, but the
    # PHYSICAL parameters keep a healthy learning rate (>=0.3x) so they do
    # not freeze long before they have converged. (In PINN-inverse-solve a
    # single CosineAnnealingLR drove BOTH groups to 1e-6, stalling the
    # parameters.)
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

    def losses_for(c, n_pts):
        """data, IC, physics losses for one condition."""
        Ld = ((c["net"](c["t_d"]) - c["y_d"])**2).mean()
        Lic = ((c["net"](torch.zeros(1, 1, device=DEVICE))[0] - c["y0"])**2
               ).mean()
        tc = torch.rand(n_pts, 1, device=DEVICE) * T
        zc, dzc = time_derivatives(c["net"], tc)
        Lp = (physics_residual(tc, zc, dzc, physics_params(c["forcing"]))**2
              ).mean()
        return Ld, Lic, Lp

    safe = name.replace(" ", "_").replace("/", "_")
    t0 = wall.perf_counter()
    lam_phys_cur = float(lam_phys)

    for ep in range(1, adam_epochs + 1):
        # ---- adaptive data/physics balancing (grad-norm, Wang et al.) ----
        if adaptive_weights and (ep % weight_every == 0 or ep == 1):
            Ld_s = sum(((c["net"](c["t_d"]) - c["y_d"])**2).mean()
                       for c in conds)
            Lp_s = 0.0
            for c in conds:
                tc = torch.rand(n_colloc // 4, 1, device=DEVICE) * T
                zc, dzc = time_derivatives(c["net"], tc)
                Lp_s = Lp_s + (physics_residual(
                    tc, zc, dzc, physics_params(c["forcing"]))**2).mean()
            g_d = _grad_norm(lam_data * Ld_s, net_params)
            g_p = _grad_norm(Lp_s, net_params)
            target = float((g_d / (g_p + 1e-30)).item())
            target = min(max(target, 1e-2), 1e3)
            lam_phys_cur = (1 - weight_beta) * lam_phys_cur + weight_beta * target

        opt.zero_grad()
        Ld_t = Lic_t = Lp_t = 0.0
        for c in conds:
            Ld, Lic, Lp = losses_for(c, n_colloc)
            Ld_t = Ld_t + Ld
            Lic_t = Lic_t + Lic
            Lp_t = Lp_t + Lp
        nC = len(conds)
        Ld_t, Lic_t, Lp_t = Ld_t/nC, Lic_t/nC, Lp_t/nC

        loss = lam_data*Ld_t + lam_phys_cur*Lp_t + lam_ic*Lic_t
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
            print(f"    Adam {ep:>5}/{adam_epochs}  L={loss.item():.2e}  "
                  f"Ld={Ld_t.item():.2e}  Lp={Lp_t.item():.2e}  "
                  f"Lic={Lic_t.item():.2e}  λp={lam_phys_cur:.2f}  | {est}  "
                  f"[{dt:.0f}s]")
            with open(os.path.join(out_dir, f"{safe}_history.json"), "w") as fh:
                json.dump(hist, fh)

    if lbfgs_steps > 0:
        print(f"    L-BFGS ({lbfgs_steps} steps) ...")
        n_col_lb = n_colloc // 5
        lbfgs = torch.optim.LBFGS(
            net_params + list(params.parameters()),
            lr=0.5, max_iter=20, history_size=50,
            line_search_fn="strong_wolfe")

        for step in range(1, lbfgs_steps + 1):
            def closure():
                lbfgs.zero_grad()
                Ld_t = Lic_t = Lp_t = 0.0
                for c in conds:
                    Ld, Lic, Lp = losses_for(c, n_col_lb)
                    Ld_t = Ld_t + Ld
                    Lic_t = Lic_t + Lic
                    Lp_t = Lp_t + Lp
                nC = len(conds)
                tot = (lam_data*Ld_t + lam_phys_cur*Lp_t + lam_ic*Lic_t) / nC
                tot.backward()
                return tot
            loss = lbfgs.step(closure)
            if step % 50 == 0 or step == lbfgs_steps:
                cur = params.values()
                hist["epoch"].append(adam_epochs + step)
                hist["loss"].append(float(loss.detach()))
                hist["Ld"].append(float("nan"))
                hist["Lp"].append(float("nan"))
                hist["Lic"].append(float("nan"))
                hist["lam_phys"].append(lam_phys_cur)
                for k in UNKNOWN:
                    hist[k].append(cur[k])
                dt = wall.perf_counter() - t0
                est = _fmt_est(cur, true_vals)
                print(f"    LBFGS {step:>4}/{lbfgs_steps}  "
                      f"L={float(loss.detach()):.2e}  | {est}  [{dt:.0f}s]")

    # evaluate each condition's recovered trajectory (ctrl is primary)
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
    print(f"  recovered W={final['W']:.3f} thetaP={final['thetaP']:.3f}  "
          f"({wall.perf_counter()-t0:.0f}s)")

    for c in conds:
        torch.save(c["net"].state_dict(),
                   os.path.join(out_dir, f"{safe}_{c['name']}_net.pt"))
    torch.save(params.state_dict(), os.path.join(out_dir, f"{safe}_params.pt"))
    with open(os.path.join(out_dir, f"{safe}_history.json"), "w") as fh:
        json.dump(hist, fh)
    with open(os.path.join(out_dir, f"{safe}_recovered.json"), "w") as fh:
        json.dump({"recovered": final, "true": true_vals,
                   "init": INIT_GUESS}, fh, indent=2)

    ctrl = CONDITIONS[0]["name"]
    obs = {"t": conds[0]["t_arr"].ravel(),
           "y": conds[0]["y_arr"], "idx": conds[0]["idx"]}
    return sols[ctrl], params, hist, obs, true_vals
