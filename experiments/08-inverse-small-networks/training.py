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
                  param_refine_steps=400,
                  param_refine_colloc=8000,
                  lr=1e-3,
                  lr_param=5e-3,
                  net_weight_decay=1e-4,
                  net_steps=5,
                  par_steps=5,
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

    Two changes over `PINN-inverse-multicond` attack the low PINN recovery
    count (gradient starvation + net overfitting), WITHOUT adding data:

    * SMALLER, REGULARISED state nets (see width/depth in main.py + the
      `net_weight_decay` here). The 207k-param nets of the base folder are
      wildly over-capacity for a smooth 7-D trajectory: they both overfit the
      noisy observations and absorb the physics residual for whatever wrong
      parameters currently hold, starving the 36 parameters of gradient. A
      smaller, weight-decayed net cannot do either, so gradient is forced back
      into the parameters. In an inverse PINN, less overfitting == better
      recovery (the two goals coincide, they do not trade off).

    * ALTERNATING (two-timescale / coordinate) optimisation instead of one
      joint Adam step. Each cycle takes `net_steps` net-only updates (on
      data+physics+IC) then `par_steps` parameter-only updates (on the physics
      residual). The net no longer wins the race to drive the residual to zero
      before the parameters have moved — they get a sustained gradient every
      cycle.
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

    # SEPARATE optimisers so the two groups can be stepped independently in
    # the alternating scheme (below). Weight decay is applied to the NET ONLY
    # — never to the physical parameters (decaying them would bias every
    # unknown toward zero).
    opt_net = torch.optim.Adam(net_params, lr=lr,
                               weight_decay=net_weight_decay)
    opt_par = torch.optim.Adam(params.parameters(), lr=lr_param)

    # Decoupled schedules: the net cosine-anneals almost to zero, but the
    # PHYSICAL parameters keep a healthy learning rate (>=0.3x) so they do
    # not freeze long before they have converged. (In PINN-inverse-solve a
    # single CosineAnnealingLR drove BOTH groups to 1e-6, stalling the
    # parameters.) Each scheduler advances once per step of its own group;
    # both groups take `adam_epochs` steps total, so the cosine period matches.
    def lam_net(ep):
        r = 1e-3
        return r + (1 - r) * 0.5 * (1 + math.cos(math.pi * ep / adam_epochs))

    def lam_par(ep):
        r = 0.3
        return r + (1 - r) * 0.5 * (1 + math.cos(math.pi * ep / adam_epochs))

    sched_net = torch.optim.lr_scheduler.LambdaLR(opt_net, lam_net)
    sched_par = torch.optim.lr_scheduler.LambdaLR(opt_par, lam_par)

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

    def phys_loss(n_pts):
        """Physics residual only (the parameter phase's objective), averaged
        over conditions. The physical parameters enter only here."""
        Lp_t = 0.0
        for c in conds:
            tc = torch.rand(n_pts, 1, device=DEVICE) * T
            zc, dzc = time_derivatives(c["net"], tc)
            Lp_t = Lp_t + (physics_residual(
                tc, zc, dzc, physics_params(c["forcing"]))**2).mean()
        return Lp_t / len(conds)

    ep = 0
    while ep < adam_epochs:
        # ---- adaptive data/physics balancing (grad-norm, Wang et al.) ----
        # Recomputed at cycle boundaries; the net requires grad here.
        if adaptive_weights and (ep % weight_every == 0):
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

        # ---------- NET PHASE: net-only updates on data+physics+IC ----------
        for _ in range(net_steps):
            if ep >= adam_epochs:
                break
            opt_net.zero_grad()
            opt_par.zero_grad()
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
            opt_net.step()
            sched_net.step()
            ep += 1

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
                with open(os.path.join(out_dir,
                                       f"{safe}_history.json"), "w") as fh:
                    json.dump(hist, fh)

        # -------- PARAM PHASE: parameter-only updates on the residual -------
        # Freeze the nets so the residual gradient flows only to the physical
        # parameters — they can no longer be starved by the far more flexible
        # state nets soaking up the residual first.
        for p in net_params:
            p.requires_grad_(False)
        for _ in range(par_steps):
            opt_par.zero_grad()
            Lp_only = phys_loss(n_colloc)
            Lp_only.backward()
            opt_par.step()
            sched_par.step()
        for p in net_params:
            p.requires_grad_(True)

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

    # ------------------------------------------------------------------
    # Stage 3 — PARAMETER REFINEMENT (gradient matching on a frozen net)
    # ------------------------------------------------------------------
    # In the joint stages the very flexible state nets win the race to drive
    # the physics residual down, so once they fit the data the gradient
    # flowing to the 36 physical parameters is starved and they freeze at the
    # wrong value (the "low loss at wrong params" signature; see session log
    # 2026-06-30). Here we FREEZE the nets — which now track the true
    # trajectory because the data term pinned them — and optimise ONLY the
    # parameters against the physics residual on a large, fixed collocation
    # set summed over every condition. With the trajectory (and its autodiff
    # derivative) held fixed this is a well-conditioned gradient-matching
    # least-squares: the residual is large at the wrong params (the true
    # trajectory does not satisfy the wrong-param ODE) so the parameters get a
    # strong, sustained gradient that the joint stage denied them.
    if param_refine_steps > 0:
        print(f"    param-refine ({param_refine_steps} steps, nets frozen) ...")
        for p in net_params:
            p.requires_grad_(False)
        # precompute each frozen net's state + derivative on a fixed grid
        colloc = []
        for c in conds:
            tc = (torch.rand(param_refine_colloc, 1, device=DEVICE) * T
                  ).requires_grad_(True)
            zc, dzc = time_derivatives(c["net"], tc)
            colloc.append((tc, zc.detach(), dzc.detach(), c["forcing"]))

        ref_opt = torch.optim.LBFGS(
            list(params.parameters()), lr=0.5, max_iter=20,
            history_size=50, line_search_fn="strong_wolfe")

        def ref_closure():
            ref_opt.zero_grad()
            Lp_t = 0.0
            for tc, zc, dzc, forcing in colloc:
                Lp_t = Lp_t + (physics_residual(
                    tc, zc, dzc, physics_params(forcing))**2).mean()
            Lp_t = Lp_t / len(colloc)
            Lp_t.backward()
            return Lp_t

        for step in range(1, param_refine_steps + 1):
            loss = ref_opt.step(ref_closure)
            if step % 50 == 0 or step == param_refine_steps:
                cur = params.values()
                hist["epoch"].append(adam_epochs + lbfgs_steps + step)
                hist["loss"].append(float(loss.detach()))
                hist["Ld"].append(float("nan"))
                hist["Lp"].append(float(loss.detach()))
                hist["Lic"].append(float("nan"))
                hist["lam_phys"].append(lam_phys_cur)
                for k in UNKNOWN:
                    hist[k].append(cur[k])
                dt = wall.perf_counter() - t0
                est = _fmt_est(cur, true_vals)
                print(f"    refine {step:>4}/{param_refine_steps}  "
                      f"Lp={float(loss.detach()):.2e}  | {est}  [{dt:.0f}s]")

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
