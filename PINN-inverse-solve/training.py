import os
import json
import time as wall

import numpy as np
import torch

from config import (BASELINE, REGIMES, DEVICE, VAR_NAMES,
                    UNKNOWN, INIT_GUESS)
from model import ForwardPINN, InverseParams, time_derivatives
from residual import physics_residual


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


def train_inverse(name, t_ref, y_ref, *,
                  T=150.0,
                  width=256, depth=4,
                  n_fourier=16, fourier_sigma=4.0,
                  n_colloc=20_000,
                  n_data=80,
                  noise_std=0.0,
                  adam_epochs=2000,
                  lbfgs_steps=400,
                  lr=1e-3,
                  lr_param=5e-3,
                  lam_data=1.0,
                  lam_phys=1.0,
                  lam_ic=20.0,
                  seed=42,
                  log_every=100,
                  out_dir="."):
    torch.manual_seed(seed)
    true_p = {**BASELINE, **REGIMES[name]}            # ground truth (held out)
    true_vals = {k: true_p[k] for k in UNKNOWN}

    # sparse, possibly noisy observations of the trajectory
    t_arr, y_arr, idx = sample_sparse(t_ref, y_ref, n_data, noise_std, seed)
    t_d = torch.tensor(t_arr, device=DEVICE)
    y_d = torch.tensor(y_arr, device=DEVICE)
    y0  = torch.tensor(y_ref[0], device=DEVICE)        # IC anchor (known)

    net = ForwardPINN(T_max=T, width=width, depth=depth,
                      n_fourier=n_fourier, fourier_sigma=fourier_sigma).to(DEVICE)
    params = InverseParams(INIT_GUESS).to(DEVICE)
    n_net = sum(q.numel() for q in net.parameters())
    print(f"  arch {depth}×{width}  net params {n_net:,}  "
          f"unknowns {UNKNOWN}")
    print(f"  init guess {params.values()}  |  true {true_vals}")

    # network and physical parameters get separate (larger) learning rates
    opt = torch.optim.Adam([
        {"params": net.parameters(),    "lr": lr},
        {"params": params.parameters(), "lr": lr_param},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=adam_epochs, eta_min=1e-6)

    hist = dict(epoch=[], loss=[], Ld=[], Lp=[], Lic=[],
                **{k: [] for k in UNKNOWN})

    def physics_params():
        return {**BASELINE, **params.dict()}

    safe = name.replace(" ", "_").replace("/", "_")
    t0 = wall.perf_counter()

    for ep in range(1, adam_epochs + 1):
        opt.zero_grad()

        z_d = net(t_d)
        Ld  = ((z_d - y_d)**2).mean()
        Lic = ((net(torch.zeros(1, 1, device=DEVICE))[0] - y0)**2).mean()

        tc  = torch.rand(n_colloc, 1, device=DEVICE) * T
        zc, dzc = time_derivatives(net, tc)
        res = physics_residual(tc, zc, dzc, physics_params())
        Lp  = (res**2).mean()

        loss = lam_data*Ld + lam_phys*Lp + lam_ic*Lic
        loss.backward()
        opt.step()
        sched.step()

        if ep % log_every == 0 or ep == 1:
            cur = params.values()
            hist["epoch"].append(ep)
            hist["loss"].append(loss.item())
            hist["Ld"].append(Ld.item())
            hist["Lp"].append(Lp.item())
            hist["Lic"].append(Lic.item())
            for k in UNKNOWN:
                hist[k].append(cur[k])
            dt = wall.perf_counter() - t0
            est = "  ".join(f"{k}={cur[k]:.3f}(*{true_vals[k]:.2f})"
                            for k in UNKNOWN)
            print(f"    Adam {ep:>5}/{adam_epochs}  L={loss.item():.2e}  "
                  f"Ld={Ld.item():.2e}  Lp={Lp.item():.2e}  "
                  f"Lic={Lic.item():.2e}  | {est}  [{dt:.0f}s]")
            with open(os.path.join(out_dir, f"{safe}_history.json"), "w") as fh:
                json.dump(hist, fh)

    if lbfgs_steps > 0:
        print(f"    L-BFGS ({lbfgs_steps} steps, {n_colloc//5} colloc) ...")
        n_col_lb = n_colloc // 5
        lbfgs = torch.optim.LBFGS(
            list(net.parameters()) + list(params.parameters()),
            lr=0.5, max_iter=20, history_size=50,
            line_search_fn="strong_wolfe")

        for step in range(1, lbfgs_steps + 1):
            def closure():
                lbfgs.zero_grad()
                Ld_  = ((net(t_d) - y_d)**2).mean()
                Lic_ = ((net(torch.zeros(1, 1, device=DEVICE))[0]
                         - y0)**2).mean()
                tc_  = torch.rand(n_col_lb, 1, device=DEVICE) * T
                zc_, dzc_ = time_derivatives(net, tc_)
                Lp_ = (physics_residual(tc_, zc_, dzc_,
                                        physics_params())**2).mean()
                tot = lam_data*Ld_ + lam_phys*Lp_ + lam_ic*Lic_
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
                for k in UNKNOWN:
                    hist[k].append(cur[k])
                dt = wall.perf_counter() - t0
                est = "  ".join(f"{k}={cur[k]:.3f}(*{true_vals[k]:.2f})"
                                for k in UNKNOWN)
                print(f"    LBFGS {step:>4}/{lbfgs_steps}  "
                      f"L={float(loss):.2e}  | {est}  [{dt:.0f}s]")

    with torch.no_grad():
        t_eval = torch.linspace(0, T, 6000, device=DEVICE).reshape(-1, 1)
        z_eval = net(t_eval).cpu().numpy()

    sol = {"t": t_eval.cpu().numpy().ravel()}
    for i, vn in enumerate(VAR_NAMES):
        sol[vn] = z_eval[:, i]

    final = params.values()
    print(f"  recovered {final}  |  true {true_vals}  "
          f"({wall.perf_counter()-t0:.0f}s)")

    torch.save(net.state_dict(), os.path.join(out_dir, f"{safe}_net.pt"))
    torch.save(params.state_dict(), os.path.join(out_dir, f"{safe}_params.pt"))
    with open(os.path.join(out_dir, f"{safe}_history.json"), "w") as fh:
        json.dump(hist, fh)
    with open(os.path.join(out_dir, f"{safe}_recovered.json"), "w") as fh:
        json.dump({"recovered": final, "true": true_vals,
                   "init": INIT_GUESS}, fh, indent=2)

    obs = {"t": t_arr.ravel(), "y": y_arr, "idx": idx}
    return sol, net, params, hist, obs, true_vals
