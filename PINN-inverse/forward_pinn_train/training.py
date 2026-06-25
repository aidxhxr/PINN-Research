import os
import json
import time as wall

import numpy as np
import torch

from config import BASELINE, REGIMES, DEVICE, VAR_NAMES
from model import ForwardPINN, time_derivatives
from residual import physics_residual


def train_regime(name, t_ref, y_ref, *,
                 T=3000.0,
                 width=256, depth=4,
                 n_fourier=16, fourier_sigma=4.0,
                 n_colloc=200_000,
                 n_data=3000,
                 adam_epochs=5000,
                 lbfgs_steps=500,
                 lr=1e-3,
                 lam_data=0.7,
                 lam_phys=0.3,
                 seed=42,
                 log_every=250,
                 out_dir="."):
    torch.manual_seed(seed)
    p = {**BASELINE, **REGIMES[name]}

    idx = np.linspace(0, len(t_ref)-1, n_data, dtype=int)
    t_d = torch.tensor(t_ref[idx, None],  device=DEVICE)
    y_d = torch.tensor(y_ref[idx],        device=DEVICE)

    net = ForwardPINN(T_max=T, width=width, depth=depth,
                      n_fourier=n_fourier, fourier_sigma=fourier_sigma).to(DEVICE)
    n_params = sum(q.numel() for q in net.parameters())
    print(f"  arch {depth}×{width}  params {n_params:,}")

    hist = dict(epoch=[], loss=[], Ld=[], Lp=[])

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=adam_epochs, eta_min=1e-6)

    safe = name.replace(" ", "_").replace("/", "_")

    t0 = wall.perf_counter()
    for ep in range(1, adam_epochs + 1):
        opt.zero_grad()

        z_d = net(t_d)
        Ld  = ((z_d - y_d)**2).mean()

        tc  = torch.rand(n_colloc, 1, device=DEVICE) * T
        zc, dzc = time_derivatives(net, tc)
        res = physics_residual(tc, zc, dzc, p)
        Lp  = (res**2).mean()

        loss = lam_data * Ld + lam_phys * Lp
        loss.backward()
        opt.step()
        sched.step()

        if ep % log_every == 0 or ep == 1:
            hist["epoch"].append(ep)
            hist["loss"].append(loss.item())
            hist["Ld"].append(Ld.item())
            hist["Lp"].append(Lp.item())
            dt = wall.perf_counter() - t0
            print(f"    Adam {ep:>5}/{adam_epochs}  "
                  f"L={loss.item():.2e}  Ld={Ld.item():.2e}  "
                  f"Lp={Lp.item():.2e}  lr={sched.get_last_lr()[0]:.1e}  "
                  f"[{dt:.0f}s]")
            torch.save(net.state_dict(),
                       os.path.join(out_dir, f"{safe}_adam_ep{ep}.pt"))
            with open(os.path.join(out_dir, f"{safe}_history.json"), "w") as fh:
                json.dump(hist, fh)

    if lbfgs_steps > 0:
        print(f"    L-BFGS ({lbfgs_steps} steps, {n_colloc//5} colloc) ...")
        n_col_lb = n_colloc // 5
        lbfgs = torch.optim.LBFGS(
            net.parameters(), lr=0.5, max_iter=20,
            history_size=50, line_search_fn="strong_wolfe")

        for step in range(1, lbfgs_steps + 1):
            def closure():
                lbfgs.zero_grad()
                Ld_ = ((net(t_d) - y_d)**2).mean()
                tc_  = torch.rand(n_col_lb, 1, device=DEVICE) * T
                zc_, dzc_ = time_derivatives(net, tc_)
                Lp_ = (physics_residual(tc_, zc_, dzc_, p)**2).mean()
                tot = lam_data*Ld_ + lam_phys*Lp_
                tot.backward()
                return tot
            loss = lbfgs.step(closure)
            if step % 100 == 0 or step == lbfgs_steps:
                dt = wall.perf_counter() - t0
                print(f"    LBFGS {step:>4}/{lbfgs_steps}  "
                      f"L={loss:.2e}  [{dt:.0f}s]")
                torch.save(net.state_dict(),
                           os.path.join(out_dir, f"{safe}_lbfgs_step{step}.pt"))

    with torch.no_grad():
        t_eval = torch.linspace(0, T, 6000, device=DEVICE).reshape(-1, 1)
        z_eval = net(t_eval).cpu().numpy()

    sol = {"t": t_eval.cpu().numpy().ravel()}
    for i, vn in enumerate(VAR_NAMES):
        sol[vn] = z_eval[:, i]

    dt_total = wall.perf_counter() - t0
    print(f"  done  ({dt_total:.0f}s total)")

    torch.save(net.state_dict(), os.path.join(out_dir, f"{safe}_final.pt"))
    with open(os.path.join(out_dir, f"{safe}_history.json"), "w") as fh:
        json.dump(hist, fh)

    return sol, net, hist
