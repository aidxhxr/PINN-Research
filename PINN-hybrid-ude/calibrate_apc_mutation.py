"""Calibrate one APC-mutation neural mechanism across known severities.

The inverse PINN trains one process per biological regime, but thetaP is a
single constant inside a regime. Jointly learning an arbitrary function of
thetaP and thetaP itself from that one input value is non-identifiable.

This first stage therefore learns one shared function from synthetic reference
trajectories at known APC severities. It uses the same derivative-free
trapezoidal idea as the inverse PINN: only trajectory values enter the loss.
Intermediate severities are held out for an interpolation check. The resulting
network is frozen before the ordinary inverse PINN estimates thetaP.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

from config import BASELINE, CONDITIONS, Y0
from hybrid import APCMutationNN, true_term
from odes import _ode_rhs


TRAIN_THETA = (1.00, 0.75, 0.50, 0.25)
VALID_THETA = (0.875, 0.625, 0.375)
CAL_CONDITIONS = ("ctrl", "earlyATRA", "wntInhib")


def _forcing(name):
    for cond in CONDITIONS:
        if cond["name"] == name:
            return cond["forcing"]
    raise KeyError(name)


def make_records(theta_values, *, T, n_points, stride):
    """Generate exact synthetic trajectories, then retain a modest fixed grid."""
    records = []
    for theta in theta_values:
        for condition in CAL_CONDITIONS:
            params = {
                **BASELINE,
                "W": 0.80,
                "thetaP": theta,
                **_forcing(condition),
            }
            t_eval = np.linspace(0.0, T, n_points)
            sol = solve_ivp(
                lambda t, y: _ode_rhs(t, y, params),
                (0.0, T),
                Y0,
                t_eval=t_eval,
                method="Radau",
                rtol=1e-10,
                atol=1e-12,
            )
            if not sol.success:
                raise RuntimeError(f"theta={theta}/{condition}: {sol.message}")
            t = torch.as_tensor(sol.t[::stride].copy()).reshape(-1, 1)
            y = torch.as_tensor(sol.y.T[::stride].copy())
            records.append({
                "thetaP": theta,
                "condition": condition,
                "t": t,
                "y": y,
                "params": params,
            })
    return records


def apc_integral_loss(net, records):
    """Mean normalized APC trapezoidal residual over all trajectories."""
    total = torch.zeros((), dtype=torch.get_default_dtype())
    for record in records:
        t, y, params = record["t"], record["y"], record["params"]
        b, apc, h5, h13 = (y[:, i:i + 1] for i in range(4))
        loss = torch.full_like(apc, 1.0 - record["thetaP"])
        delta_p = 1.0 + net(loss)
        production = ((1.0 + params["rho5"] * h5) /
                      (1.0 + params["rhoB"] * b +
                       params["rho13"] * h13))
        rhs = (production - delta_p * apc) / params["epsP"]
        dt = t[1:] - t[:-1]
        residual = ((apc[1:] - apc[:-1]) / dt -
                    0.5 * (rhs[1:] + rhs[:-1]))
        scale = max(float(torch.max(torch.abs(apc))), 0.05)
        total = total + torch.mean((residual / scale) ** 2)
    return total / len(records)


def curve_metrics(net, theta_values):
    losses = np.asarray([1.0 - theta for theta in theta_values])
    learned = net.curve(losses)
    truth = true_term("apc_mutation", losses)
    rmse = float(np.sqrt(np.mean((learned - truth) ** 2)))
    denom = float(np.sqrt(np.mean(truth ** 2))) or 1.0
    return {
        "thetaP": list(theta_values),
        "apc_loss": losses.tolist(),
        "learned": learned.tolist(),
        "truth": truth.tolist(),
        "rmse": rmse,
        "nrmse": rmse / denom,
    }


def train_one(records, *, seed, adam_epochs, lbfgs_steps, wd, log_every):
    torch.manual_seed(seed)
    net = APCMutationNN(width=5, depth=2)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    history = []
    started = time.time()
    for epoch in range(1, adam_epochs + 1):
        opt.zero_grad(set_to_none=True)
        physics = apc_integral_loss(net, records)
        l2 = wd * net.l2()
        loss = physics + l2
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % log_every == 0 or epoch == adam_epochs:
            row = {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "physics": float(physics.detach()),
                "l2": float(l2.detach()),
            }
            history.append(row)
            print(
                f"  seed={seed} adam {epoch:5d}/{adam_epochs} "
                f"L={row['loss']:.3e} Lp={row['physics']:.3e} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    optimizer = torch.optim.LBFGS(
        net.parameters(),
        lr=0.5,
        max_iter=lbfgs_steps,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = apc_integral_loss(net, records) + wd * net.l2()
        loss.backward()
        return loss

    optimizer.step(closure)
    final = float(apc_integral_loss(net, records).detach())
    print(f"  seed={seed} final physics={final:.3e}", flush=True)
    return net, history, final


def plot_result(out_dir, grid, learned, truth, train_metrics, valid_metrics):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(grid, truth, "--", color="#cc3333", lw=2.2,
            label=r"synthetic truth $\delta_{P1}(1-\theta_P)$")
    ax.plot(grid, learned, color="#1f77b4", lw=2.4,
            label=r"learned $f_{\rm APC}(1-\theta_P)$")
    ax.scatter(train_metrics["apc_loss"], train_metrics["learned"],
               color="#2ca02c", s=48, zorder=3, label="calibration severities")
    ax.scatter(valid_metrics["apc_loss"], valid_metrics["learned"],
               facecolors="white", edgecolors="#9467bd", linewidths=1.8,
               s=58, zorder=3, label="held-out severities")
    ax.set_xlabel(r"APC functional loss $1-\theta_P$")
    ax.set_ylabel(r"excess degradation $\delta_P-1$")
    ax.set_title("Hybrid neural-mechanistic APC mutation term")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learned_term_apc_mutation.png"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--adam", type=int, default=2000)
    parser.add_argument("--lbfgs", type=int, default=250)
    parser.add_argument("--T", type=float, default=150.0)
    parser.add_argument("--points", type=int, default=2401)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--wd", type=float, default=1e-8)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Generating cross-severity APC calibration trajectories", flush=True)
    train_records = make_records(
        TRAIN_THETA, T=args.T, n_points=args.points, stride=args.stride)
    valid_records = make_records(
        VALID_THETA, T=args.T, n_points=args.points, stride=args.stride)

    best = None
    starts = []
    for start in range(args.starts):
        seed = 42 + 1000 * start
        net, history, final = train_one(
            train_records,
            seed=seed,
            adam_epochs=args.adam,
            lbfgs_steps=args.lbfgs,
            wd=args.wd,
            log_every=args.log_every,
        )
        starts.append({"start": start, "seed": seed, "physics": final})
        if best is None or final < best["physics"]:
            best = {
                "net": net,
                "history": history,
                "physics": final,
                "start": start,
                "seed": seed,
            }

    checkpoint = os.path.join(args.out, "apc_mutation.pt")
    torch.save(best["net"].state_dict(), checkpoint)
    train_metrics = curve_metrics(best["net"], TRAIN_THETA)
    valid_metrics = curve_metrics(best["net"], VALID_THETA)
    validation_physics = float(
        apc_integral_loss(best["net"], valid_records).detach())

    grid = np.linspace(0.0, 0.75, 401)
    learned = best["net"].curve(grid)
    truth = true_term("apc_mutation", grid)
    curve_rmse = float(np.sqrt(np.mean((learned - truth) ** 2)))
    curve_nrmse = curve_rmse / (float(np.sqrt(np.mean(truth ** 2))) or 1.0)
    diffs = np.diff(learned)

    result = {
        "term": "apc_mutation",
        "formula_replaced": "deltaP1 * (1 - thetaP)",
        "hybrid_formula": "deltaP = 1 + f_APC(1 - thetaP)",
        "train_thetaP": list(TRAIN_THETA),
        "valid_thetaP": list(VALID_THETA),
        "conditions": list(CAL_CONDITIONS),
        "fixed_W": 0.80,
        "best_start": best["start"],
        "best_seed": best["seed"],
        "train_physics": best["physics"],
        "validation_physics": validation_physics,
        "train_metrics": train_metrics,
        "validation_metrics": valid_metrics,
        "curve_rmse": curve_rmse,
        "curve_nrmse": curve_nrmse,
        "anchor_value": float(best["net"].curve([0.0])[0]),
        "minimum_output": float(np.min(learned)),
        "minimum_increment": float(np.min(diffs)),
        "starts": starts,
        "checkpoint": os.path.abspath(checkpoint),
    }
    with open(os.path.join(args.out, "apc_calibration.json"), "w") as handle:
        json.dump(result, handle, indent=2)

    plot_result(args.out, grid, learned, truth, train_metrics, valid_metrics)
    print(
        f"APC calibration complete: held-out NRMSE="
        f"{valid_metrics['nrmse']:.2%}, curve NRMSE={curve_nrmse:.2%}, "
        f"anchor={result['anchor_value']:.3e}, "
        f"min increment={result['minimum_increment']:.3e}",
        flush=True,
    )
    print(f"CHECKPOINT={os.path.abspath(checkpoint)}", flush=True)


if __name__ == "__main__":
    main()
