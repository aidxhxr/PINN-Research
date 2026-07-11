"""Regenerate the standard inverse-PINN plots for a run_boost.py run dir.

run_boost.py fans the 4 regimes out as separate processes and only writes the
per-regime artifacts (``*_history.json``, ``*_recovered.json``, ``*_net.pt``) —
it never calls plotting.py, so its run dirs have no PNGs (unlike main.py runs).
This script rebuilds the same plot set (losses, param convergence, param error,
recovery bars, per-state trajectories, stemness) from those saved artifacts, so
completed run_boost runs get the same figures as the other folders.

Usage:
    python3 regen_plots.py runs/20260702_160102_integral
    python3 regen_plots.py runs/20260702_160102_integral --no-states  # fast

The state plots reload the saved ctrl-condition nets and reconstruct the sparse
observations; everything else comes straight from the JSON histories.
"""
import argparse
import json
import os
import pickle
import re

import numpy as np
import torch

from config import (REGIMES, CONDITIONS, VAR_NAMES, UNKNOWN, INIT_GUESS,
                    DEVICE)
from model import ForwardPINN
from reference import generate_references
from training import sample_sparse
import plotting

# knobs run_boost.py trains with (kept in sync with run_boost.py / main.py)
WIDTH, DEPTH, N_FOURIER, FOURIER_SIGMA = 256, 4, 16, 4.0
N_DATA, NOISE_STD, BASE_SEED = 150, 0.002, 42
CTRL = CONDITIONS[0]["name"]

# variant -> net activation (from run_boost.py VARIANTS); detected off dir name
VARIANT_ACT = {"integral": "gelu", "plumb": "gelu", "base": "gelu",
               "siren": "siren"}


def _safe(name):
    return name.replace(" ", "_")


def _detect_activation(run_dir):
    tail = os.path.basename(os.path.normpath(run_dir)).lower()
    for variant, act in VARIANT_ACT.items():
        if tail.endswith(variant):
            return act, variant
    return "gelu", "unknown"


def _best_start_seed(run_dir, name, final_phys):
    """Recover the winning multi-start seed by matching the per-start physics
    residual printed in ``<safe>.log`` to the saved ``final_phys`` (so the
    reconstructed observation scatter matches the net that was actually kept).
    Falls back to start 0 (seed=BASE_SEED) if the log is unavailable."""
    log = os.path.join(run_dir, f"{_safe(name)}.log")
    if not os.path.exists(log):
        return BASE_SEED
    pat = re.compile(rf">> {re.escape(name)} start (\d+):.*phys=([0-9.eE+-]+)")
    best_s, best_d = 0, float("inf")
    with open(log) as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                s, phys = int(m.group(1)), float(m.group(2))
                d = abs(phys - final_phys)
                if d < best_d:
                    best_s, best_d = s, d
    return BASE_SEED + 1000 * best_s


def _load_refs(run_dir, T):
    cache = os.path.join(run_dir, "refs.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    print("  refs.pkl not found — regenerating references (scipy) ...")
    return generate_references(T=T, n_pts=5000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--no-states", action="store_true",
                    help="skip the net-reload state/stemness plots (fast)")
    ap.add_argument("--T", type=float, default=150.0)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    act, variant = _detect_activation(run_dir)

    # ---- load per-regime histories + recovered params ----
    all_hist, recovered, true_params = {}, {}, {}
    missing = []
    for name in REGIMES:
        safe = _safe(name)
        hp = os.path.join(run_dir, f"{safe}_history.json")
        rp = os.path.join(run_dir, f"{safe}_recovered.json")
        if not (os.path.exists(hp) and os.path.exists(rp)):
            missing.append(name)
            continue
        with open(hp) as fh:
            all_hist[name] = json.load(fh)
        with open(rp) as fh:
            rec = json.load(fh)
        recovered[name] = rec["recovered"]
        true_params[name] = rec["true"]

    if missing:
        print(f"  ! skipping regimes with no artifacts yet: {missing}")
    if len(all_hist) < len(REGIMES):
        # plotting.py iterates over all REGIMES, so we can only draw the
        # multi-regime figures once every regime has finished.
        print(f"  only {len(all_hist)}/{len(REGIMES)} regimes complete — the "
              f"plots span all regimes, so run again once the rest finish.")
        return

    print(f"[regen] {run_dir}\n  variant={variant} activation={act}  "
          f"regimes={list(all_hist)}")

    # plotting.py writes PNGs into the CWD — point that at the run dir.
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        print("  losses / param-convergence / param-error / recovery-bars ...")
        plotting.plot_losses(all_hist)
        plotting.plot_param_convergence(all_hist, true_params)
        plotting.plot_param_error(all_hist, true_params)
        plotting.plot_recovery_bars(recovered, true_params)

        # ---- best-8 (best-converging) companion figures ----
        best8 = plotting.best_params(recovered, true_params, k=8)
        print(f"  best-8 params: {best8}")
        plotting.plot_param_convergence(
            all_hist, true_params, params=best8,
            fname="inv_param_convergence_best8.png",
            title_suffix="  — 8 best-converging params")
        plotting.plot_param_error(
            all_hist, true_params, params=best8,
            fname="inv_param_error_best8.png")
        plotting.plot_recovery_bars(
            recovered, true_params, params=best8,
            fname="inv_recovery_bars_best8.png",
            suptitle="Inverse PINN — true vs recovered (8 best-converging params)")

        if not args.no_states:
            print("  reloading ctrl nets for state + stemness plots ...")
            refs = _load_refs(run_dir, args.T)
            solutions, observations = {}, {}
            for name in REGIMES:
                t_ref, y_ref = refs[name][CTRL]
                t_ref = np.asarray(t_ref); y_ref = np.asarray(y_ref)
                scale = np.maximum(np.abs(y_ref).max(axis=0), 0.05)
                net = ForwardPINN(T_max=args.T, width=WIDTH, depth=DEPTH,
                                  n_fourier=N_FOURIER,
                                  fourier_sigma=FOURIER_SIGMA,
                                  out_scale=scale, activation=act).to(DEVICE)
                sd = torch.load(
                    os.path.join(run_dir, f"{_safe(name)}_{CTRL}_net.pt"),
                    map_location=DEVICE)
                net.load_state_dict(sd)
                net.eval()
                with torch.no_grad():
                    t_eval = torch.linspace(0, args.T, 6000,
                                            device=DEVICE).reshape(-1, 1)
                    z = net(t_eval).cpu().numpy()
                sol = {"t": t_eval.cpu().numpy().ravel()}
                for i, vn in enumerate(VAR_NAMES):
                    sol[vn] = z[:, i]
                solutions[name] = sol

                with open(os.path.join(
                        run_dir, f"{_safe(name)}_recovered.json")) as fh:
                    final_phys = json.load(fh)["final_phys"]
                seed = _best_start_seed(run_dir, name, final_phys)
                t_arr, y_arr, idx = sample_sparse(t_ref, y_ref, N_DATA,
                                                  NOISE_STD, seed)  # ctrl=ci 0
                observations[name] = {"t": t_arr.ravel(), "y": y_arr,
                                      "idx": idx}
            plotting.plot_states(solutions, refs, observations)
    finally:
        os.chdir(cwd)

    pngs = sorted(f for f in os.listdir(run_dir) if f.endswith(".png"))
    print(f"  wrote {len(pngs)} PNGs -> {run_dir}")
    for f in pngs:
        print(f"    {f}")


if __name__ == "__main__":
    main()
