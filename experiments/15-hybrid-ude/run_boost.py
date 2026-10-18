"""Single-regime driver for the PINN-boost ablation.

Trains ONE regime (with internal multi-start) under a named VARIANT and writes
<regime>_recovered.json into the run dir. Kept single-regime so the 4 regimes
can run as 4 parallel OS processes on the one big GPU (64 cores split across
them), which is how we exploit the hardware instead of a 4x-longer serial run.

Usage:
    python3 run_boost.py --regime "Normal" --variant integral --out <dir> \
                         --starts 3 --threads 14
"""
import argparse
import os
import pickle

import torch

from config import REGIMES
from reference import generate_references
from training import train_inverse


# ----------------------------------------------------------------------
# VARIANTS — each is a set of the levers layered on the multicond PINN.
#   base    : the levers-off control (== the excite Phase-1 PINN, for anchor)
#   plumb   : + relative weighting + deterministic collocation + multi-start
#   integral: plumb + derivative-FREE trapezoidal residual (the main lever)
#   siren   : plumb + SIREN sinusoidal net (faithful autodiff dz/dt)
# ----------------------------------------------------------------------
VARIANTS = {
    "base": dict(rel_weight=False, colloc_mode="random", residual_mode="deriv",
                 activation="gelu", n_starts=1),
    "plumb": dict(rel_weight=True, colloc_mode="fixed", residual_mode="deriv",
                  activation="gelu", n_starts=3),
    "integral": dict(rel_weight=True, colloc_mode="fixed",
                     residual_mode="integral", activation="gelu", n_starts=3),
    "siren": dict(rel_weight=True, colloc_mode="fixed", residual_mode="deriv",
                  activation="siren", siren_w0=30.0, n_starts=3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=list(REGIMES))
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--adam", type=int, default=2000)
    ap.add_argument("--lbfgs", type=int, default=150)
    ap.add_argument("--refine", type=int, default=600)
    ap.add_argument("--T", type=float, default=150.0)
    ap.add_argument("--refs", default=None,
                    help="pickle cache of generate_references(); built if absent")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    knobs = dict(VARIANTS[args.variant])
    if args.starts is not None:
        knobs["n_starts"] = args.starts

    print(f"=== regime={args.regime}  variant={args.variant}  "
          f"knobs={knobs} ===", flush=True)

    # references are built ONCE by run_boost.sh and cached; each parallel
    # regime process just loads its slice (must happen before any CUDA op so
    # the scipy fork-pool is clean).
    if args.refs and os.path.exists(args.refs):
        with open(args.refs, "rb") as fh:
            refs = pickle.load(fh)[args.regime]
    else:
        refs = generate_references(T=args.T)[args.regime]

    train_inverse(
        args.regime, refs,
        T=args.T,
        width=256, depth=4, n_fourier=16, fourier_sigma=4.0,
        n_colloc=8_000, n_data=150, noise_std=0.002,
        adam_epochs=args.adam, lbfgs_steps=args.lbfgs,
        param_refine_steps=args.refine, param_refine_colloc=8_000,
        lr=1e-3, lr_param=5e-3, lam_data=1.0, lam_phys=1.0, lam_ic=20.0,
        adaptive_weights=True, seed=42, log_every=200,
        out_dir=args.out, **knobs)


if __name__ == "__main__":
    main()
