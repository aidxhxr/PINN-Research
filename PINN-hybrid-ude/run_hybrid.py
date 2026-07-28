"""Single-regime driver for the neural-mechanistic hybrid (UDE) ablation.

Same shape as run_boost.py -- one regime per OS process so the 4 regimes run in
parallel on the one GPU -- but the VARIANT here selects the HYBRID configuration
rather than the residual plumbing (which is fixed at the proven `integral`
setting: relative weighting + deterministic collocation + derivative-free
trapezoidal residual + multi-start).

The hybrid knobs travel via ENV VARS because config.py must see them at import
time (UNKNOWN shrinks when a term's parameters are absorbed by the net), and
each parallel regime is its own interpreter.

`control` is the FULL mechanistic model (all 36 unknowns, hybrid off) -- i.e. a
re-run of pinn-boost `integral` under identical seeds. aggregate.py then scores
every variant on the INTERSECTION of recovered parameters, so the hybrid's 34
are compared against the control's same 34 and the denominator is honest.

Usage:
    HYBRID_TERM=ra_h5 python3 run_hybrid.py --regime Normal --variant ra_h5 \
        --out <dir> --starts 3 --threads 14
"""
import argparse
import os
import pickle

import torch

# NOTE: env must be set BEFORE config is imported (see run_hybrid.sh).
from config import REGIMES, HYBRID_TERM, UNKNOWN
from reference import generate_references
from training import train_inverse


# Residual plumbing is FIXED at the winning pinn-boost `integral` setting so the
# only thing varying across variants is the hybrid itself.
INTEGRAL = dict(rel_weight=True, colloc_mode="fixed",
                residual_mode="integral", activation="gelu", n_starts=3)

# Variant name -> the HYBRID_TERM the launcher must have exported. Kept here as
# documentation and as a hard consistency check against what config imported.
VARIANTS = {
    "control":     dict(term=None),             # full mechanistic model, 36
    "ra_h5":       dict(term="ra_h5"),          # headline
    "ra_h5_nc":    dict(term="ra_h5"),          # + HYBRID_CONSTRAINT=none
    "ra_h5_wdlo":  dict(term="ra_h5"),          # + HYBRID_WD=0    (no reg)
    "ra_h5_wdhi":  dict(term="ra_h5"),          # + HYBRID_WD=1e-6 (dominates: shows the units trap)
    "bm_myc":      dict(term="bm_myc"),         # H3 test: anchor actually BINDS here
    "bm_myc_nc":   dict(term="bm_myc"),        # + HYBRID_CONSTRAINT=none
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=list(REGIMES))
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--adam", type=int, default=2000)
    ap.add_argument("--lbfgs", type=int, default=150)
    ap.add_argument("--refine", type=int, default=600)
    ap.add_argument("--T", type=float, default=150.0)
    ap.add_argument("--refs", default=None)
    args = ap.parse_args()

    want = VARIANTS[args.variant]["term"]
    if HYBRID_TERM != want:
        raise SystemExit(
            f"variant {args.variant!r} expects HYBRID_TERM={want!r} but config "
            f"imported {HYBRID_TERM!r} -- the launcher must export it BEFORE "
            f"python starts.")

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    knobs = dict(INTEGRAL, n_starts=args.starts)
    print(f"=== regime={args.regime}  variant={args.variant}  "
          f"hybrid_term={HYBRID_TERM}  n_unknown={len(UNKNOWN)}  "
          f"knobs={knobs} ===", flush=True)

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
