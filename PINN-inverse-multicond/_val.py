"""Validate the multicond recipe on Normal only (reduced epochs).
Checks whether 6 conditions + dense data + the param-refinement stage beats
the old 6/36-under-10%. Writes to a scratch dir, not runs/."""
import os, numpy as np
from config import REGIMES, UNKNOWN, CONDITIONS
from reference import generate_references
from training import train_inverse

OUT = "/tmp/claude-13253/-home-29-aidahxr-PINN-Research/241c42cf-0e25-4df5-9ae4-a57d0b23335d/scratchpad/val_mc"
os.makedirs(OUT, exist_ok=True)
print(f"{len(CONDITIONS)} conditions: {[c['name'] for c in CONDITIONS]}")
refs = generate_references(T=150.0, n_pts=5000)
name = "Normal"
sol, params, hist, obs, true_vals = train_inverse(
    name, refs[name], T=150.0,
    width=256, depth=4, n_fourier=16, fourier_sigma=4.0,
    n_colloc=10_000, n_data=150, noise_std=0.002,
    adam_epochs=1500, lbfgs_steps=300,
    param_refine_steps=400, param_refine_colloc=10_000,
    lr=1e-3, lr_param=5e-3, lam_data=1.0, lam_phys=1.0, lam_ic=20.0,
    adaptive_weights=True, out_dir=OUT)
rec = params.values()
errs = {k: abs(rec[k]-true_vals[k])/abs(true_vals[k])*100 for k in UNKNOWN}
ok = sum(e <= 10 for e in errs.values())
print(f"\n=== VAL Normal: {ok}/{len(UNKNOWN)} under 10%  "
      f"(W={errs['W']:.1f}%  thetaP={errs['thetaP']:.1f}%) ===")
for k in sorted(UNKNOWN, key=lambda k: errs[k]):
    print(f"  {k:10s} rec={rec[k]:8.3f} true={true_vals[k]:7.3f} err={errs[k]:6.1f}%"
          + ("  OK" if errs[k] <= 10 else ""))
