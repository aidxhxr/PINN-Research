"""Quick single-regime validation: does denser, lower-noise data un-stick
the parameters (esp. W) that froze at the wrong value in the 60-pt run?

Runs ONLY the Normal regime with reduced epochs so we can see the trend in
~12-15 min before committing to a full 4-regime run. Prints per-param error
at the end. Writes nothing into runs/ (uses a scratch out_dir)."""
import sys, os, json
import numpy as np

from config import REGIMES, UNKNOWN
from reference import generate_references
from training import train_inverse

OUT = "/tmp/claude-13253/-home-29-aidahxr-PINN-Research/241c42cf-0e25-4df5-9ae4-a57d0b23335d/scratchpad/test_dense"
os.makedirs(OUT, exist_ok=True)

# config knobs from CLI:  n_data noise_std adam lbfgs [tag]
n_data    = int(sys.argv[1]) if len(sys.argv) > 1 else 300
noise_std = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
adam      = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
lbfgs     = int(sys.argv[4]) if len(sys.argv) > 4 else 400
tag       = sys.argv[5] if len(sys.argv) > 5 else "dense"

print(f"=== TEST {tag}: n_data={n_data} noise={noise_std} adam={adam} lbfgs={lbfgs} ===")
refs = generate_references(T=150.0, n_pts=5000)

name = "Normal"
sol, params, hist, obs, true_vals = train_inverse(
    name, refs[name], T=150.0,
    width=256, depth=4, n_fourier=16, fourier_sigma=4.0,
    n_colloc=12_000, n_data=n_data, noise_std=noise_std,
    adam_epochs=adam, lbfgs_steps=lbfgs,
    lr=1e-3, lr_param=5e-3,
    lam_data=1.0, lam_phys=1.0, lam_ic=20.0,
    adaptive_weights=True, out_dir=OUT)

rec = params.values()
errs = {k: abs(rec[k]-true_vals[k])/abs(true_vals[k])*100 for k in UNKNOWN}
order = sorted(UNKNOWN, key=lambda k: errs[k])
print(f"\n=== {tag} per-param error (best->worst) ===")
ok = 0
for k in order:
    flag = "OK" if errs[k] <= 10 else ""
    if errs[k] <= 10: ok += 1
    print(f"  {k:10s} rec={rec[k]:8.3f} true={true_vals[k]:7.3f}  err={errs[k]:6.1f}%  {flag}")
print(f"\n{ok}/{len(UNKNOWN)} under 10%  |  W err={errs['W']:.1f}%  thetaP err={errs['thetaP']:.1f}%")
