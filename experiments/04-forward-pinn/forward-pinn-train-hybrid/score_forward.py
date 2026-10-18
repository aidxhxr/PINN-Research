"""Score a trained forward PINN against the scipy Radau reference.

Writes a machine-readable table (JSON + text) so the numbers quoted in the
paper are backed by an artifact on disk rather than an ad-hoc print. Reports
BOTH relative L2 and NRMSE: APC settles near zero in the high-WNT regimes, so
its relative L2 is dominated by the small denominator and NRMSE is the fair
summary there.

Usage:  python3 score_forward.py <run_dir> <out_prefix>
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

from config import BASELINE, REGIMES, Y0, VAR_LABELS, DEVICE
from model import ForwardPINN
from odes import _ode_rhs

T = 150.0
N = 6000

run_dir = sys.argv[1]
out_prefix = sys.argv[2]


def ref_solution(name):
    p = {**BASELINE, **REGIMES[name]}
    t_eval = np.linspace(0, T, N)
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0,
                    t_eval=t_eval, method="Radau", rtol=1e-10, atol=1e-12)
    return t_eval, sol.y.T


def pinn_solution(name):
    safe = name.replace(" ", "_").replace("/", "_")
    net = ForwardPINN(T_max=T, width=256, depth=4).to(DEVICE)
    net.load_state_dict(torch.load(
        os.path.join(run_dir, f"{safe}_final.pt"), map_location=DEVICE))
    net.eval()
    with torch.no_grad():
        t = torch.linspace(0, T, N, device=DEVICE).reshape(-1, 1)
        y = net(t).cpu().numpy()
    return t.cpu().numpy().ravel(), y


results = {}
lines = []
lines.append(f"run_dir: {run_dir}")
lines.append(f"reference: solve_ivp Radau rtol=1e-10 atol=1e-12, T={T:.0f}, N={N}")
lines.append("")
lines.append("Relative L2 error (%)  PINN vs scipy reference")
lines.append("-" * 92)
lines.append(f"{'regime':<18s}" + "".join(f"{v[:8]:>9s}" for v in VAR_LABELS) + f"{'MEAN':>9s}")

for name in REGIMES:
    _, yr = ref_solution(name)
    _, yp = pinn_solution(name)
    rel = [float(np.linalg.norm(yp[:, i] - yr[:, i]) / np.linalg.norm(yr[:, i]))
           for i in range(7)]
    rng = [float(yr[:, i].max() - yr[:, i].min()) for i in range(7)]
    nrmse = [float(np.sqrt(np.mean((yp[:, i] - yr[:, i]) ** 2)) / rng[i])
             for i in range(7)]
    results[name] = {"rel_l2": rel, "nrmse": nrmse,
                     "rel_l2_mean": float(np.mean(rel)),
                     "nrmse_mean": float(np.mean(nrmse)),
                     "rel_l2_mean_no_apc": float(np.mean([rel[i] for i in range(7) if i != 1]))}
    lines.append(f"{name:<18s}" + "".join(f"{e*100:>8.2f}%" for e in rel)
                 + f"{np.mean(rel)*100:>8.2f}%")

lines.append("")
lines.append("NRMSE (%)  = RMSE / (max-min of reference)")
lines.append("-" * 92)
lines.append(f"{'regime':<18s}" + "".join(f"{v[:8]:>9s}" for v in VAR_LABELS) + f"{'MEAN':>9s}")
for name in REGIMES:
    n = results[name]["nrmse"]
    lines.append(f"{name:<18s}" + "".join(f"{e*100:>8.2f}%" for e in n)
                 + f"{np.mean(n)*100:>8.2f}%")

grand_rel = float(np.mean([results[n]["rel_l2_mean"] for n in REGIMES]))
grand_nrmse = float(np.mean([results[n]["nrmse_mean"] for n in REGIMES]))
grand_rel_no_apc = float(np.mean([results[n]["rel_l2_mean_no_apc"] for n in REGIMES]))
lines.append("")
lines.append(f"GRAND MEAN relative L2 : {grand_rel*100:.2f}%")
lines.append(f"GRAND MEAN rel L2 (no APC): {grand_rel_no_apc*100:.2f}%")
lines.append(f"GRAND MEAN NRMSE       : {grand_nrmse*100:.2f}%")

text = "\n".join(lines)
print(text)
with open(f"{out_prefix}.txt", "w") as f:
    f.write(text + "\n")
with open(f"{out_prefix}.json", "w") as f:
    json.dump({"run_dir": run_dir, "per_regime": results,
               "grand_mean_rel_l2": grand_rel,
               "grand_mean_rel_l2_no_apc": grand_rel_no_apc,
               "grand_mean_nrmse": grand_nrmse}, f, indent=2)
print(f"\nwrote {out_prefix}.txt and {out_prefix}.json")
