"""Load trained final models, compare against scipy reference.

Produces:
  compare_grid.png   3x3 PINN(solid) vs scipy(dashed) for 7 vars + stemness + RA input
  compare_errors.png per-variable relative-L2 error bars
and prints a quantitative convergence table.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BASELINE, REGIMES, Y0, VAR_NAMES, VAR_LABELS, DEVICE
from model import ForwardPINN
from odes import _ode_rhs, _ra_input
from metrics import stemness
from scipy.integrate import solve_ivp

import glob
RUN_DIR = max(glob.glob(os.path.join("runs", "*")), key=os.path.getmtime)
T = 150.0        # horizon the model was trained on (active-dynamics window)
N = 6000
XMAX = T         # plot/compare over the whole trained horizon

# ---- scipy reference (same settings as reference.py) -------------------
def ref_solution(name):
    p = {**BASELINE, **REGIMES[name]}
    t_eval = np.linspace(0, T, N)
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0,
                    t_eval=t_eval, method="Radau", rtol=1e-10, atol=1e-12)
    return t_eval, sol.y.T  # (N,7)

# ---- PINN solution from saved final weights ----------------------------
def pinn_solution(name):
    safe = name.replace(" ", "_").replace("/", "_")
    net = ForwardPINN(T_max=T, width=256, depth=4).to(DEVICE)
    net.load_state_dict(torch.load(
        os.path.join(RUN_DIR, f"{safe}_final.pt"), map_location=DEVICE))
    net.eval()
    with torch.no_grad():
        t = torch.linspace(0, T, N, device=DEVICE).reshape(-1, 1)
        y = net(t).cpu().numpy()
    return t.cpu().numpy().ravel(), y

def to_soldict(t, y):
    d = {"t": t}
    for i, vn in enumerate(VAR_NAMES):
        d[vn] = y[:, i]
    return d

def rel_l2(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

# ---- run ---------------------------------------------------------------
print("Loading references + PINN solutions ...")
refs, pinns = {}, {}
for name in REGIMES:
    refs[name]  = ref_solution(name)
    pinns[name] = pinn_solution(name)

# quantitative table -----------------------------------------------------
print("\n" + "=" * 78)
print(f"Relative L2 error  PINN vs scipy   (active window 0-{XMAX:.0f})")
print("=" * 78)
header = f"{'regime':<18s}" + "".join(f"{vl[:7]:>9s}" for vl in VAR_LABELS) + f"{'MEAN':>9s}"
print(header)
err_table = {}
for name in REGIMES:
    tr, yr = refs[name]
    tp, yp = pinns[name]
    w = tr <= XMAX
    errs = [rel_l2(yp[w, i], yr[w, i]) for i in range(7)]
    err_table[name] = errs
    row = f"{name:<18s}" + "".join(f"{e*100:>8.2f}%" for e in errs)
    row += f"{np.mean(errs)*100:>8.2f}%"
    print(row)

# ATRA-window error
print("\nATRA-window (40<=tau<=80) relative L2 error:")
for name in REGIMES:
    tr, yr = refs[name]
    tp, yp = pinns[name]
    m = (tr >= 40) & (tr <= 80)
    errs = [rel_l2(yp[m, i], yr[m, i]) for i in range(7)]
    print(f"  {name:<18s} mean {np.mean(errs)*100:6.2f}%   "
          f"RA={errs[5]*100:6.2f}%")

# ---- comparison grid 3x3 ----------------------------------------------
COLORS = dict(zip(REGIMES, ["tab:blue", "tab:green", "tab:purple", "tab:pink"]))
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
panels = list(zip(VAR_NAMES, VAR_LABELS))

for ax, (vn, vl) in zip(axes.flat[:7], panels):
    i = VAR_NAMES.index(vn)
    for name in REGIMES:
        tr, yr = refs[name]; tp, yp = pinns[name]
        ax.plot(tp, yp[:, i], "-", lw=2, color=COLORS[name], label=f"{name} PINN")
        ax.plot(tr, yr[:, i], "--", lw=1, color=COLORS[name], alpha=0.55)
    ax.axvspan(BASELINE["tau1"], BASELINE["tau2"], alpha=0.10, color="orange")
    ax.set_xlim(0, XMAX)
    ax.set_title(vl); ax.set_xlabel(r"$\tau$")

# panel 8: RA input mu_R(tau)
ax = axes.flat[7]
for name in REGIMES:
    p = {**BASELINE, **REGIMES[name]}
    t = np.linspace(0, XMAX, N)
    ax.plot(t, [_ra_input(x, p) for x in t], color=COLORS[name], lw=1, label=name)
ax.axvspan(BASELINE["tau1"], BASELINE["tau2"], alpha=0.10, color="orange")
ax.set_xlim(0, XMAX)
ax.set_title(r"RA input $\mu_R(\tau)$ (forcing)"); ax.set_xlabel(r"$\tau$")

# panel 9: stemness
ax = axes.flat[8]
for name in REGIMES:
    p = {**BASELINE, **REGIMES[name]}
    tr, yr = refs[name]; tp, yp = pinns[name]
    ax.plot(tp, stemness(to_soldict(tp, yp), p), "-", lw=2, color=COLORS[name])
    ax.plot(tr, stemness(to_soldict(tr, yr), p), "--", lw=1,
            color=COLORS[name], alpha=0.55)
ax.axvspan(BASELINE["tau1"], BASELINE["tau2"], alpha=0.10, color="orange")
ax.set_xlim(0, XMAX)
ax.set_title("Stemness index"); ax.set_xlabel(r"$\tau$")

axes.flat[0].legend(fontsize=7, ncol=1)
fig.suptitle("PINN (solid) vs scipy reference (dashed) — 4 regimes",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(RUN_DIR, "compare_grid.png")
fig.savefig(out, dpi=130); plt.close(fig)
print(f"\nSaved {out}")

# ---- error bar chart ---------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(7); w = 0.2
for k, name in enumerate(REGIMES):
    ax.bar(x + k*w, np.array(err_table[name]) * 100, w,
           label=name, color=COLORS[name])
ax.set_xticks(x + 1.5*w); ax.set_xticklabels(VAR_LABELS, rotation=20)
ax.set_ylabel("Relative L2 error (%)")
ax.set_title("PINN vs scipy reference — per-variable error")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out = os.path.join(RUN_DIR, "compare_errors.png")
fig.savefig(out, dpi=130); plt.close(fig)
print(f"Saved {out}")
