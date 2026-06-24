"""Re-render pinn7_*.png from saved final weights (no retraining).

Loads the trained final models + scipy references and calls plot_all,
which now zooms to the active-dynamics window (XMAX). Saves into plots/.
"""
import os
import glob
import numpy as np
import torch

from config import BASELINE, REGIMES, Y0, VAR_NAMES, DEVICE
from model import ForwardPINN
from odes import _ode_rhs
from plotting import plot_all
from scipy.integrate import solve_ivp

RUN_DIR = max(glob.glob(os.path.join("runs", "*")), key=os.path.getmtime)
T = 150.0
N = 6000


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
        os.path.join(RUN_DIR, f"{safe}_final.pt"), map_location=DEVICE))
    net.eval()
    with torch.no_grad():
        t = torch.linspace(0, T, N, device=DEVICE).reshape(-1, 1)
        y = net(t).cpu().numpy()
    d = {"t": t.cpu().numpy().ravel()}
    for i, vn in enumerate(VAR_NAMES):
        d[vn] = y[:, i]
    return d


print("Loading references + PINN solutions ...")
refs = {name: ref_solution(name) for name in REGIMES}
solutions = {name: pinn_solution(name) for name in REGIMES}

os.makedirs("plots", exist_ok=True)
os.chdir("plots")
plot_all(solutions, refs)
print("Re-rendered pinn7_*.png into plots/")
