"""Export existing forward-PINN predictions and Radau references, without training.

Repository-only preparation step. Normal poster builds use the saved NPZ and
do not need Torch, checkpoints, or the research model outside the source ZIP.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "PINN-smaller/forward-pinn-train-hybrid"
RUN = SOURCE / "runs/20260712_204546"
sys.path.insert(0, str(SOURCE))
from config import BASELINE, REGIMES, Y0
from model import ForwardPINN
from odes import _ode_rhs

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
t = np.linspace(0, 150, 6000)
net_t = torch.linspace(0, 150, 6000).reshape(-1, 1)
names = list(REGIMES)
predictions, references, checkpoints = [], [], {}
for name in names:
    checkpoint = RUN / (name.replace(" ", "_") + "_final.pt")
    net = ForwardPINN(T_max=150, width=256, depth=4)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    with torch.inference_mode():
        predictions.append(net(net_t).numpy())
    parameters = {**BASELINE, **REGIMES[name]}
    sol = solve_ivp(lambda tau, z: _ode_rhs(tau, z, parameters), (0, 150), Y0,
                    t_eval=t, method="Radau", rtol=1e-10, atol=1e-12)
    assert sol.success, sol.message
    references.append(sol.y.T)
    checkpoints[name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    print(f"Exported {name}: saved PINN + Radau; shape {sol.y.T.shape}", flush=True)

predictions, references = np.asarray(predictions), np.asarray(references)
assert predictions.shape == references.shape == (4, 6000, 7)
assert np.isfinite(predictions).all() and np.isfinite(references).all()
np.savez_compressed(ROOT / "data/forward_trajectories.npz", t=t, regimes=names,
                    states=["b", "p", "h5", "h13", "m", "r", "c"],
                    pinn=predictions, reference=references,
                    tau1=BASELINE["tau1"], tau2=BASELINE["tau2"])
(ROOT / "data/forward_trajectories_provenance.json").write_text(json.dumps({
    "run": str(RUN.relative_to(ROOT.parent)),
    "checkpoint_sha256": checkpoints, "state_order": ["b", "p", "h5", "h13", "m", "r", "c"],
    "reference": {"solver": "Radau", "rtol": 1e-10, "atol": 1e-12},
    "t_span": [0, 150], "n_plot_points": len(t), "n_training_observations": 40,
    "pulse": [BASELINE["tau1"], BASELINE["tau2"]],
    "baseline": BASELINE, "regimes": REGIMES,
    "trained_during_export": False,
}, indent=2) + "\n")
print("Saved reusable forward_trajectories.npz and provenance.", flush=True)
