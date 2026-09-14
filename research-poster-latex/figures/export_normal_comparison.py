"""Repository-only export of existing inverse-PINN fits and matched references.

Run in tmux. Poster builds use saved arrays; no fitting is performed here.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SOURCE = REPO / "PINN-inverse-pinn-boost"
RUN = SOURCE / "runs/20260711_203325_integral"
sys.path.insert(0, str(SOURCE))
from config import BASELINE, REGIMES, Y0, CONDITIONS
from model import ForwardPINN
from odes import _ode_rhs

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
t = np.linspace(0, 150, 6000)
conditions = ["noATRA", "ctrl"]
references, predictions, hashes, parameters = [], [], {}, {}
for condition in conditions:
    overrides = next(c["forcing"] for c in CONDITIONS if c["name"] == condition)
    p = {**BASELINE, **REGIMES["Normal"], **overrides}
    parameters[condition] = p
    sol = solve_ivp(lambda tau, z: _ode_rhs(tau, z, p), (0, 150), Y0,
                    t_eval=t, method="Radau", rtol=1e-10, atol=1e-12)
    assert sol.success, sol.message
    references.append(sol.y.T)
    checkpoint = RUN / f"Normal_{condition}_net.pt"
    net = ForwardPINN(T_max=150, width=256, depth=4)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    with torch.inference_mode():
        predictions.append(net(torch.tensor(t[:, None])).numpy())
    hashes[str(checkpoint.relative_to(REPO))] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    print(f"{condition}: DR={p['DR']}; reference and saved PINN exported", flush=True)

references, predictions = np.asarray(references), np.asarray(predictions)
relative_l2 = np.linalg.norm(predictions-references, axis=1)/np.linalg.norm(references, axis=1)
assert references.shape == predictions.shape == (2, 6000, 7)
assert np.isfinite(predictions).all()
np.savez_compressed(ROOT/"data/normal_treatment_comparison.npz", t=t,
    conditions=conditions, states=["b", "p", "h5", "h13", "m", "r", "c"],
    reference=references, pinn=predictions, relative_l2=relative_l2,
    DR=[parameters[c]["DR"] for c in conditions], tau1=40., tau2=88.)
for name in ["Normal_history.json", "Normal_recovered.json", "Normal.log"]:
    source = RUN/name
    hashes[str(source.relative_to(REPO))] = hashlib.sha256(source.read_bytes()).hexdigest()
    (ROOT/"data"/("normal_inverse_"+name.removeprefix("Normal_"))).write_bytes(source.read_bytes())
for name in ["config.py", "model.py", "training.py", "run_boost.py", "odes.py"]:
    source = SOURCE/name
    hashes[str(source.relative_to(REPO))] = hashlib.sha256(source.read_bytes()).hexdigest()
provenance = {"run": str(RUN.relative_to(REPO)), "regime": "Normal",
    "conditions": conditions, "parameters": parameters, "initial_state": Y0.tolist(),
    "reference": {"solver": "Radau", "rtol": 1e-10, "atol": 1e-12},
    "network": {"width": 256, "depth": 4, "activation": "GELU",
                "n_fourier": 16, "fourier_sigma": 4., "per_state_output_scaling": True},
    "training": {"kind": "integral inverse PINN", "n_conditions": 10,
                 "n_observations_per_condition": 150, "observation_noise_std": .002,
                 "n_biological_parameters": 36, "starts": 2, "selected_start": 0,
                 "adam_epochs": 2000, "lbfgs_steps": 150,
                 "physics_weight": "adaptive", "ic_weight": 20,
                 "loss_curve_scope": "saved Adam checkpoints, selected start; all ten conditions"},
    "relative_l2_percent": {c:dict(zip(["b","p","h5","h13","m","r","c"],
                                     (100*relative_l2[i]).tolist())) for i,c in enumerate(conditions)},
    "source_sha256": hashes, "trained_during_export": False}
(ROOT/"data/normal_treatment_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
print(json.dumps(provenance["relative_l2_percent"],indent=2), flush=True)
for i,c in enumerate(conditions):
    for j in [4,1]:
        a=references[i,t>=30,j]
        print(c,["b","p","h5","h13","m","r","c"][j],
              "range after tau=30:",float(a.min()),float(a.max()),flush=True)
