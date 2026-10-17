"""At a given theta, does the head solve reproduce the ODE solution?"""
import sys, torch, numpy as np, bench, sgn
from wnt_pinn.model.parameters import BASELINE, UNKNOWN
from wnt_pinn.networks.state import ForwardPINN
reg = sys.argv[1]; cis = [int(v) for v in sys.argv[2].split(",")]; lam = float(sys.argv[3])
P = "../PINN-inverse-pinn-boost/runs/20260711_203325_integral"; safe = reg.replace(" ", "_")
raw = torch.load(f"{P}/{safe}_params.pt", map_location="cpu")
rec = {k: (1 / (1 + np.exp(-float(raw[f"raw.{k}"]))) if k == "thetaP" else BASELINE[k] * np.exp(float(raw[f"raw.{k}"]))) for k in UNKNOWN}
x = torch.tensor(bench.to_x(rec)); dev = torch.device("cpu"); torch.set_num_threads(32)
data = bench.make_data(reg, 42)
for ci in cis:
    d = data[ci]
    sd = torch.load(f"{P}/{safe}_{d['name']}_net.pt", map_location="cpu")
    net = ForwardPINN(T_max=150., width=256, depth=4, n_fourier=16); net.load_state_dict(sd)
    f = sgn.Features(net, 512, 0.5, 0); c = sgn.Cond(f, d, 3000, 100., dev, net)
    tt = d["t_ref"][::5]
    yode = bench.solve({**BASELINE, **rec, **d["forcing"]}, tt)
    rode = ((bench.solve({**BASELINE, **rec, **d["forcing"]}, d["t_d"]) - d["y_d"]) * d["w"] / bench.NOISE)
    A, cost = sgn.head_solve(c, c.A, x, lam, 15, 1e-10)
    zh = c.z(f(torch.tensor(tt).reshape(-1, 1)), A).numpy()
    rd, rp, ric = c.residual(A, x, lam)
    e = np.abs((zh - yode) * d["w"]).max(0)
    print(f"{d['name']:10s} ODE-at-theta data Sr2={float((rode**2).sum()):.3e} | head data Sr2={float(rd@rd):.3e} phys={float(rp@rp):.3e} | max |head-ODE| per var {e.round(3)} at t={tt[np.abs((zh-yode)*d['w']).argmax(0)].round(1)}", flush=True)
