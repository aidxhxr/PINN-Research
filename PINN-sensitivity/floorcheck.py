"""Where in time does the head basis fail to satisfy the physics at TRUE theta?"""
import sys, torch, numpy as np, bench, sgn
from wnt_pinn.networks.state import ForwardPINN
reg = sys.argv[1]; ci = int(sys.argv[2]); extra = sys.argv[3] if len(sys.argv) > 3 else ""
P = "../PINN-inverse-pinn-boost/runs/20260711_203325_integral"; safe = reg.replace(" ", "_")
torch.set_num_threads(24); dev = torch.device("cpu")
d = bench.make_data(reg, 42)[ci]
sd = torch.load(f"{P}/{safe}_{d['name']}_net.pt", map_location="cpu")
net = ForwardPINN(T_max=150., width=256, depth=4, n_fourier=16); net.load_state_dict(sd)
f = sgn.Features(net, 512, 0.5, 0)
if extra == "exp":
    f = sgn.Features(net, 512, 0.5, 0, 0, n_exp=24)
c = sgn.Cond(f, d, 3000, 100., dev, net)
x = torch.tensor(bench.to_x(bench.true_params(reg)))
for lam in [100., 1000.]:
    A, cost = sgn.head_solve(c, c.A, x, lam, 15, 1e-10)
    rd, rp, ric = c.residual(A, x, lam)
    R2 = (rp.reshape(-1, 7) ** 2).sum(1).numpy()
    t = c.t[1:, 0].numpy()
    bins = [0, 1, 3, 10, 30, 60, 100, 150]
    share = [R2[(t >= a) & (t < b)].sum() / R2.sum() for a, b in zip(bins[:-1], bins[1:])]
    print(f"{d['name']} {extra or 'base'} lam={lam:.0e} data={float(rd@rd):.4e} phys/lam={float(rp@rp)/lam:.3e} "
          f"share of phys residual by t-bin {list(zip(bins[:-1], np.round(share, 3)))}", flush=True)
