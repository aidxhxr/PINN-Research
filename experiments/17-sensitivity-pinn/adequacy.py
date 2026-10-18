"""Feature adequacy at the TRUE theta: per condition, fit the head alone
(GN over A at fixed x) at increasing lam; report data chi2/n (1 = noise
level) and max weighted state error vs the Radau reference. Isolates the
representational floor of a head basis from the optimisation of theta."""
import argparse, json, os, torch, numpy as np, bench, sgn
from wnt_pinn.networks.state import ForwardPINN
ap = argparse.ArgumentParser()
ap.add_argument("regime"); ap.add_argument("--pinn-dir", default=None)
ap.add_argument("--n-rff", type=int, default=512); ap.add_argument("--f-max", type=float, default=0.5)
ap.add_argument("--n-spline", type=int, default=0); ap.add_argument("--n-grid", type=int, default=3000)
ap.add_argument("--conds", default="all"); ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = torch.device(a.device); safe = a.regime.replace(" ", "_")
x = torch.tensor(bench.to_x(bench.true_params(a.regime)), device=dev)
data = bench.make_data(a.regime, 42)
sel = range(len(data)) if a.conds == "all" else [int(v) for v in a.conds.split(",")]
tot = {}
for ci in sel:
    d = data[ci]; net = None
    if a.pinn_dir:
        sd = torch.load(os.path.join(a.pinn_dir, f"{safe}_{d['name']}_net.pt"), map_location="cpu")
        net = ForwardPINN(T_max=150., width=256, depth=4, n_fourier=16); net.load_state_dict(sd)
    f = sgn.Features(net, a.n_rff, a.f_max, 0, a.n_spline)
    c = sgn.Cond(f, d, a.n_grid, 100., dev, net)
    if net is None:
        c.smooth_fit_head(f)
    tr = torch.tensor(d["t_ref"][::5]).reshape(-1, 1); Hr = f(tr).to(dev)
    A = c.A.clone(); line = []
    for lam in [1., 10., 100., 1000.]:
        for it in range(4):
            Aa, Bx, ga, Xx, gx = c.normal_eq(A, x, lam)
            dA = -torch.linalg.solve(Aa + 1e-9 * torch.diag(torch.diag(Aa)) + 1e-12 * torch.diag(Aa).mean() * torch.eye(Aa.shape[0], device=dev), ga)
            A = A + dA.reshape(7, -1).T
        rd, rp, ric = c.residual(A, x, lam)
        err = float((((c.z(Hr, A).cpu().numpy() - d["y_ref"][::5]) * d["w"])).__abs__().max())
        chi = float(rd @ rd) / rd.numel()
        tot.setdefault(lam, []).append(chi)
        line.append(f"lam={lam:.0e}: chi2/n={chi:.3f} err={err:.1e}")
    print(f"{d['name']:10s} nh={c.nh} | " + " | ".join(line), flush=True)
print("mean chi2/n:", {k: round(float(np.mean(v)), 3) for k, v in tot.items()})
