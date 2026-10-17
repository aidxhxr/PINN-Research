import sys, torch, numpy as np, bench, sgn
from wnt_pinn.networks.state import ForwardPINN
reg = "Normal"; ng = int(sys.argv[1]); ci = int(sys.argv[2])
d = bench.make_data(reg, 42)[ci]
sd = torch.load(f"../PINN-inverse-pinn-boost/runs/20260711_203325_integral/{reg}_{d['name']}_net.pt", map_location="cpu")
net = ForwardPINN(T_max=150., width=256, depth=4, n_fourier=16); net.load_state_dict(sd)
c = sgn.Cond(net, d, ng, 100.)
x = torch.tensor(bench.to_x(bench.true_params(reg)))
tr = torch.tensor(d["t_ref"]).reshape(-1, 1)
Hr = sgn.features(net, tr)
def err(A): return float(((c.z(Hr, A).numpy() - d["y_ref"]) * d["w"]).__abs__().max())
A = c.A.clone()
print(d["name"], "start: max weighted state err vs ref", err(A))
for lam in [1e-2, 1, 1e2, 1e4, 1e6]:
    for it in range(6):
        Ja, Jx, r = c.jacobian(A, x, lam)
        M = Ja.T @ Ja; g = Ja.T @ r
        dA = -torch.linalg.solve(M + 1e-6 * torch.diag(torch.diag(M)) + 1e-10 * torch.eye(M.shape[0]), g)
        A = A + dA.reshape(7, -1).T
    rd, rp, ric = c.residual(A, x, lam)
    print(f"lam={lam:.0e} data={float(rd@rd):.4e} (n={rd.numel()}) phys/lam={float(rp@rp)/lam:.3e} err={err(A):.3e}")
