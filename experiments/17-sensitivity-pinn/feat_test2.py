import sys, torch, numpy as np, bench, sgn
from wnt_pinn.networks.state import ForwardPINN
NOBASE = len(sys.argv) > 5
reg = "Normal"; ng = int(sys.argv[1]); ci = int(sys.argv[2]); nrf = int(sys.argv[3]); fmax = float(sys.argv[4])
d = bench.make_data(reg, 42)[ci]
sd = torch.load(f"../10-inverse-integral-residual/runs/20260711_203325_integral/{reg}_{d['name']}_net.pt", map_location="cpu")
net = ForwardPINN(T_max=150., width=256, depth=4, n_fourier=16); net.load_state_dict(sd)
g = torch.Generator().manual_seed(0)
om = torch.rand(nrf // 2, generator=g) * 2 * np.pi * fmax       # rad / time
ph = torch.zeros(nrf // 2)
_orig = sgn.features
def feats(t):
    base = _orig(net, t)
    if NOBASE: base = base[:, -1:]
    if nrf == 0: return base
    a = t * om[None, :]
    # bump-free: sin/cos of random frequencies, plus a decaying set for the t=0 transient
    return torch.cat([base, torch.sin(a), torch.cos(a)], 1)
sgn.features = lambda n, t: feats(t)
c = sgn.Cond(net, d, ng, 100.)
A0 = torch.cat([net.net[-1].weight.detach().T, net.net[-1].bias.detach()[None]], 0)
c.A = torch.zeros(c.Hg.shape[1], 7)
if not NOBASE: c.A[:257] = A0
c.nh = c.A.shape[0]
x = torch.tensor(bench.to_x(bench.true_params(reg)))
tr = torch.tensor(d["t_ref"]).reshape(-1, 1); Hr = feats(tr)
def err(A): return float(((c.z(Hr, A).numpy() - d["y_ref"]) * d["w"]).__abs__().max())
A = c.A.clone()
print(d["name"], "nh", c.nh, "start err", err(A))
for lam in [1, 1e2, 1e4, 1e6]:
    for it in range(5):
        Ja, Jx, r = c.jacobian(A, x, lam)
        M = Ja.T @ Ja; gg = Ja.T @ r
        dA = -torch.linalg.solve(M + 1e-7 * torch.diag(torch.diag(M)) + 1e-9 * torch.eye(M.shape[0]), gg)
        A = A + dA.reshape(7, -1).T
    rd, rp, ric = c.residual(A, x, lam)
    print(f"lam={lam:.0e} data={float(rd@rd):.4e} (n={rd.numel()}) phys/lam={float(rp@rp)/lam:.3e} err={err(A):.3e}", flush=True)
