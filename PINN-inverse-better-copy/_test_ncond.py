"""Decisive: does adding conditions raise the identifiability ceiling?

Same gradient-match protocol (fit nets to data only, then L-BFGS params on
the physics residual), but with a configurable NUMBER of experimental
conditions. Compares how many of the 36 params land <10% as we add conditions.
Uses scipy to generate each condition's reference on the fly."""
import sys, time, numpy as np, torch
from scipy.integrate import solve_ivp
from config import BASELINE, REGIMES, DEVICE, UNKNOWN, INIT_GUESS, PARAM_RANGE, NOMINAL, Y0
from model import ForwardPINN, InverseParams, time_derivatives
from residual import physics_residual
from odes import _ode_rhs
from training import sample_sparse

T, N_DATA, NOISE = 150.0, 300, 0.001
torch.manual_seed(42)

# pool of experimental conditions (vary only the KNOWN RA-forcing protocol)
COND_POOL = [
    {"name": "ctrl",       "forcing": {}},
    {"name": "noATRA",     "forcing": {"DR": 0.0}},
    {"name": "earlyATRA",  "forcing": {"DR": 2.50, "tau1": 20.0, "tau2": 55.0}},
    {"name": "lateATRA",   "forcing": {"DR": 2.00, "tau1": 90.0, "tau2": 130.0}},
    {"name": "lowATRA",    "forcing": {"DR": 0.60, "tau1": 40.0, "tau2": 80.0}},
    {"name": "strongCirc", "forcing": {"AR": 0.12, "DR": 0.0}},
    {"name": "highMu",     "forcing": {"mu0": 0.70, "DR": 1.0}},
    {"name": "wideATRA",   "forcing": {"DR": 1.50, "tau1": 20.0, "tau2": 120.0}},
]
NCOND = int(sys.argv[1]) if len(sys.argv) > 1 else 6
conds_def = COND_POOL[:NCOND]
name = "Normal"
true_p = {**BASELINE, **REGIMES[name]}
true_vals = {k: true_p[k] for k in UNKNOWN}

def solve(forcing):
    p = {**true_p, **forcing}
    t_ref = np.linspace(0, T, 5000)
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0, T), Y0, t_eval=t_ref,
                    method="Radau", rtol=1e-10, atol=1e-12)
    return t_ref, sol.y.T

print(f"=== {NCOND} conditions: {[c['name'] for c in conds_def]} ===")
t0 = time.time()
conds = []
for ci, c in enumerate(conds_def):
    t_ref, y_ref = solve(c["forcing"])
    scale = np.maximum(np.abs(y_ref).max(axis=0), 0.05)
    net = ForwardPINN(T_max=T, width=256, depth=4, n_fourier=16,
                      fourier_sigma=4.0, out_scale=scale).to(DEVICE)
    t_arr, y_arr, _ = sample_sparse(t_ref, y_ref, N_DATA, NOISE, 42 + ci)
    conds.append(dict(forcing=c["forcing"], net=net,
                      t_d=torch.tensor(t_arr, device=DEVICE),
                      y_d=torch.tensor(y_arr, device=DEVICE)))

# Stage A: data-only fit
for c in conds:
    opt = torch.optim.Adam(c["net"].parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500, eta_min=1e-6)
    for ep in range(2500):
        opt.zero_grad()
        L = ((c["net"](c["t_d"]) - c["y_d"])**2).mean(); L.backward(); opt.step(); sched.step()
    for p in c["net"].parameters(): p.requires_grad_(False)
print(f"Stage A done [{time.time()-t0:.0f}s]")

# Stage B: gradient-match params
params = InverseParams(INIT_GUESS, PARAM_RANGE, NOMINAL).to(DEVICE)
def pp(forcing): return {**BASELINE, **params.dict(), **forcing}
NC = 8000
colloc = []
for c in conds:
    tc = (torch.rand(NC, 1, device=DEVICE) * T).requires_grad_(True)
    z, dz = time_derivatives(c["net"], tc)
    colloc.append((tc, z.detach(), dz.detach()))
lbfgs = torch.optim.LBFGS(list(params.parameters()), lr=0.5, max_iter=20,
                          history_size=50, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad(); L = 0.0
    for (tc, z, dz), c in zip(colloc, conds):
        L = L + (physics_residual(tc, z, dz, pp(c["forcing"]))**2).mean()
    L.backward(); return L
for step in range(1, 401):
    lbfgs.step(closure)

rec = params.values()
errs = {k: abs(rec[k]-true_vals[k])/abs(true_vals[k])*100 for k in UNKNOWN}
ok = sum(e <= 10 for e in errs.values())
print(f"\n=== {NCOND} conditions: {ok}/{len(UNKNOWN)} under 10%  "
      f"(W err={errs['W']:.1f}%  thetaP err={errs['thetaP']:.1f}%)  [{time.time()-t0:.0f}s] ===")
for k in sorted(UNKNOWN, key=lambda k: errs[k]):
    print(f"  {k:10s} err={errs[k]:6.1f}%" + ("  OK" if errs[k] <= 10 else ""))
