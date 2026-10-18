"""Decisive experiment: does the TRAJECTORY carry the parameter info?

Stage A: fit one state net per condition to DENSE data ONLY (no physics) so
         the net == true trajectory (and its autodiff derivative == true dz/dt).
Stage B: freeze the nets, recover the 36 params by GRADIENT MATCHING — L-BFGS
         on params alone minimizing the physics residual ||dz/dt - RHS||^2
         over a large collocation set, summed across all 3 conditions.

If W -> 0.80 here, the info is in the data and the joint-training optimisation
was the culprit (param gradient starvation). If W stays ~0.98, the problem is
deeper (net-derivative quality / structural)."""
import os, time, numpy as np, torch
from config import (BASELINE, REGIMES, CONDITIONS, DEVICE, UNKNOWN,
                    INIT_GUESS, PARAM_RANGE, NOMINAL, VAR_NAMES)
from model import ForwardPINN, InverseParams, time_derivatives
from residual import physics_residual
from reference import generate_references
from training import sample_sparse

T = 150.0
N_DATA = 400
NOISE = 0.001
torch.manual_seed(42)

refs = generate_references(T=T, n_pts=5000)
name = "Normal"
true_p = {**BASELINE, **REGIMES[name]}
true_vals = {k: true_p[k] for k in UNKNOWN}

conds = []
for ci, c in enumerate(CONDITIONS):
    t_ref, y_ref = refs[name][c["name"]]
    scale = np.maximum(np.abs(y_ref).max(axis=0), 0.05)
    net = ForwardPINN(T_max=T, width=256, depth=4, n_fourier=16,
                      fourier_sigma=4.0, out_scale=scale).to(DEVICE)
    t_arr, y_arr, idx = sample_sparse(t_ref, y_ref, N_DATA, NOISE, 42 + ci)
    conds.append(dict(name=c["name"], forcing=c["forcing"], net=net,
                      t_d=torch.tensor(t_arr, device=DEVICE),
                      y_d=torch.tensor(y_arr, device=DEVICE)))

# ---- Stage A: data-only fit of each net ----
print("Stage A: fit nets to data only ...")
t0 = time.time()
for c in conds:
    opt = torch.optim.Adam(c["net"].parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2500, eta_min=1e-6)
    for ep in range(2500):
        opt.zero_grad()
        L = ((c["net"](c["t_d"]) - c["y_d"])**2).mean()
        L.backward(); opt.step(); sched.step()
    # check fit vs the (noise-free) reference
    t_ref, y_ref = refs[name][c["name"]]
    with torch.no_grad():
        pred = c["net"](torch.tensor(t_ref[:, None], device=DEVICE)).cpu().numpy()
    rmse = np.sqrt(((pred - y_ref)**2).mean())
    print(f"  {c['name']:10s} data L={L.item():.2e}  traj RMSE vs ref={rmse:.4f}  [{time.time()-t0:.0f}s]")

# ---- Stage B: gradient-match params (nets frozen) ----
print("\nStage B: gradient-match params (nets frozen) ...")
for c in conds:
    for p in c["net"].parameters():
        p.requires_grad_(False)
params = InverseParams(INIT_GUESS, PARAM_RANGE, NOMINAL).to(DEVICE)

def phys_params(forcing):
    return {**BASELINE, **params.dict(), **forcing}

# fixed large collocation set per condition (precompute net states once)
NC = 8000
colloc = []
for c in conds:
    tc = (torch.rand(NC, 1, device=DEVICE) * T).requires_grad_(True)
    z, dz = time_derivatives(c["net"], tc)
    colloc.append((tc, z.detach(), dz.detach()))

lbfgs = torch.optim.LBFGS(list(params.parameters()), lr=0.5, max_iter=20,
                          history_size=50, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad()
    L = 0.0
    for (tc, z, dz), c in zip(colloc, conds):
        L = L + (physics_residual(tc, z, dz, phys_params(c["forcing"]))**2).mean()
    L.backward(); return L
for step in range(1, 301):
    L = lbfgs.step(closure)
    if step % 50 == 0 or step == 1:
        cur = params.values()
        me = 100*np.mean([abs(cur[k]-true_vals[k])/abs(true_vals[k]) for k in UNKNOWN])
        print(f"  step {step:4d}  Lp={float(L):.2e}  W={cur['W']:.3f}(*0.80) "
              f"thetaP={cur['thetaP']:.3f}(*1.00)  <err>={me:.0f}%  [{time.time()-t0:.0f}s]")

rec = params.values()
errs = {k: abs(rec[k]-true_vals[k])/abs(true_vals[k])*100 for k in UNKNOWN}
ok = sum(e <= 10 for e in errs.values())
print(f"\n=== gradient-match result: {ok}/{len(UNKNOWN)} under 10% ===")
for k in sorted(UNKNOWN, key=lambda k: errs[k]):
    print(f"  {k:10s} rec={rec[k]:8.3f} true={true_vals[k]:7.3f}  err={errs[k]:6.1f}%"
          + ("  OK" if errs[k] <= 10 else ""))
