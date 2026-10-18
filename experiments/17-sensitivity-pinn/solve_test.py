"""How accurately can the TangentNet solve state + sensitivity equations
physics-only? Train at the TRUE x for one regime, report state error vs the
Radau reference and the data cost ratio vs the noise floor."""
import sys, time, torch, numpy as np
import bench, tpinn
reg = sys.argv[1]; n_adam = int(sys.argv[2]); n_lbfgs = int(sys.argv[3])
ncol = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
dev = torch.device("cuda")
prob = tpinn.Problem(reg, 42, dev, ncol, 256, 4, 16, 4.0, 0)
x = torch.tensor(bench.to_x(bench.true_params(reg)), device=dev)
netp = prob.nets_params()
t0 = time.time()

def err():
    out = []
    with torch.no_grad():
        for c, d in zip(prob.conds, prob.data):
            t = torch.tensor(d["t_ref"][::10], device=dev).reshape(-1, 1)
            z, _ = prob.state(c, t, x, x)
            out.append(float(((z.cpu().numpy() - d["y_ref"][::10]) * d["w"]).__abs__().max()))
    return max(out), float(np.mean(out))

def show(tag):
    Ld, Ls, Lsen = prob.losses(x, x)
    JtJ, Jtr, F, cost = prob.gn_system(x, x)
    noise = sum(((torch.tensor(d["y_d"]) - torch.tensor(d["y_ref"][np.searchsorted(d["t_ref"], d["t_d"])])) * torch.tensor(d["w"])).pow(2).sum() for d in prob.data)
    print(f"{tag}: Ls={float(Ls):.2e} Lsen={float(Lsen):.2e} maxerr={err()[0]:.2e} meanerr={err()[1]:.2e} "
          f"cost={cost:.4e} noise_cost={float(noise):.4e} [{time.time()-t0:.0f}s]", flush=True)

opt = torch.optim.Adam(netp, lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_adam, 1e-5)
for ep in range(1, n_adam + 1):
    opt.zero_grad(); Ld, Ls, Lsen = prob.losses(x, x); (Ls + Lsen).backward(); opt.step(); sched.step()
    if ep % 1000 == 0: show(f"adam {ep}")
lb = torch.optim.LBFGS(netp, lr=1, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
for k in range(1, n_lbfgs + 1):
    def cl():
        lb.zero_grad(); Ld, Ls, Lsen = prob.losses(x, x); l = Ls + Lsen; l.backward(); return l
    lb.step(cl)
    if k % 25 == 0: show(f"lbfgs {k}")
