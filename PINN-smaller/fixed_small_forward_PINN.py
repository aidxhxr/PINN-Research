# Converted from fixed_small_forward_PINN.ipynb (repo path: PINN-smaller/fixed_small_forward_PINN.ipynb) on 2026-09-23.
# Cells appear in notebook order; code is verbatim. Lines that were IPython-only
# ("!shell", "%magic") are kept but commented with "[ipython-only, skipped]".
# Run from this directory:  cd PINN-smaller && python3 fixed_small_forward_PINN.py

# %% [cell 1]
# [ipython-only, skipped] !pip3 install numpy

# %% [cell 2]
# [ipython-only, skipped] !uv pip install numpy

# %% [cell 3]
# [ipython-only, skipped] !pip3 install numpy

# %% [cell 4]
# [ipython-only, skipped] !pip3 install numpy

# %% [cell 5]


# %% [cell 6]
import numpy

# %% [cell 7]
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
import time as wall

# %% [cell 8]
# Hardware
torch.set_num_threads(48)
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting up baseline parameters
BASELINE = dict(
    W=0.80, thetaP=1.00,
    nB=2, nM=2, nH=2,
    eta13=0.75, kappa13=0.55, lambdaP=1.60, lambda5=1.30, kappa5=0.50,
    epsP=1.00, rho5=1.10, rhoB=1.10, rho13=1.30, deltaP1=3.50,
    eps5=1.20, a5=0.15, etaR=2.50, kappaR=0.40, etaM=2.50, kappaM=0.50,
    eps13=1.00, a13=0.18, etaB13=0.95, kappaB13=0.50,
    etaM13=0.55, kappaM13=0.50,
    epsM=0.60, aM=0.18, etaBM=1.35, kappaBM=0.50,
    epsR=0.40, lambdaC=0.85,
    epsC=0.80, aC=0.08, etaRC=1.50, kappaRC=0.50,
    etaBC=1.50, kappaBC=0.50,
    mu0=0.35, AR=0.04, TR=24.0, phi=0.0,
    DR=0, q=0.30, tau1=40.0, tau2=80.0,
    alpha13=1.00, alpha5=1.00,
)

REGIMES = {
    "Normal":            dict(W=0.80, thetaP=1.00),
    "Early adenoma":     dict(W=1.00, thetaP=0.75),
    "Cancer-like":       dict(W=1.50, thetaP=0.50),
    "Strong APC-mutant": dict(W=2.00, thetaP=0.25),
}

Y0 = np.array([0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40])
VAR_NAMES  = ["b", "apc", "h5", "h13", "m", "r", "c"]
VAR_LABELS = [r"$\beta$-catenin", "APC", "HOXA5", "HOXA13",
              "MYC", "RA", "CYP26A1"]

# %% [cell 9]
# 7 ODE system  (numpy is for scipy reference solver)
# feel free to change the solve method.

def _hill(x, K, n=1):
    x = max(x, 0.0)
    return x**n / (K**n + x**n)


def _ra_input(tau, p):
    dietary   = p["AR"] * (1.0 + np.cos(2*np.pi * tau / p["TR"] - p["phi"]))
    treatment = 0.5 * p["DR"] * (
        np.tanh(p["q"]*(tau - p["tau1"])) - np.tanh(p["q"]*(tau - p["tau2"])))
    return p["mu0"] + dietary + treatment


def _ode_rhs(tau, y, p):
    b, apc, h5, h13, m, r, c = np.maximum(y, 0.0)
    dP  = 1.0 + p["deltaP1"] * (1.0 - p["thetaP"])
    muR = _ra_input(tau, p)

    db   = (p["W"] + p["eta13"]*_hill(h13, p["kappa13"], p["nH"])
            - b - p["lambdaP"]*apc*b
            - p["lambda5"]*h5*b / (p["kappa5"] + b))
    dapc = (1/p["epsP"]) * (
            (1 + p["rho5"]*h5) / (1 + p["rhoB"]*b + p["rho13"]*h13) - dP*apc)
    dh5  = (1/p["eps5"]) * (
            p["a5"] + p["etaR"]*_hill(r, p["kappaR"], 1)
            - h5 - p["etaM"]*m*h5 / (p["kappaM"] + m))
    dh13 = (1/p["eps13"]) * (
            p["a13"] + p["etaB13"]*_hill(b, p["kappaB13"], p["nB"])
            + p["etaM13"]*_hill(m, p["kappaM13"], p["nM"]) - h13)
    dm   = (1/p["epsM"]) * (
            p["aM"] + p["etaBM"]*_hill(b, p["kappaBM"], p["nB"]) - m)
    dr   = (1/p["epsR"]) * (muR - r - p["lambdaC"]*c*r)
    dc   = (1/p["epsC"]) * (
            p["aC"] + p["etaRC"]*_hill(r, p["kappaRC"], 1)
            + p["etaBC"]*_hill(b, p["kappaBC"], p["nB"]) - c)
    return [db, dapc, dh5, dh13, dm, dr, dc]


# %% [cell 10]
# Reference data  (scipy Radau basically)
def _solve_one(args):
    name, T, n_pts = args
    p = {**BASELINE, **REGIMES[name]} # pointers so we have edit access
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p),
                    (0, T), Y0,
                    t_eval=np.linspace(0, T, n_pts),
                    method="Radau", rtol=1e-10, atol=1e-12) # feel free to change the solver
    assert sol.success, f"{name} failed: {sol.message}"
    return name, sol.t, sol.y.T          # (n_pts, 7)


def generate_references(T=3000.0, n_pts=5000):
    refs = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for name, t, y in pool.map(
                _solve_one,
                [(n, T, n_pts) for n in REGIMES]):
            refs[name] = (t, y)
            print(f"  {name:<20s}  t=[0, {t[-1]:.0f}]  shape={y.shape}")
    return refs



# %% [cell 11]
# ODE residuals for torch
def _hill_t(x, K, n=1):
    return x**n / (K**n + x**n)


def _ra_t(t, p):
    dietary   = p["AR"] * (1.0 + torch.cos(2*np.pi * t / p["TR"] - p["phi"]))
    treatment = 0.5 * p["DR"] * (
        torch.tanh(p["q"]*(t - p["tau1"])) - torch.tanh(p["q"]*(t - p["tau2"])))
    return p["mu0"] + dietary + treatment


def physics_residual(t, z, dz, p):
    """ODE residual  dz/dt - f(t, z; p).   Shape: (N, 7)."""
    b, apc, h5, h13, m, r, c = (z[:, i:i+1] for i in range(7))
    dP  = 1.0 + p["deltaP1"] * (1.0 - p["thetaP"])
    muR = _ra_t(t, p)

    f0 = (p["W"] + p["eta13"]*_hill_t(h13, p["kappa13"], p["nH"])
          - b - p["lambdaP"]*apc*b
          - p["lambda5"]*h5*b / (p["kappa5"] + b))
    f1 = (1/p["epsP"]) * (
          (1 + p["rho5"]*h5) / (1 + p["rhoB"]*b + p["rho13"]*h13) - dP*apc)
    f2 = (1/p["eps5"]) * (
          p["a5"] + p["etaR"]*_hill_t(r, p["kappaR"], 1)
          - h5 - p["etaM"]*m*h5 / (p["kappaM"] + m))
    f3 = (1/p["eps13"]) * (
          p["a13"] + p["etaB13"]*_hill_t(b, p["kappaB13"], p["nB"])
          + p["etaM13"]*_hill_t(m, p["kappaM13"], p["nM"]) - h13)
    f4 = (1/p["epsM"]) * (
          p["aM"] + p["etaBM"]*_hill_t(b, p["kappaBM"], p["nB"]) - m)
    f5 = (1/p["epsR"]) * (muR - r - p["lambdaC"]*c*r)
    f6 = (1/p["epsC"]) * (
          p["aC"] + p["etaRC"]*_hill_t(r, p["kappaRC"], 1)
          + p["etaBC"]*_hill_t(b, p["kappaBC"], p["nB"]) - c)

    return dz - torch.cat([f0, f1, f2, f3, f4, f5, f6], dim=1)

# %% [cell 12]
# PINN network
class ForwardPINN(nn.Module):
    def __init__(self, T_max, n_vars=7, width=256, depth=4):
        super().__init__()
        self.T_max = T_max
        layers = [nn.Linear(1, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, n_vars))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t / self.T_max)


def time_derivatives(net, t):
    """Returns  z, dz/dt  with graph retained for backprop."""
    t = t.clone().requires_grad_(True)
    z = net(t)
    dz = torch.zeros_like(z)
    for i in range(z.shape[1]):
        dz[:, i:i+1] = torch.autograd.grad(
            z[:, i].sum(), t, create_graph=True)[0]
    return z, dz


# %% [cell 13]
# Training loop
def train_regime(name, t_ref, y_ref, *,
                 T=3000.0,
                 width=256, depth=4,
                 n_colloc=200_000,
                 n_data=3000,
                 adam_epochs=5000,
                 lbfgs_steps=500,
                 lr=1e-3,
                 lam_data=0.7,
                 lam_phys=0.3,
                 seed=42,
                 log_every=250):
    """Train one forward PINN for a single biological regime."""
    torch.manual_seed(seed)
    p = {**BASELINE, **REGIMES[name]}

    # reference data (subsample for data loss) 
    idx = np.linspace(0, len(t_ref)-1, n_data, dtype=int)
    t_d = torch.tensor(t_ref[idx, None],  device=DEVICE)   # (n_data, 1)
    y_d = torch.tensor(y_ref[idx],        device=DEVICE)   # (n_data, 7)

    # network 
    net = ForwardPINN(T_max=T, width=width, depth=depth).to(DEVICE)
    n_params = sum(q.numel() for q in net.parameters())
    print(f"  arch {depth}×{width}  params {n_params:,}")

    hist = dict(epoch=[], loss=[], Ld=[], Lp=[])

    # Do 1: Adam + cosine LR 
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=adam_epochs, eta_min=1e-6)

    t0 = wall.perf_counter()
    for ep in range(1, adam_epochs + 1):
        opt.zero_grad()

        # data loss
        z_d = net(t_d)
        Ld  = ((z_d - y_d)**2).mean()

        # physics loss (bcs we do random collocation each step)
        tc  = torch.rand(n_colloc, 1, device=DEVICE) * T
        zc, dzc = time_derivatives(net, tc)
        res = physics_residual(tc, zc, dzc, p)
        Lp  = (res**2).mean()

        loss = lam_data * Ld + lam_phys * Lp
        loss.backward()
        opt.step()
        sched.step()

        if ep % log_every == 0 or ep == 1:
            hist["epoch"].append(ep)
            hist["loss"].append(loss.item())
            hist["Ld"].append(Ld.item())
            hist["Lp"].append(Lp.item())
            dt = wall.perf_counter() - t0
            print(f"    Adam {ep:>5}/{adam_epochs}  "
                  f"L={loss.item():.2e}  Ld={Ld.item():.2e}  "
                  f"Lp={Lp.item():.2e}  lr={sched.get_last_lr()[0]:.1e}  "
                  f"[{dt:.0f}s]")

    # Do 2: L-BFGS polishrt
    if lbfgs_steps > 0:
        print(f"    L-BFGS ({lbfgs_steps} steps, {n_colloc//5} colloc) ...")
        n_col_lb = n_colloc // 5          # fewer points for speed, but feel free to experiemtn with this nate.
        lbfgs = torch.optim.LBFGS(
            net.parameters(), lr=0.5, max_iter=20,
            history_size=50, line_search_fn="strong_wolfe")

        for step in range(1, lbfgs_steps + 1):
            def closure():
                lbfgs.zero_grad()
                Ld_ = ((net(t_d) - y_d)**2).mean()
                tc_  = torch.rand(n_col_lb, 1, device=DEVICE) * T
                zc_, dzc_ = time_derivatives(net, tc_)
                Lp_ = (physics_residual(tc_, zc_, dzc_, p)**2).mean()
                tot = lam_data*Ld_ + lam_phys*Lp_
                tot.backward()
                return tot
            loss = lbfgs.step(closure)
            if step % 100 == 0 or step == lbfgs_steps:
                dt = wall.perf_counter() - t0
                print(f"    LBFGS {step:>4}/{lbfgs_steps}  "
                      f"L={loss:.2e}  [{dt:.0f}s]")

    # Evaluate on a fine grid 
    with torch.no_grad():
        t_eval = torch.linspace(0, T, 6000, device=DEVICE).reshape(-1, 1)
        z_eval = net(t_eval).cpu().numpy()

    sol = {"t": t_eval.cpu().numpy().ravel()}
    for i, vn in enumerate(VAR_NAMES):
        sol[vn] = z_eval[:, i]

    dt_total = wall.perf_counter() - t0
    print(f"  done  ({dt_total:.0f}s total)")
    return sol, net, hist


# %% [cell 14]
# stemness
def stemness(sol, p):
    return (sol["b"] * (1 + p["alpha13"]*sol["h13"])
            / ((1 + sol["apc"]) * (1 + p["alpha5"]*sol["h5"])))


# %% [cell 15]
# plotting 

def plot_all(solutions, refs):
    """PINN (solid line) vs scipy reference (dashed) for each var basically"""
    for i, (key, label) in enumerate(zip(VAR_NAMES, VAR_LABELS)):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name in REGIMES:
            sol   = solutions[name]
            tr, yr = refs[name]
            ax.plot(sol["t"],  sol[key], lw=2.0, label=f"{name} (PINN)")
            ax.plot(tr, yr[:, i], "--", lw=1.0, alpha=0.6,
                    label=f"{name} (ref)")
        ax.axvspan(BASELINE["tau1"], BASELINE["tau2"],
                   alpha=0.08, color="orange", label="ATRA")
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(label)
        ax.set_title(label); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(f"pinn7_{key}.png", dpi=150)
        plt.show(); plt.close(fig)

    # stemness calc
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in REGIMES:
        sol = solutions[name]
        pc  = {**BASELINE, **REGIMES[name]}
        ax.plot(sol["t"], stemness(sol, pc), lw=2.0, label=name)
    ax.axvspan(BASELINE["tau1"], BASELINE["tau2"],
               alpha=0.08, color="orange", label="ATRA")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("Stemness")
    ax.set_title("Stemness index"); ax.legend()
    fig.tight_layout(); fig.savefig("pinn7_stemness.png", dpi=150)
    plt.show(); plt.close(fig)


def plot_losses(all_hist):
    n = len(all_hist)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    for ax, (name, h) in zip(axes[0], all_hist.items()):
        ax.semilogy(h["epoch"], h["loss"], "k-",  label="total")
        ax.semilogy(h["epoch"], h["Ld"],   "b--", label="data")
        ax.semilogy(h["epoch"], h["Lp"],   "r--", label="physics")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(name); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("pinn7_losses.png", dpi=150)
    plt.show(); plt.close(fig)


def print_summary(solutions):
    print("\n" + "="*70)
    print("Final values & stemness")
    print("="*70)
    for name in REGIMES:
        sol = solutions[name]
        pc  = {**BASELINE, **REGIMES[name]}
        S   = stemness(sol, pc)
        t   = sol["t"]
        mask = (t >= pc["tau1"]) & (t <= pc["tau2"])
        print(f"\n{name}")
        for vn, vl in zip(VAR_NAMES, VAR_LABELS):
            print(f"  {vl:<16s} {sol[vn][-1]:.4f}")
        print(f"  {'Stemness':<16s} {S[-1]:.4f}")
        if mask.any():
            print(f"  Min S (ATRA)    {S[mask].min():.4f}")
            print(f"  Mean S (ATRA)   {S[mask].mean():.4f}")



# %% [cell 16]
# Main
def main(T=3000.0):
    print("="*60)
    print("Forward PINN - 7-ODE Reduced WNT-RA-HOX Model")
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name()}")
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {gb:.1f} GB")
    print(f"Threads: {torch.get_num_threads()}")
    print("="*60)

    # 1. Reference solutions; IDK about Radau. Might change.
    print("\n[1/3] Reference solutions (scipy Radau) ...")
    refs = generate_references(T=T, n_pts=5000)

    # 2. Train
    print("\n[2/3] Training PINNs ...")
    solutions, all_hist = {}, {}
    for name in REGIMES:
        print(f"\n── {name} ──")
        tr, yr = refs[name]
        sol, net, hist = train_regime(
            name, tr, yr,
            T=T,
            width=256,
            depth=4,
            n_colloc=200_000,
            n_data=3000,
            adam_epochs=5000,
            lbfgs_steps=500,
            lr=1e-3,
            lam_data=0.7,
            lam_phys=0.3,
        )
        solutions[name] = sol
        all_hist[name]  = hist

    # 3. Plotting
    print("\n[3/3] Plotting ...")
    plot_all(solutions, refs)
    plot_losses(all_hist)
    print_summary(solutions)

    return solutions, refs, all_hist


if __name__ == "__main__":
    solutions, refs, histories = main()


# %% [cell 17]
