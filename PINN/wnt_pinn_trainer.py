"""
PINN trainer for the stiff 27-state Wnt/Retinoid/HOX ODE.
Uses adaptive per-equation weighting, time-marching windows pinned at Wnt on/off, and L-BFGS polish.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- PHYSICS IMPORT (verified faithful module from wnt_model_physics.py) -----
USING_STUB = False
from wnt_model_physics import (build_params, initial_state, residuals,
                               forcing_W, forcing_gtild, K16_dyn, bctf)
# Fallback stub (interface-compatible, NOT faithful) — only if you must run
# without the real physics module present:
# USING_STUB = True
# from _physics_stub import (build_params, initial_state, residuals,
#                            forcing_W, forcing_gtild, K16_dyn, bctf)

# ---- state indices (MATLAB U(k) -> z[:,k-1]) --------------------------------
iRo, iRa, iA, iR, iB, iBr, iN, iNr, iC, iCr, iDc, iDn, iBc = range(0, 13)
iV, iDi, iDb, iBp, iDa, iP, iBa, iX = range(13, 21)
iH5, iM, iMi, iCa, iH13, iCi = range(21, 27)
N_STATE = 27
# residual layout: res_expl has the 23 explicit eqs (all but Da,P,Ba,X),
# res_impl has the 4 implicit WNT mass-matrix eqs.
EXPL_IDX = [i for i in range(N_STATE) if i not in (iDa, iP, iBa, iX)]
N_RES = 27  # 23 explicit + 4 implicit


# =============================================================================
# Network
# =============================================================================
class PINN(nn.Module):
    """MLP with tanh, hard IC constraint out=y0+net(s)*s*scale, and learnable per-state log-scale."""

    def __init__(self, n, scale, t_win, P, nodes=128, layers=5):
        super().__init__()
        self.t_win = float(t_win)
        self.register_buffer("scale", torch.as_tensor(scale).reshape(1, -1))
        # per-state learnable log-scale (multiplicative, init 1.0)
        self.log_s = nn.Parameter(torch.zeros(1, n))
        Bf = (np.pi / 6) * P['a']
        self.register_buffer("freqs", torch.tensor([Bf, 2 * Bf]))
        n_feat = 1 + 2 * 2  # s + (sin,cos) for each of 2 freqs
        seq = [nn.Linear(n_feat, nodes), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(nodes, nodes), nn.Tanh()]
        seq += [nn.Linear(nodes, n)]
        self.seq = nn.Sequential(*seq)
        for m in self.seq:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def _feat(self, t, t0):
        s = (t - t0) / self.t_win
        ft = t * self.freqs.reshape(1, -1)
        return torch.cat([s, torch.sin(ft), torch.cos(ft)], 1)

    def forward(self, t, t0, y0):
        s = (t - t0) / self.t_win
        eff = self.scale * torch.exp(self.log_s)
        return y0 + self.seq(self._feat(t, t0)) * s * eff


def make_scale(y0_np):
    """Per-state scale: max(|y0|, pathway-specific floor)."""
    scale = np.abs(y0_np).astype(float)
    floor = np.full(N_STATE, 1e-3)
    floor[0:13] = 1e-2     # RA block (O(1e-2..1e-1))
    floor[13:21] = 1e-3    # WNT block (several states ~1e-3..1e-2 non-dim)
    floor[21:27] = 1e-2    # HOX block (O(1e-2..1))
    return np.maximum(scale, floor)


# =============================================================================
# Autodiff time derivative (vectorized over states, single graph)
# =============================================================================
def time_deriv(net, t, t0, y0):
    """Return (z, dz/dt) for all 27 states via batched autodiff."""
    t = t.clone().requires_grad_(True)
    z = net(t, t0, y0)
    n = z.shape[1]
    try:
        # batched: one row per output component, each a unit seed
        eye = torch.eye(n, dtype=z.dtype, device=z.device)
        seeds = eye.unsqueeze(1).expand(n, z.shape[0], n)  # [n, B, n]
        grads = torch.autograd.grad(z, t, grad_outputs=seeds,
                                    create_graph=True, is_grads_batched=True)[0]
        dz = grads.squeeze(-1).transpose(0, 1)  # [B, n]
    except (RuntimeError, TypeError):
        cols = []
        for i in range(n):
            gi = torch.autograd.grad(z[:, i].sum(), t, create_graph=True)[0]
            cols.append(gi)
        dz = torch.cat(cols, 1)
    return z, dz


# =============================================================================
# Adaptive per-equation residual weighting
# =============================================================================
class AdaptiveWeights:
    def __init__(self, n_res, beta=0.9, eps=1e-12, device=DEVICE):
        self.ema = torch.ones(n_res, dtype=torch.get_default_dtype(), device=device)
        self.beta = beta
        self.eps = eps
        self.frozen = False

    def update(self, per_eq_ms):
        if self.frozen:
            return
        with torch.no_grad():
            self.ema.mul_(self.beta).add_((1 - self.beta) * per_eq_ms.detach())

    def weights(self):
        return 1.0 / (self.ema + self.eps)


def _residual_loss(net, ts, t0c, y0, P, aw, update=True):
    z, dz = time_deriv(net, ts, t0c, y0)
    re, ri = residuals(ts, z, dz, P)            # [B,23], [B,4]
    res = torch.cat([re, ri], 1)                # [B,27]
    per_eq_ms = (res ** 2).mean(dim=0)          # [27]
    if update:
        aw.update(per_eq_ms)
    w = aw.weights()
    loss = (w * per_eq_ms).mean()
    return loss, per_eq_ms.detach()


# =============================================================================
# Per-window training: Adam warm-up + L-BFGS polish
# =============================================================================
def train_window(net, t0, t1, y0, P, iters=2000, lr=2e-3, n_col=2000,
                 lbfgs_iters=50, aw=None, seed=0):
    torch.manual_seed(seed)
    if aw is None:
        aw = AdaptiveWeights(N_RES)
    t0c = torch.full((n_col, 1), t0, device=DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1),
                                                       eta_min=lr * 0.05)
    last = float("nan")
    for it in range(iters):
        ts = torch.rand(n_col, 1, device=DEVICE) * (t1 - t0) + t0
        opt.zero_grad()
        loss, _ = _residual_loss(net, ts, t0c, y0, P, aw, update=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
        last = loss.item()
    # ---- L-BFGS polish (weights frozen for a stable closure) ----
    if lbfgs_iters and lbfgs_iters > 0:
        aw.frozen = True
        ts = torch.rand(n_col, 1, device=DEVICE) * (t1 - t0) + t0
        lb = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_iters,
                               line_search_fn="strong_wolfe",
                               tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            lb.zero_grad()
            l, _ = _residual_loss(net, ts, t0c, y0, P, aw, update=False)
            l.backward()
            return l
        try:
            lb.step(closure)
            with torch.no_grad():
                lp, _ = _residual_loss(net, ts, t0c, y0, P, aw, update=False)
                last = lp.item()
        except RuntimeError:
            pass  # LBFGS can fail on stiff windows; keep the Adam result
        aw.frozen = False
    return last, aw


# =============================================================================
# Time-marching solver
# =============================================================================
def solve_pinn(model, gamma1=1.0, T=None, win_len=1000.0, iters=2000,
               lr=2e-3, n_col=2000, n_out=300, nodes=128, layers=5,
               lbfgs_iters=50, max_windows=None, seed=0, verbose=True):
    """Time-march the PINN window by window and return dict of t and output vars."""
    P = build_params(model, gamma1=gamma1)
    if T is None:
        T = P['T']
    y0_np = initial_state(P)
    edges = sorted(set(list(np.arange(0., T + win_len, win_len)) + [3000., 25000.]))
    edges = np.array([e for e in edges if e <= T + 1e-9])
    if edges[-1] < T - 1e-9:
        edges = np.append(edges, T)
    n_win = len(edges) - 1
    if max_windows is not None:
        n_win = min(n_win, max_windows)

    y0 = torch.tensor(y0_np, device=DEVICE).reshape(1, -1)
    aw = AdaptiveWeights(N_RES)  # carried across windows so weights warm-start
    AW_RESET_TIMES = {3000.0, 25000.0}
    out_t, out_z, win_losses = [], [], []
    for wd in range(n_win):
        t0, t1 = float(edges[wd]), float(edges[wd + 1])
        if t0 in AW_RESET_TIMES:
            aw = AdaptiveWeights(N_RES)
        scale = make_scale(y0.cpu().numpy().ravel())
        net = PINN(N_STATE, scale, t1 - t0, P, nodes, layers).to(DEVICE)
        loss, aw = train_window(net, t0, t1, y0, P, iters=iters, lr=lr,
                                n_col=n_col, lbfgs_iters=lbfgs_iters, aw=aw,
                                seed=seed)
        win_losses.append(loss)
        with torch.no_grad():
            tg = torch.linspace(t0, t1, n_out, device=DEVICE).reshape(-1, 1)
            t0g = torch.full_like(tg, t0)
            zg = net(tg, t0g, y0)
            out_t.append(tg.cpu().numpy().ravel())
            out_z.append(zg.cpu().numpy())
            y0 = net(torch.full((1, 1), t1, device=DEVICE),
                     torch.full((1, 1), t0, device=DEVICE), y0).detach()
        if verbose:
            print(f"  [{model}] win {wd + 1:>3}/{n_win}  t<= {t1:>7.0f}  "
                  f"loss={loss:.3e}")

    t = np.concatenate(out_t)
    Z = np.concatenate(out_z)
    Pp = build_params(model, gamma1=gamma1)
    Ba = Z[:, iBa]; Ci = Z[:, iCi]
    BcatTcf = bctf(torch.tensor(Ba), torch.tensor(Ci), Pp).numpy()
    return dict(t=t, win_losses=win_losses, edges=edges[:n_win + 1],
                H5=Z[:, iH5], H13=Z[:, iH13], M=Z[:, iM], Mi=Z[:, iMi],
                Ca=Z[:, iCa], Ci=Ci, Ba=Ba, P=Z[:, iP], Bp=Z[:, iBp],
                BcatTcf=BcatTcf, V=Z[:, iV], Di=Z[:, iDi], Db=Z[:, iDb],
                Da=Z[:, iDa], X=Z[:, iX], Nr=Z[:, iNr], R=Z[:, iR])


# =============================================================================
# Plotting -- reproduces the figures in normasysl_call.m
# =============================================================================
C_BLACK = (0.1, 0.1, 0.1)
C_RED = (0.6, 0.0, 0.0)
SHADE = (0.95, 0.95, 0.95)
SHADE_START, SHADE_END = 3000.0, 25000.0


def _shade(ax):
    yl = ax.get_ylim()
    ax.fill_between([SHADE_START, SHADE_END], yl[0], yl[1], color=SHADE, zorder=0)
    ax.set_ylim(yl)


def _cmp_panel(ax, const, dyn, ref_const, ref_dyn, key, title):
    if const is not None:
        ax.plot(const['t'], const[key], color=C_BLACK, lw=2.2,
                label=r'$K_{16}$ constant')
    if dyn is not None:
        ax.plot(dyn['t'], dyn[key], color=C_RED, lw=2.2,
                label=r'$K_{16}$ dynamic')
    if ref_const is not None and key in ref_const:
        ax.plot(ref_const['t'], ref_const[key], color=C_BLACK, lw=1.0, ls='--')
    if ref_dyn is not None and key in ref_dyn:
        ax.plot(ref_dyn['t'], ref_dyn[key], color=C_RED, lw=1.0, ls='--')
    ax.set_title(title)
    ax.set_xlabel(r'time ($\tau$)')
    ax.set_ylabel('(non-dim)')
    ax.set_box_aspect(0.7)


def plot_all(const_sol, dyn_sol, ref_const=None, ref_dyn=None, show=True):
    """Plot HOX and WNT comparison figures (const=black, dyn=red), with optional scipy overlay."""
    import matplotlib.pyplot as plt
    figs = []

    # --- Fig 1: HOX comparison 2x2 (HOXA5, HOXA13, MYC, MIZ1) ---
    f1, ax = plt.subplots(2, 2, figsize=(11, 9))
    for a, (k, ttl) in zip(ax.ravel(),
                           [('H5', 'HOXA5'), ('H13', 'HOXA13'),
                            ('M', 'MYC'), ('Mi', 'MIZ1')]):
        _cmp_panel(a, const_sol, dyn_sol, ref_const, ref_dyn, k, ttl)
    ax[1, 1].legend(fontsize=10, loc='best')
    f1.suptitle('Comparison of constant vs dynamic $K_{16}$ on HOX pathway',
                fontsize=15)
    f1.tight_layout()
    figs.append(f1)

    # --- Fig 2: WNT comparison 2x2 (APC, beta-cat, beta-cat:TCF, P-beta-cat) ---
    f2, ax = plt.subplots(2, 2, figsize=(11, 9))
    for a, (k, ttl) in zip(ax.ravel(),
                           [('P', 'APC'), ('Ba', r'$\beta$-catenin'),
                            ('BcatTcf', r'$\beta$-cat:TCF'),
                            ('Bp', r'Phospho-$\beta$-cat')]):
        _cmp_panel(a, const_sol, dyn_sol, ref_const, ref_dyn, k, ttl)
    ax[1, 1].legend(fontsize=10, loc='best')
    f2.suptitle('Comparison of constant vs dynamic $K_{16}$ on WNT pathway',
                fontsize=15)
    f2.tight_layout()
    figs.append(f2)

    # --- Fig 3: HOX components 2x3 with shaded Wnt-on region ---
    f3, ax = plt.subplots(2, 3, figsize=(15, 9))
    hox6 = [('H5', r'HOXA5 ($H_a$)'), ('H13', r'HOXA13 ($H_i$)'),
            ('M', r'MYC ($M$)'), ('Mi', r'MIZ1 ($M_i$)'),
            ('Ca', r'MYC:MIZ1 ($C_a$)'), ('Ci', r'HOXA13:$\beta$-cat ($C_i$)')]
    sol = const_sol if const_sol is not None else dyn_sol
    rref = ref_const if const_sol is not None else ref_dyn
    for a, (k, ttl) in zip(ax.ravel(), hox6):
        a.plot(sol['t'], sol[k], color=C_BLACK, lw=2.2)
        if rref is not None and k in rref:
            a.plot(rref['t'], rref[k], color=C_BLACK, lw=1.0, ls='--')
        a.set_title(ttl); a.set_xlabel(r'time ($\tau$)')
        a.set_ylabel('(non-dim)'); _shade(a)
    f3.suptitle('HOX components (Wnt-on region shaded)', fontsize=15)
    f3.tight_layout()
    figs.append(f3)

    # --- Fig 4: WNT dynamics 2x2 with shaded Wnt-on region ---
    f4, ax = plt.subplots(2, 2, figsize=(11, 9))
    wnt4 = [('P', r'APC ($P$)'), ('Ba', r'$\beta$-catenin ($B_a$)'),
            ('BcatTcf', r'$\beta$-cat:TCF ($B_t$)'),
            ('Bp', r'Phospho-$\beta$-cat ($B_p$)')]
    sol4 = dyn_sol if dyn_sol is not None else const_sol
    rref4 = ref_dyn if dyn_sol is not None else ref_const
    for a, (k, ttl) in zip(ax.ravel(), wnt4):
        a.plot(sol4['t'], sol4[k], color=C_BLACK, lw=2.2)
        if rref4 is not None and k in rref4:
            a.plot(rref4['t'], rref4[k], color=C_BLACK, lw=1.0, ls='--')
        a.set_title(ttl); a.set_xlabel(r'time ($\tau$)')
        a.set_ylabel('(non-dim)'); _shade(a)
    f4.suptitle('WNT pathway dynamics (Wnt-on region shaded)', fontsize=15)
    f4.tight_layout()
    figs.append(f4)

    if show:
        plt.show()
    return figs


# =============================================================================
# Smoke test
# =============================================================================
def smoke_test():
    """Quick sanity check: train 3 windows and print per-window losses."""
    print("=" * 64)
    print(f"SMOKE TEST  (physics = {'STUB' if USING_STUB else 'REAL'}, "
          f"device = {DEVICE})")
    print("=" * 64)
    sol = solve_pinn("const", win_len=1000.0, iters=300, lr=2e-3,
                     n_col=1024, n_out=80, lbfgs_iters=20, max_windows=3,
                     seed=0, verbose=True)
    losses = sol['win_losses']
    print("-" * 64)
    print("per-window adaptive-weighted losses:", [f"{l:.3e}" for l in losses])
    print("=" * 64)
    return sol


if __name__ == "__main__":
    smoke_test()
