"""Fisher Information Matrix (FIM) practical-identifiability analysis for the
REDUCED 8-parameter WNT-RA-HOX inverse problem.

This is the sister of ../PINN-fisher-matrix (all 36 params). Here we restrict
the unknowns to the 8 MOST-IDENTIFIABLE parameters found by the full-36 FIM
(lowest flat-subspace participation, worst-case across all 4 regimes):

    SUBSET = [lambdaC, etaBM, kappaBM, kappaBC, etaBC, etaRC, kappaB13, kappaR]

i.e. the WNT->MYC / WNT->CYP26 / RA->CYP26 and Hill-scale terms that the excite
conditions drive directly. All 28 other parameters are held FIXED at their true
regime values, so this asks: once we only try to recover the well-excited
handful, does the FIM confirm a clean, well-conditioned, near-diagonal problem
(the opposite of the sloppy 36-param case)?

Follows the local-sensitivity / FIM route of Yazdani, Lu, Raissi & Karniadakis,
"Systems biology informed deep learning for inferring parameters and hidden
dynamics" (PLoS Comput Biol 2020, §S1 + Figs 6/S3/S6). The recipe there:

    y = h(x) + eps,   eps ~ N(0, sigma^2)                          (Eq 1c)
    S[obs, j]        = d y_obs / d p_j        (local sensitivity)
    FIM              = (1/sigma^2) S^T S                            (Eq S2)
    Cov              = FIM^{-1}  (pseudo-inverse when singular)     (Eq S3)
    sigma_j          = sqrt(Cov_jj)           (param std => practical ident.)
    R_jk             = Cov_jk / sqrt(Cov_jj Cov_kk)  (correlation matrix)
    null eigenvectors of FIM  => the non-identifiable directions   (Fig 6)

We work in LOG-parameters (S_log[:,j] = p_j * dy/dp_j) so every one of the 36
parameters, whose true values span 0.04..3.5, contributes on a common
dimensionless footing; the diagonal of Cov_log is then each parameter's squared
coefficient of variation (relative std). thetaP, being in (0,1), is left in
natural units for its column (log is fine too since values are O(1)).

Observation model matches recover_odefit.py: all 7 states are observed under all
10 excitation conditions, sigma = 0.002. The FIM is assembled at the TRUE
parameter values of each regime (the local-sensitivity FIM the paper uses for
its identifiability verdict).

Run:  python3 fisher_matrix.py <out_dir> [regime|all]
      regime in {Normal, Early adenoma, Cancer-like, Strong APC-mutant, all}
      default: all
"""
import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.integrate import solve_ivp

# --- single source of truth: pull the model from the excite folder ----------
HERE = os.path.dirname(os.path.abspath(__file__))
EXCITE = os.path.join(os.path.dirname(HERE), "PINN-inverse-multicond-excite")
sys.path.insert(0, EXCITE)
from config import BASELINE, REGIMES, CONDITIONS, Y0, VAR_NAMES           # noqa
from odes import _ode_rhs                                                 # noqa

# restrict to the 8 most-identifiable unknowns (see module docstring); the other
# 28 parameters stay at their true regime values.
UNKNOWN = ["lambdaC", "etaBM", "kappaBM", "kappaBC",
           "etaBC", "etaRC", "kappaB13", "kappaR"]

# ---- observation / sensitivity recipe --------------------------------------
T       = 150.0     # horizon (active-dynamics window)
N_TIME  = 200       # uniform observation grid per condition (FIM quadrature)
SIGMA   = 0.002     # measurement noise std (matches recover_odefit / profiling)
REL_STEP = 1e-3     # central-difference relative perturbation for sensitivities
K = len(UNKNOWN)


def solve(p, t_eval):
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p), (0.0, T), Y0,
                    t_eval=t_eval, method="LSODA", rtol=1e-9, atol=1e-11)
    if not sol.success:
        return None
    return sol.y.T                       # (n_time, 7)


def sensitivity_matrix(true_p):
    """Central-difference LOG-sensitivities stacked over all conditions.

    Returns S_log with shape (n_obs, K): row = one (condition, time, state)
    observation, column = d(observable)/d(ln p_j) = p_j * d(observable)/dp_j.
    """
    t_eval = np.linspace(0.0, T, N_TIME)
    cols = []
    for j, name in enumerate(UNKNOWN):
        p0 = float(true_p[name])
        h = REL_STEP * (abs(p0) if p0 != 0.0 else 1.0)
        p_plus = {**true_p, name: p0 + h}
        p_minus = {**true_p, name: p0 - h}
        blocks_p, blocks_m = [], []
        for c in CONDITIONS:
            yp = solve({**p_plus,  **c["forcing"]}, t_eval)
            ym = solve({**p_minus, **c["forcing"]}, t_eval)
            if yp is None or ym is None:
                raise RuntimeError(f"solve failed perturbing {name}")
            blocks_p.append(yp.ravel())
            blocks_m.append(ym.ravel())
        yp = np.concatenate(blocks_p)
        ym = np.concatenate(blocks_m)
        # d y / d ln p = p * (y+ - y-) / (2h)  = (y+ - y-) / (2 * REL_STEP)
        cols.append(p0 * (yp - ym) / (2.0 * h))
    return np.column_stack(cols)         # (n_obs, K)


def analyse(regime, out_dir):
    safe = regime.replace(" ", "_").replace("/", "_")
    true_p = {**BASELINE, **REGIMES[regime]}
    true_vals = np.array([float(true_p[k]) for k in UNKNOWN])

    t0 = time.time()
    S = sensitivity_matrix(true_p)                       # (n_obs, K), log-sens
    FIM = (S.T @ S) / (SIGMA ** 2)                       # (K, K)
    dt = time.time() - t0
    print(f"[{regime}] FIM built in {dt:.1f}s  "
          f"(n_obs={S.shape[0]}, K={K})", flush=True)

    # eigen-decomposition (symmetric PSD, ascending)
    evals, evecs = np.linalg.eigh(FIM)
    evals = np.clip(evals, 0, None)
    cond_number = evals[-1] / evals[evals > 0].min() if (evals > 0).any() else np.inf

    # ---- identifiability via the null / near-null subspace (paper Fig 6) ----
    # Variance of the estimate along an eigen-direction is 1/lambda (CRLB).  A
    # direction with lambda < NULL_TOL has relative std > 1 (>100% uncertainty)
    # => practically non-identifiable.  We do NOT trust the pinv diagonal here:
    # for a 1e17-conditioned FIM, pinv silently DROPS the sloppy directions and
    # would falsely report every parameter as well-determined.  Instead we
    # measure each parameter's PARTICIPATION in the flat subspace,
    #   part_j = sqrt( sum_{lambda_k < NULL_TOL}  V[j,k]^2 )  in [0, 1],
    # i.e. how much of the parameter's own axis lies in the non-identifiable
    # subspace.  part_j ~ 1 => the parameter is (almost) purely a flat direction.
    NULL_TOL = 1.0            # CRLB: lambda < 1 <=> relative std > 100%
    SOFT_TOL = evals[-1] * 1e-4   # merely "sloppy" (weakly constrained) band
    null_mask = evals < NULL_TOL
    soft_mask = (evals < SOFT_TOL) & ~null_mask
    part_null = np.sqrt((evecs[:, null_mask] ** 2).sum(1))
    part_soft = np.sqrt((evecs[:, soft_mask] ** 2).sum(1))

    verdict = np.where(part_null > 0.5, "NON-IDENT",
              np.where((part_null > 0.2) | (part_soft > 0.5), "WEAK", "IDENT"))

    # correlation matrix from a floored inverse (bounded, scale-free): floor the
    # eigenvalues so the near-null directions contribute a large-but-finite
    # covariance instead of being dropped.  |R_jk| -> 1 flags a non-identifiable
    # PAIR (the classic S3/S6 read of the paper).
    lam_floor = np.maximum(evals, evals[-1] * 1e-12)
    cov = (evecs / lam_floor) @ evecs.T
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    R = np.clip(cov / np.outer(d, d), -1.0, 1.0)
    np.fill_diagonal(R, 1.0)

    # correlated pairs (|R|>0.99) — the practically non-identifiable combinations
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            if abs(R[i, j]) > 0.99:
                pairs.append((UNKNOWN[i], UNKNOWN[j], float(R[i, j])))
    pairs.sort(key=lambda t: -abs(t[2]))

    # ---- save numerics ------------------------------------------------------
    np.save(os.path.join(out_dir, f"{safe}_FIM.npy"), FIM)
    np.save(os.path.join(out_dir, f"{safe}_corr.npy"), R)
    order = np.argsort(part_null + 0.01 * part_soft)     # best-constrained first
    csv = ["param,true_value,null_participation,soft_participation,verdict"]
    for i in order:
        csv.append(f"{UNKNOWN[i]},{true_vals[i]:.6g},{part_null[i]:.4f},"
                   f"{part_soft[i]:.4f},{verdict[i]}")
    with open(os.path.join(out_dir, f"{safe}_fim_ident.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")
    with open(os.path.join(out_dir, f"{safe}_correlated_pairs.csv"), "w") as fh:
        fh.write("param_i,param_j,R\n")
        for a, b, r in pairs:
            fh.write(f"{a},{b},{r:.4f}\n")

    n_id = int((verdict == "IDENT").sum())
    n_wk = int((verdict == "WEAK").sum())
    n_no = int((verdict == "NON-IDENT").sum())
    summary = dict(regime=regime, n_obs=int(S.shape[0]), K=K, sigma=SIGMA,
                   cond_number=float(cond_number),
                   eig_max=float(evals[-1]), eig_min=float(evals[0]),
                   n_null_directions=int(null_mask.sum()),
                   n_ident=n_id, n_weak=n_wk, n_nonident=n_no,
                   nonident_params=[UNKNOWN[i] for i in range(K)
                                    if verdict[i] == "NON-IDENT"],
                   n_correlated_pairs=len(pairs),
                   top_correlated_pairs=[[a, b, round(r, 3)]
                                         for a, b, r in pairs[:10]])
    with open(os.path.join(out_dir, f"{safe}_fim_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[{regime}] cond(FIM)={cond_number:.2e}  null-dirs={null_mask.sum()}  "
          f"IDENT={n_id} WEAK={n_wk} NON-IDENT={n_no}  "
          f"|R|>0.99 pairs={len(pairs)}", flush=True)
    if n_no:
        print(f"    NON-IDENT: {[UNKNOWN[i] for i in range(K) if verdict[i]=='NON-IDENT']}",
              flush=True)

    # ---- figures ------------------------------------------------------------
    plot_regime(regime, safe, FIM, R, evals, part_null, part_soft, verdict, out_dir)
    plot_soft_eigvecs(regime, safe, evals, evecs, out_dir)
    return dict(regime=regime, evals=evals, part_null=part_null,
                verdict=verdict, cond_number=cond_number)


def plot_regime(regime, safe, FIM, R, evals, part_null, part_soft, verdict, out_dir):
    labels = UNKNOWN
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # (a) |FIM| heatmap, log10 magnitude
    axa = fig.add_subplot(gs[0, 0])
    with np.errstate(divide="ignore"):
        mag = np.log10(np.abs(FIM) + 1e-30)
    finite = mag[np.isfinite(mag) & (np.abs(FIM) > 0)]
    vmin = np.percentile(finite, 5) if finite.size else 0
    im = axa.imshow(mag, cmap="viridis", vmin=vmin, vmax=mag.max())
    axa.set_title("(a) Fisher Information Matrix  $\\log_{10}|F_{ij}|$",
                  fontsize=12, fontweight="bold")
    _tick(axa, labels)
    fig.colorbar(im, ax=axa, fraction=0.046, pad=0.04)

    # (b) correlation matrix from FIM^{-1}
    axb = fig.add_subplot(gs[0, 1])
    im = axb.imshow(R, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -1.0, 1.0))
    axb.set_title("(b) Parameter correlation matrix  $R_{ij}$  "
                  "($|R|\\!\\to\\!1$ = non-identifiable pair)",
                  fontsize=12, fontweight="bold")
    _tick(axb, labels)
    fig.colorbar(im, ax=axb, fraction=0.046, pad=0.04)

    # (c) eigenvalue spectrum (sloppiness)
    axc = fig.add_subplot(gs[1, 0])
    ev = np.clip(evals[::-1], 1e-30, None)               # descending
    axc.semilogy(range(1, len(ev) + 1), ev, "o-", color="#1f5fa8", ms=5)
    axc.axhline(ev[0] * 1e-6, ls="--", color="grey", lw=1)
    axc.text(len(ev), ev[0] * 1e-6, "  cond$^{-1}\\!=10^{6}$ floor",
             va="center", fontsize=9, color="grey")
    axc.set_xlabel("eigenvalue index (largest → smallest)")
    axc.set_ylabel("FIM eigenvalue")
    axc.set_title("(c) FIM eigenvalue spectrum — sloppy directions decay",
                  fontsize=12, fontweight="bold")
    axc.grid(alpha=0.3, which="both")

    # (d) participation of each parameter in the non-identifiable subspace
    axd = fig.add_subplot(gs[1, 1])
    score = part_null + 0.01 * part_soft
    order = np.argsort(-score)                           # worst-constrained first
    cmap = {"IDENT": "#2c7d3f", "WEAK": "#e08a1e", "NON-IDENT": "#c0392b"}
    cols = [cmap[v] for v in verdict[order]]
    y = np.arange(K)
    axd.barh(y, part_null[order], color=cols, label="null ($\\lambda<1$)")
    axd.barh(y, part_soft[order], left=part_null[order], color=cols, alpha=0.35,
             label="sloppy ($\\lambda<10^{-4}\\lambda_{max}$)")
    axd.set_yticks(y)
    axd.set_yticklabels([labels[i] for i in order],
                        fontsize=10 if K <= 12 else 6.5)
    axd.set_xlim(0, 1.02)
    axd.axvline(0.5, ls="--", color="#c0392b", lw=1)
    axd.set_xlabel("participation in flat subspace  "
                   "$\\sqrt{\\sum_{k\\in\\mathrm{null}} V_{jk}^2}$")
    axd.set_title("(d) Non-identifiable-subspace participation\n"
                  "green IDENT · orange WEAK · red NON-IDENT",
                  fontsize=11, fontweight="bold")
    axd.invert_yaxis()
    axd.grid(alpha=0.3, axis="x")
    axd.legend(fontsize=7, loc="lower right")
    if (part_null + part_soft).max() < 1e-3:
        axd.text(0.5, 0.5, "all parameters IDENT\n(no flat / sloppy participation)",
                 transform=axd.transAxes, ha="center", va="center",
                 fontsize=13, fontweight="bold", color="#2c7d3f")

    fig.suptitle(f"Fisher-matrix identifiability — regime: {regime}   "
                 f"({K} params · 10 excitation conditions · 7 observed states · "
                 f"$\\sigma$={SIGMA})", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(out_dir, f"{safe}_fisher.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{regime}] wrote {safe}_fisher.png", flush=True)


def plot_soft_eigvecs(regime, safe, evals, evecs, out_dir, n=4):
    """Paper Fig 6 analogue: the components of the n SOFTEST eigenvectors (the
    smallest-eigenvalue = most non-identifiable directions). The dominant
    component of each names the parameter that direction fails to constrain."""
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.1 * n), sharex=True)
    x = np.arange(K)
    for k, ax in enumerate(axes):                        # k=0 is softest
        v = evecs[:, k]
        v = v * np.sign(v[np.argmax(np.abs(v))])         # sign convention
        dom = np.argmax(np.abs(v))
        cols = ["#c0392b" if i == dom else "#1f5fa8" for i in x]
        ax.bar(x, v, color=cols)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"$\\lambda_{{{k+1}}}$={evals[k]:.2e}", fontsize=9)
        ax.text(0.005, 0.9, f"dominant: {UNKNOWN[dom]}", transform=ax.transAxes,
                va="top", fontsize=10, fontweight="bold", color="#c0392b")
        ax.grid(alpha=0.25, axis="y")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(UNKNOWN, rotation=45, ha="right",
                             fontsize=10 if K <= 12 else 6.5)
    fig.suptitle(f"Softest FIM eigenvectors (non-identifiable directions) — "
                 f"{regime}\ndominant parameter of each flat direction in red",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, f"{safe}_soft_eigvecs.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _tick(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    fs = 10 if len(labels) <= 12 else 5.5
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)


def plot_cross_regime(results, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    colours = ["#1f5fa8", "#2c7d3f", "#e08a1e", "#c0392b"]
    for res, col in zip(results, colours):
        ev = np.clip(res["evals"][::-1], 1e-30, None)
        ax.semilogy(range(1, len(ev) + 1), ev, "o-", ms=4, color=col,
                    label=f"{res['regime']}  (cond={res['cond_number']:.1e})")
    ax.set_xlabel("eigenvalue index (largest → smallest)")
    ax.set_ylabel("FIM eigenvalue")
    ax.set_title("FIM eigenvalue spectra across regimes\n"
                 "(more low eigenvalues = more non-identifiable directions)",
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(out_dir, "cross_regime_spectra.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote cross_regime_spectra.png", flush=True)


def main(out_dir=".", regime="all"):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 70, flush=True)
    print(f"Fisher Information Matrix analysis — {K} params, "
          f"{len(CONDITIONS)} conditions, sigma={SIGMA}", flush=True)
    print("=" * 70, flush=True)
    regimes = list(REGIMES) if regime == "all" else [regime]
    results = [analyse(r, out_dir) for r in regimes]
    if len(results) > 1:
        plot_cross_regime(results, out_dir)
    print("\nDone. Outputs in", out_dir, flush=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    reg = sys.argv[2] if len(sys.argv) > 2 else "all"
    main(out, reg)
