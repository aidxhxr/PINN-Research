# Converted from sa.ipynb (repo path: SA-big/sa.ipynb) on 2026-09-23.
# Cells appear in notebook order; code is verbatim. Lines that were IPython-only
# ("!shell", "%magic") are kept but commented with "[ipython-only, skipped]".
# Run from this directory:  cd SA-big && python3 sa.py

# %% [markdown] [cell 1]
# # SA-big — Sensitivity analysis of the 27-state Wnt / Retinoid / HOX model
#
# Large-system counterpart of `PINN/sensitivity_analysis.ipynb`. Same three
# methods (local elasticity, Morris screening, Sobol) applied to the full
# 27-state reaction model `model_big.modelsys_2025_k16dynamic`.
#
# The heavy machinery (baseline parameters, the ±30 % box, the parallel
# model evaluator) lives in `sa_core.py`; this notebook drives it and plots
# the results inline. See `README.md` for the full description.
#
# > **Runtime:** each solve is ~35 s; the cells below parallelise across cores.
# > Increase `MORRIS_N` / `SOBOL_N` for smoother results at higher cost.

# %% [cell 2]
import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

import sa_core as sc

BLUE, ORANGE = "#1f77b4", "#ff7f0e"
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#444444",
    "axes.grid":False,"font.size":11,"axes.titlesize":12,"legend.frameon":False,
})
PARAMS, OUTPUT_NAMES = sc.PARAMS, sc.OUTPUT_NAMES
PLOT_DIR, RES_DIR = "sa_plots", "results"
os.makedirs(PLOT_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)
print(f"{len(PARAMS)} params, {len(OUTPUT_NAMES)} outputs")
print("params :", PARAMS)
print("outputs:", OUTPUT_NAMES)

# %% [markdown] [cell 3]
# ## Baseline
# Nominal settled outputs (time-average over `[6000, 12000]`).

# %% [cell 4]
base = sc.baseline_outputs()
for n, v in zip(OUTPUT_NAMES, base):
    print(f"  {n:<10} {v: .5f}")

# %% [markdown] [cell 5]
# ## 1. Local sensitivity (normalized elasticity)
# Central-difference response of each output to a ±5 % step in each parameter,
# normalized to a dimensionless elasticity `(dy/dp)·(p/y)`.

# %% [cell 6]
REL_STEP = 0.05
X, meta = [], []
for j, key in enumerate(PARAMS):
    for sign in (+1, -1):
        ov = {k: sc.BASELINE[k] for k in PARAMS}
        ov[key] = sc.BASELINE[key]*(1+sign*REL_STEP)
        X.append([ov[k] for k in PARAMS]); meta.append((j, sign))
Yloc = sc.evaluate_matrix(np.array(X))
yj = {m: r for m, r in zip(meta, Yloc)}
Sloc = np.zeros((len(OUTPUT_NAMES), len(PARAMS)))
for j, key in enumerate(PARAMS):
    d = sc.BASELINE[key]*REL_STEP
    dydp = (yj[(j,+1)]-yj[(j,-1)])/(2*d)
    with np.errstate(divide="ignore", invalid="ignore"):
        Sloc[:, j] = dydp*sc.BASELINE[key]/np.where(base==0, np.nan, base)
Sloc = np.nan_to_num(Sloc)
print("elasticity matrix:", Sloc.shape)

# %% [cell 7]
# combined heatmap (outputs x params)
fig, ax = plt.subplots(figsize=(13,5))
vmax = np.abs(Sloc).max() or 1.0
im = ax.imshow(Sloc, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(PARAMS))); ax.set_xticklabels(PARAMS, rotation=90, fontsize=8)
ax.set_yticks(range(len(OUTPUT_NAMES))); ax.set_yticklabels(OUTPUT_NAMES)
ax.set_title("Local sensitivity (elasticity) — all outputs")
fig.colorbar(im, ax=ax, shrink=0.8, label="normalized sensitivity")
fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/local_heatmap_all.png", dpi=150); plt.show()

# %% [cell 8]
# per-output signed bars
for o, name in enumerate(OUTPUT_NAMES):
    s = Sloc[o]; order = np.argsort(np.abs(s))
    colors = [BLUE if s[i]>=0 else ORANGE for i in order]
    fig, ax = plt.subplots(figsize=(7,8))
    ax.barh([PARAMS[i] for i in order], s[order], color=colors)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("normalized local sensitivity (elasticity)")
    ax.set_title(f"{name} — local sensitivity")
    ax.legend(handles=[Patch(color=BLUE,label="positive"),Patch(color=ORANGE,label="negative")],
              loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/local_{sc.slug(name)}.png", dpi=150); plt.show()

# %% [markdown] [cell 9]
# ## 2. Morris elementary-effects screening
# `μ*` = mean absolute elementary effect (overall influence);
# `σ` = std of the elementary effects. `σ > μ*/2` (orange) flags
# nonlinear / interacting parameters.

# %% [cell 10]
MORRIS_N = 24          # trajectories; total solves = MORRIS_N*(P+1)
Xm = morris_sample(sc.problem, MORRIS_N, num_levels=4, seed=42)
Ym = sc.evaluate_matrix(Xm)
mu_star_mat = np.zeros((len(PARAMS), len(OUTPUT_NAMES)))
for o in range(len(OUTPUT_NAMES)):
    Si = morris_analyze(sc.problem, Xm, Ym[:,o], num_levels=4, seed=42, print_to_console=False)
    mu_star_mat[:,o] = Si["mu_star"]
print("Morris done:", Xm.shape[0], "solves")

# %% [cell 11]
for o, name in enumerate(OUTPUT_NAMES):
    Si = morris_analyze(sc.problem, Xm, Ym[:,o], num_levels=4, seed=42, print_to_console=False)
    mu_star, sigma = Si["mu_star"], Si["sigma"]
    order = np.argsort(mu_star); nonlin = sigma > mu_star/2
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,7))
    ax1.scatter(mu_star, sigma, s=55, color=[ORANGE if n else BLUE for n in nonlin], zorder=3)
    for i in range(len(PARAMS)):
        ax1.annotate(PARAMS[i],(mu_star[i],sigma[i]),textcoords="offset points",xytext=(4,3),fontsize=7.5)
    lim=max(mu_star.max(),sigma.max())*1.15+1e-12
    ax1.plot([0,lim],[0,lim/2],"--",color="#888888",lw=0.9,label=r"$\sigma=\mu^*/2$")
    ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)
    ax1.set_xlabel(r"$\mu^*$"); ax1.set_ylabel(r"$\sigma$"); ax1.set_title(f"{name} — Morris"); ax1.legend()
    ax2.barh([PARAMS[i] for i in order], mu_star[order], color=[ORANGE if nonlin[i] else BLUE for i in order])
    ax2.set_xlabel(r"$\mu^*$"); ax2.set_title(f"{name} — influence ranking")
    ax2.legend(handles=[Patch(color=BLUE,label="near-linear"),Patch(color=ORANGE,label=r"nonlinear ($\sigma>\mu^*/2$)")],loc="lower right",fontsize=8)
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/morris_{sc.slug(name)}.png", dpi=150); plt.show()

# %% [cell 12]
# all-outputs mu* heatmap + aggregate ranking
norm = mu_star_mat/(mu_star_mat.max(axis=0,keepdims=True)+1e-12)
fig, ax = plt.subplots(figsize=(9,11))
im = ax.imshow(norm, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(OUTPUT_NAMES))); ax.set_xticklabels(OUTPUT_NAMES, rotation=45, ha="right")
ax.set_yticks(range(len(PARAMS))); ax.set_yticklabels(PARAMS, fontsize=8)
ax.set_title(r"Morris $\mu^*$ (column-normalized)")
fig.colorbar(im, ax=ax, shrink=0.7); fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/morris_mustar_heatmap.png", dpi=150); plt.show()

agg = norm.mean(axis=1); order = np.argsort(agg)
fig, ax = plt.subplots(figsize=(7,11))
ax.barh([PARAMS[i] for i in order], agg[order], color=BLUE)
ax.set_xlabel(r"mean normalized $\mu^*$ across outputs")
ax.set_title("Overall parameter influence (Morris, aggregated)")
fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/morris_overall_ranking.png", dpi=150); plt.show()
for i in np.argsort(-agg)[:10]:
    print(f"  {PARAMS[i]:<10} {agg[i]:.3f}")

# %% [markdown] [cell 13]
# ## 3. Sobol variance decomposition (optional — expensive)
# First-order `S1` and total-order `ST`. `ST ≫ S1` signals interaction.
# Total solves = `SOBOL_N·(P+2)`; bump `SOBOL_N` (power of 2) for tighter CIs.

# %% [cell 14]
SOBOL_N = 32           # power of 2; total solves = SOBOL_N*(P+2)
Xs = sobol_sample.sample(sc.problem, SOBOL_N, calc_second_order=False, seed=42)
Ys = sc.evaluate_matrix(Xs)
ST_mat = np.zeros((len(PARAMS), len(OUTPUT_NAMES)))
x = np.arange(len(PARAMS))
for o, name in enumerate(OUTPUT_NAMES):
    Si = sobol_analyze.analyze(sc.problem, Ys[:,o], calc_second_order=False, seed=42, print_to_console=False)
    S1, ST = Si["S1"], Si["ST"]; ST_mat[:,o]=ST
    fig, ax = plt.subplots(figsize=(13,5)); w=0.4
    ax.bar(x-w/2, S1, w, color=BLUE, label="first-order $S_1$")
    ax.bar(x+w/2, ST, w, color=ORANGE, label="total-order $S_T$")
    ax.axhline(0,color="#444444",lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(PARAMS, rotation=90, fontsize=8)
    ax.set_ylabel("Sobol index"); ax.set_title(f"Sobol — {name}"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/sobol_{sc.slug(name)}.png", dpi=150); plt.show()

# %% [cell 15]
norm = ST_mat/(ST_mat.max(axis=0,keepdims=True)+1e-12)
fig, ax = plt.subplots(figsize=(9,11))
im = ax.imshow(norm, cmap="magma", aspect="auto")
ax.set_xticks(range(len(OUTPUT_NAMES))); ax.set_xticklabels(OUTPUT_NAMES, rotation=45, ha="right")
ax.set_yticks(range(len(PARAMS))); ax.set_yticklabels(PARAMS, fontsize=8)
ax.set_title(r"Sobol $S_T$ (column-normalized)")
fig.colorbar(im, ax=ax, shrink=0.7); fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/sobol_ST_heatmap.png", dpi=150); plt.show()

# %% [markdown] [cell 16]
# ## 4. Cross-method synthesis — model-reduction recommendation
#
# Consolidate local / Morris / Sobol into one per-parameter verdict. A param is a
# **FIX candidate** only when it is negligible in *all three* methods (each
# method's max-normalized, output-aggregated score < 5% of its max).
# Writes results/combined_ranking.csv and sa_plots/combined_ranking.png.

# %% [cell 17]
import synthesize
synthesize.main()
