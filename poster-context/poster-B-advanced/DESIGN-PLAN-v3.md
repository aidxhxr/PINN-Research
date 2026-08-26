# Poster B — v3 plan (LOCKED). Project walkthrough, built from ready assets.

Supersedes v1 and v2. v3 folds in the user's rules of 2026-08-26 and — after an
asset sweep — replaces almost every "BUILD" with "REUSE". Four new figures in v2
becomes **zero to one** in v3.

## Locked rules

1. **Use the ready schema.** `network-diagram/schematic-better.pdf` — vector,
   clean, legend included. Placed as-is. Do **not** recolor, redraw or
   substitute. (The v1 recolored `schematic_poster.svg` is retired.)
2. **Non-dimensionalization is the essence and must be visible**: dimensional
   system → scaling choice → dimensionless system → 36 parameters → stemness
   index. This is column 1's spine, not a footnote.
3. **Reuse existing graphs.** Everything below is on disk. New figures only
   where nothing exists.
4. **More references, small footprint** — ~14 refs in a two-row footer band at
   16–17 pt. They should read as a dense strip, not a section.
5. **Authors: Amirkhan Aidarkhan, Pascal Kataboh.**
6. **No eyebrow line above the title.** The title is the first thing on the
   board. (Removes the old `IDENTIFIABILITY & EXPERIMENT DESIGN…` kicker.)
7. **No "PINN Research · WNT–RA–HOX model of…" affiliation line.** Author names
   only.

## Canvas and chrome (unchanged from v1/v2)

48 × 36 in landscape · garnet `#9C2745` chrome on paper `#F7F6F3` · Source
Serif 4 display, Inter body, IBM Plex Mono for parameters · SA palette
(`#1f77b4`/`#ff7f0e`/grey/`#cc3333`) inside data figures · 4 columns, numbered
badges, findings as section heads, an implication line under every figure.

Title band loses its kicker and its affiliation line, so it shrinks to ~4.6 in.
The roadmap strip (~2.6 in) replaces v1's "Only got 2 minutes?" strip and names
the four columns. Columns gain the reclaimed height (~28 in).

**Title**: "Physics-Informed Neural Networks for a WNT–RA–HOX Model of
Colorectal Cancer Stemness" · subtitle: *from a nondimensionalized 7-ODE model
to sparse-data forward solves, Bayesian identifiability, and neural-mechanistic
hybrids.*

---

## Column 1 — THE MODEL

*Head*: **"Seven equations, nondimensionalized to 36 parameters."**

- Significance, 3 lines: colorectal crypt stemness; WNT proliferates, retinoic
  acid differentiates, HOX arbitrates; APC loss is the adenoma→carcinoma driver.
- **FIG (READY)** `network-diagram/schematic-better.pdf` — the schema, as-is.
- **The nondimensionalization, shown as a three-step chain** (this is the
  "essence of the job"):
  1. dimensional species and rates;
  2. **the scaling choice** — each concentration by its characteristic value
     (its initial condition), time by the reference β-catenin degradation rate
     `d_B = 1 hr⁻¹`, so `B = B₀b`, …, `τ = d_B t`. One line on *why* `d_B`:
     β-catenin is WNT's direct target and drives most other species, so its
     turnover sets the network's reference timescale.
  3. **the dimensionless system** — one representative equation shown in full
     (`db/dτ`), plus the general form
     `ε_X · dx/dτ = production − x`, with the note that every loss term is
     normalized to `−x` by the scaling and `1/ε_X` is a **pure timescale
     ratio**. Nondimensionalization changed the units and the parameter count
     (**→ 36**), not the model.
- **EQ** the stemness index `S = b(1+α₁₃h₁₃) / [(1+p)(1+α₅h₅)]` — the derived
  readout the whole project predicts.
- **Regimes strip**: Normal → Severe APC Loss differ *only* in `W` (0.8→2.0) and
  `θ_P` (1.0→0.25). Stiff: `ε_R = 0.40`, `ε_M = 0.60`.
- **FIG (READY)** `presentation-codex/.../scipy_core_dynamics.png` *or*
  `research-paper/scipy_solutions.png` — the Radau reference
  (`rtol 1e-10 / atol 1e-12`, T = 150).

## Column 2 — THE PINN

*Head*: **"One network solves it forward from 40 points — and inverts it."**

- **FIG (READY)** `presentation-codex/.../forward_architecture.pdf` — the
  forward PINN schematic: `t → hidden layers → x̂`, the DE branch with fixed
  θ, and the three losses `L_data + L_ic + L_phys` feeding the update. It
  already draws the loss, so no separate loss figure is needed.
- Architecture line: Fourier-feature time embedding (16 modes, σ = 4) →
  256-wide, 4-deep GELU MLP, 207,879 parameters. One line on why: a plain MLP's
  spectral bias renders the circadian forcing as straight lines.
- **FIG (READY)** `research-paper/forward_pinn_hybrid.png` — the honest sparse
  solve (40 observations + IC + physics), PINN dashed over the ODE solver, ATRA
  window shaded. Excellent poster legibility as-is.
- Numbers: dense-label baseline grand rel-L2 **1.06%**; sparse 40-obs
  **2.41%**. APC quoted as NRMSE (**1.26%** Severe) because its rel-L2 is a
  small-denominator artifact.
- **FIG (READY)** `presentation-codex/.../inverse_architecture.pdf` — the same
  diagram with θ now trainable. Makes the forward/inverse distinction visual
  and costs one small panel.
- **FIG (READY)** `research-paper/inv_recovery_bars_best8.png` — true vs
  recovered for the 8 best parameters across regimes.
- The ceiling and the fix, as text + stat tiles: ~8/36 from gradient starvation
  and **autodiff derivative bias**; the derivative-free **integral residual**
  takes recovery **10/4/5/7 → 17/16/10/7**, same-conditions total **37 → 50**.
  Severe does not move → what is left is structural, which hands off to col 3.

## Column 3 — THE BAYESIAN INVERSE PINN

*Head*: **"A posterior says which parameters the data can never pin down."**

- Framing: HMC over the 36 parameters with the state networks frozen — the
  Bayesian twin of the point refine stage, derivative-free because the
  likelihood reuses the integral residual. Dense mass matrix from the Laplace
  Hessian at the MAP, dual-averaging step size.
- **FIG (READY)** `PINN-bayesian/runs/20260713_204442_bayes/bayes_W_thetaP.png`
  — `W` stays tight in every regime; `θ_P` reverts toward the prior as WNT
  rises. This is the headline UQ result and it already exists.
- **FIG (mine, keep)** `fig_posterior` — the `δ_P1`–`θ_P` valley: the sampler
  **finds the degeneracy unprompted** (corr 0.998); only the product
  `δ_P1(1−θ_P)` is constrained, and at the healthy truth `θ_P = 1` it vanishes,
  so `δ_P1` is structurally free. No ready equivalent — the one figure worth
  keeping from v1.
- **FIG (READY)** `research-paper/fim_cross_regime_spectra.png` *or* my
  `fig_fim` — the Fisher cross-check: condition numbers to 3.7 × 10¹⁹,
  degrading monotonically with WNT. Prefer the ready one; fall back to `fig_fim`
  if it reads better at column width.
- **Honest box** (a strength, framed as one): the first HMC run reported
  "36/36 identifiable" — an **artifact** of summing ~210k residual terms against
  a tiny σ. **FIG (READY)** `research-paper/bayes_worst8_severe.png` shows it
  directly: posteriors confidently wrong, truth outside the 95% CI, coverage
  13/36. Honest recovery **19/20/11/8**; the run still fails its ESS and
  coverage gates, so the poster claims geometry and diagnosis, never widths.

## Column 4 — THE NEURAL-MECHANISTIC HYBRID

*Head*: **"Replace a term with a network and identifiability comes back."**

- **EQ** `f_known + f_NN` with the anchor `f_NN(0) = 0` meant to stop the
  network absorbing the basal-production constant.
- One line: five constraint parameterisations — including the monotone+bounded
  one the literature recommends — leave the basal parameter **14–203%** wrong.
- **FIG (mine, keep)** `fig_support` — no regulator ever approaches zero across
  10 conditions × 4 regimes, so the anchor is asserted where nothing observes
  it. No ready equivalent.
- The causal test as stat tiles: same edge, same equation, same parameter, same
  seeds, **eleven conditions in both arms**, one siRNA apart →
  **2.2% → 0.0%** (Normal), **1.5% → 0.0%** (Severe).
- **FIG (mine, keep)** `fig_dose`, likely re-cut to the anchor-ratio panel only.
- One line each for the evidence that no longer gets its own figure: the
  information-matched control (two extra conditions that miss the anchor change
  nothing in 8 of 8 cells) and the prospective test (`m_h13` 95.6% → 0.0% in
  Severe under a protocol chosen from a table with nothing trained).
- **Takeaways card** (dark garnet, 3 items) + **limits card** — noise floor
  ±0.5–0.9 pp, equation-local screen is an upper bound, single seed so no error
  bars, 4 of 6 pre-registered dose predictions failed.

---

## Figure ledger — 11 slots, 8 ready, 3 already built, 0 new

| # | slot | source | status |
|---|---|---|---|
| 1 | network schema | `network-diagram/schematic-better.pdf` | **READY, as-is** |
| 2 | reference dynamics | `scipy_core_dynamics.png` / `scipy_solutions.png` | **READY** |
| 3 | forward PINN architecture | `presentation-codex/.../forward_architecture.pdf` | **READY** |
| 4 | forward fit, 40 obs | `research-paper/forward_pinn_hybrid.png` | **READY** |
| 5 | inverse architecture | `presentation-codex/.../inverse_architecture.pdf` | **READY** |
| 6 | inverse recovery bars | `research-paper/inv_recovery_bars_best8.png` | **READY** |
| 7 | `W`/`θ_P` posteriors | `PINN-bayesian/runs/.../bayes_W_thetaP.png` | **READY** |
| 8 | miscalibration evidence | `research-paper/bayes_worst8_severe.png` | **READY** |
| 9 | FIM spectra | `research-paper/fim_cross_regime_spectra.png` | **READY** |
| 10 | `δ_P1`–`θ_P` valley | `figures/fig_posterior.py` | built (v1) |
| 11 | anchor support | `figures/fig_support.py` | built (v1) |
| 12 | dose–response | `figures/fig_dose.py` | built (v1), may re-cut |

Retired from v1: `fig_regimes`, `fig_fim` (fallback only), `fig_param_fail`,
`fig_attribution`, `fig_prospective`, `schematic_poster` (replaced by the ready
schema). Their scripts stay in the repo.

**Caveat to accept:** the ready figures were drawn for a paper and a deck, so
they carry their own palettes (magenta/green schema, blue/green/orange/red
dynamics) and their own titles. They will not be perfectly uniform with the SA
palette. That is the direct cost of "use the ready ones", and it is the right
trade — they are legible, correct, and already approved. Where a ready figure
has a redundant internal title, the poster crops or covers it rather than
regenerating.

## References — ~14, compact

Two rows in the footer band at 16–17 pt, semicolon-separated, no bullets:
Yang/Meng/Karniadakis (B-PINN, 2020); Raissi/Perdikaris/Karniadakis (PINN,
2019); Karniadakis et al. (Nat Rev Phys, 2021); Tancik et al. (Fourier
features, 2020); Rackauckas et al. (universal DEs, 2020); Loman & Baker
(arXiv:2510.14140); Philipps/Schmid/Hasenauer (npj Syst Biol Appl 11:101,
2025); Wang & Hill (IEEE TNN 17(1):130, 2006); Plate/Martensen/Sager
(arXiv:2408.07143); Jung & Choi (arXiv:2210.11737); Velioglu et al. (2025);
Faure et al. (Nat Commun, 2023); Raue et al. (profile likelihood, 2009);
Engl et al. (inverse problems, 2009). Only verified citations — the two flagged
unverified in the prior-art note stay off.

## Build

Same `build.sh`. Ready PDFs are placed directly by Typst; ready PNGs are
snapshot-copied into `assets/` with provenance recorded in `04-figures.md`.
Only three scripts still run (`fig_posterior`, `fig_support`, `fig_dose`), so
the build gets much faster.
