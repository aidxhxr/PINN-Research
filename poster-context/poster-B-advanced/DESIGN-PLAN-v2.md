# Poster B — RESTRUCTURED as a project walkthrough (v2 plan)

Supersedes `DESIGN-PLAN.md`. v1 was a single-claim poster (three of four columns
were the anchor result). v2 walks the project end to end, in the order the user
asked for:

**model + equations → PINN → Bayesian inverse PINN → neural-mechanistic hybrid**

Same canvas (48 × 36 in landscape), same design system (garnet/paper/ink chrome,
Source Serif 4 / Inter / IBM Plex Mono, SA palette inside figures), same build
pipeline. What changes is the argument and therefore every column.

---

## The through-line

One sentence per act, and each act sets up the next:

1. **The model** — a 7-ODE WNT–RA–HOX system whose stemness index is the
   clinical readout; four regimes span healthy → severe APC loss.
2. **The PINN** — a Fourier-feature network solves it forward from sparse data,
   and inverts it for the 36 parameters — but inversion hits a ceiling.
3. **The Bayesian inverse PINN** — HMC turns that ceiling into a *diagnosis*:
   posteriors say which parameters are identifiable and which are structurally
   degenerate, and it finds the degeneracy unprompted.
4. **The hybrid (UDE)** — replace a mechanistic term with a neural network; the
   identifiability problem returns in a new form, and experiment design (not
   architecture) is what fixes it.

The reader should leave knowing: what the model is, what a PINN does, what
uncertainty quantification bought, and what the hybrid work discovered.

## Title (pick one; recommendation first)

1. **"Physics-Informed Neural Networks for a WNT–RA–HOX Model of Colorectal
   Cancer Stemness"** · subtitle: *forward solves from sparse data, inverse
   parameter recovery, Bayesian identifiability — and what happens when a
   mechanistic term is replaced by a neural network.* (Descriptive title +
   claim-bearing subtitle; correct genre for a walkthrough.)
2. "From a stiff cancer ODE to a hybrid neural model — and what data can
   actually identify" (claim-ier, less standard).
3. "Four ways to ask what your data can identify" (punchy, vaguer).

Author band unchanged: **Pascal Kataboh**, affiliation line, QR space reserved.

## Layout changes from v1

- **Title band** stays ~5.5 in.
- **Fast-path strip shrinks** from 4.05 in to ~2.6 in and changes job: instead
  of "Only got 2 minutes?" it becomes **the roadmap** — four numbered steps that
  name the four columns, so the walkthrough is announced up front. Keeps the
  we-study / we-find / we-claim line as a single row.
- Columns grow to ~27 in tall — needed, because each column is now a full act.
- Numbered badges 1–4 stay and now genuinely mean "read in this order."

---

## Column 1 — THE MODEL

*Eyebrow* `01 · THE SYSTEM` · *Head*: **"Seven equations for a cancer decision."**

- **Why it matters** (3 lines): colorectal crypt stemness; WNT drives
  proliferation, retinoic acid drives differentiation, HOX genes arbitrate; APC
  loss is the canonical adenoma→carcinoma driver. The stemness index is the
  readout that matters clinically.
- **FIG** network schematic (already recolored) with its implication line.
- **EQ 1** — the balance-law form and two representative ODEs (β-catenin and
  HOXA5), *not* all seven, with a one-line note that all seven share the form
  `rate = production − loss`. Spelled-out symbol key beneath.
- **EQ 2** — the stemness index (the scalar readout everything is judged on).
- **Regimes table** (small, typeset): Normal / Early / Advanced / Severe differ
  only in `W` (0.8 → 2.0) and `thetaP` (1.0 → 0.25).
- **FIG** `fig_regimes` — β-catenin and APC trajectories across regimes.
  Implication: rising WNT saturates β-catenin; that is where everything later
  gets hard.
- Fine print: reference solutions are scipy Radau, `rtol 1e-10 / atol 1e-12`,
  horizon T = 150, circadian RA period 24 plus a smooth ATRA pulse.

## Column 2 — THE PINN

*Eyebrow* `02 · THE SOLVER AND THE INVERSE` · *Head*: **"One network, two
jobs — and a ceiling on the second."**

- **NEW FIG** architecture diagram: `t → Fourier features (16 modes, σ=4) → MLP
  (256 wide, 4 deep, GELU, 207,879 params) → 7 states`, with the physics
  residual branch. One line on *why* Fourier features: a plain MLP's spectral
  bias renders the circadian forcing as straight lines.
- **EQ 3** — the PINN loss: data + physics residual + initial condition.
- **Forward result.** **NEW FIG** per-regime error, dense-label baseline vs the
  honest sparse solve (40 observations + IC + physics):
  grand mean rel-L2 **1.06%** vs **2.41%**. Note APC is quoted as NRMSE
  (**1.26%** in Severe) because its rel-L2 is a small-denominator artifact.
- **Inverse result.** 36 parameters recovered from trajectories. Ill-posed:
  the PINN-specific ceiling is ~8/36 from two causes — gradient starvation and
  **autodiff derivative bias**.
- **EQ 4** — the derivative-free integral residual that removes the second
  cause.
- **NEW FIG** recovery counts per regime, baseline vs integral residual:
  **10/4/5/7 → 17/16/10/7**, clean same-conditions total **37 → 50**.
  Implication: the derivative half of the ceiling breaks; Severe does not move,
  so what remains is structural, not numerical.

## Column 3 — THE BAYESIAN INVERSE PINN

*Eyebrow* `03 · UNCERTAINTY` · *Head*: **"A posterior turns the ceiling into a
diagnosis."**

- Framing (2 lines): a point estimate cannot say *why* a parameter failed.
  HMC over the 36 parameters with the state networks frozen — the Bayesian twin
  of the point refine stage, and derivative-free because the likelihood reuses
  the integral residual.
- **EQ 5** — the potential: physics residual likelihood + prior, in the
  log/logit parameterisation.
- **NEW FIG** posterior marginals for `W` and `thetaP` across the four regimes
  (data on disk in `*_posterior_samples.npz`): `W` stays tight, `thetaP`
  degrades as WNT rises.
- **FIG** `fig_posterior` (built in v1, unused) — the `deltaP1`–`thetaP` valley.
  Implication: **the sampler discovers the degeneracy on its own** (corr 0.998);
  only the product `deltaP1·(1−thetaP)` is constrained, and at the healthy truth
  `thetaP = 1` that product vanishes, so `deltaP1` is structurally free.
- **FIG** `fig_fim` — the Fisher cross-check: IDENT/WEAK/NON-IDENT composition
  and condition numbers up to 3.7 × 10¹⁹, degrading monotonically with WNT.
  Implication: three independent views land on the same wall.
- **Honest box** (this is a strength, present it as one): the first HMC run
  reported "36/36 identifiable" and that was an **artifact** — the physics
  likelihood summed ~210k residual terms against a tiny σ, producing a pinpoint
  biased posterior. Honest recovery is **19/20/11/8**; the run still fails its
  own ESS and coverage gates, so the poster claims the *geometry* and the
  *diagnosis*, never the marginal widths.

## Column 4 — THE NEURAL-MECHANISTIC HYBRID

*Eyebrow* `04 · HYBRID / UDE` · *Head*: **"Replace a term with a network and
the identifiability problem comes back."**

- **EQ 6** — `f_known + f_NN`, with the anchor `f_NN(0) = 0` that is supposed to
  stop the network stealing the basal-production constant.
- The failure, in one line with numbers: five constraint parameterisations,
  including the monotone+bounded one the literature recommends, leave the basal
  parameter **14–203%** wrong.
- **FIG** `fig_support` — the diagnosis: no regulator ever approaches zero
  across 10 conditions × 4 regimes, so the anchor is asserted where nothing
  observes it.
- **The causal test**, as a stat-tile pair rather than the full dose-response:
  same edge, same equation, same parameter, same seeds, **eleven conditions in
  both arms** — the arms differ by one siRNA and the error goes
  **2.2% → 0.0%** (Normal), **1.5% → 0.0%** (Severe).
- **FIG** `fig_dose` at reduced size, or its right-hand panel only (error vs
  anchor ratio, showing the collapse).
- **Design table** (typeset, compact): the cheapest protocol reaching each
  edge's anchor, computed with nothing trained — and it predicted two new edges
  prospectively (`m_h13` basal error 95.6% → 0.0% in Severe).
- **Takeaways card** (dark garnet, 3 items) + **limits card**, both as in v1.

---

## What this costs — the trade to accept consciously

The anchor result loses two of its three supporting figures. Specifically, these
drop off the board:

- `fig_param_fail` — the five constraint parameterisations (becomes one line of
  text with the 14–203% range).
- `fig_attribution` — the information-matched control (becomes one sentence:
  *two extra conditions that miss the anchor change nothing in 8 of 8 cells*).
- `fig_prospective` — the prospective test (becomes one number in the design
  table paragraph).

That is the price of a walkthrough: the strongest single result gets one column
instead of three. The scripts stay in the repo, so any of them can be swapped
back in if a column turns out to have room.

## Figure inventory

| # | figure | status |
|---|---|---|
| 1 | network schematic | **have** (recolored TikZ) |
| 2 | regime dynamics | **have** (`fig_regimes`) |
| 3 | PINN architecture diagram | **BUILD** (TikZ or matplotlib) |
| 4 | forward error, dense vs sparse, per regime | **BUILD** (from `forward_error_table.json`) |
| 5 | inverse recovery counts, baseline vs integral | **BUILD** (37 → 50) |
| 6 | posterior marginals `W`/`thetaP` × 4 regimes | **BUILD** (from `*_posterior_samples.npz`) |
| 7 | `deltaP1`–`thetaP` degeneracy valley | **have** (`fig_posterior`, finally used) |
| 8 | FIM verdict composition | **have** (`fig_fim`) |
| 9 | anchor support diagram | **have** (`fig_support`) |
| 10 | dose–response (reduced) | **have** (`fig_dose`, may re-cut to one panel) |

Four new figures; six survive. Every new one has its data on disk already —
nothing needs retraining.

## Equations — six, one to two per column

Matches SIAM's own guidance (a small number of load-bearing display equations,
no derivation chains, each anchored to a figure):

1. balance law + two representative ODEs · 2. stemness index · 3. PINN loss ·
4. integral residual · 5. HMC potential · 6. `f_known + f_NN` with the anchor.

## Open question for the user (does not block building)

Poster B now contains the forward and inverse PINN, which was Poster A's
content. So either Poster A gets renarrowed (e.g. model + sensitivity analysis +
forward PINN only), or it is dropped, or the split is redrawn. Worth deciding
before Poster A is built — it does not affect Poster B.
