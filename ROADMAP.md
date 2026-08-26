# PINN-Research — a chronological review

The whole project is one long fight with a single enemy: **you cannot recover the parameters of a 7-ODE WNT–RA–HOX system from a handful of noisy trajectory samples** — it's ill-posed. Every folder is a new lever tried against that wall. Here's the story in order.

## Phase 0 — Port the model (early June)

Started by converting the MATLAB `modelsys_2025` biology model to Python/SciPy (`git`: "first file converted", "fully converted with plot"). This gave the **7-state reduced WNT–RA–HOX ODE** `[b, apc, h5, h13, m, r, c]`, 4 regimes (Normal / Early adenoma / Cancer-like / Strong APC-mutant) differing only in `W` (WNT drive) and `thetaP` (APC health), and a scipy `solve_ivp` Radau reference solver. This is the ground truth everything is measured against.

## Phase 1 — Forward PINN, and the "just lines" scare (June 20–24)

- First PINN forward solve trained… and produced **straight-line plots**. Panic: "it didn't converge."
- **Diagnosis (2026-06-24): it had converged — the flat plots were stale PNGs** from an earlier plain-MLP run. The real bug was earlier: a plain MLP on scalar time suffers **spectral bias** and physically cannot represent the circadian RA oscillation (period 24) or the ATRA pulse. → This is the **"Activation function change" / Fourier-feature** fix in git: lift time into a bank of sines/cosines (Tancik 2020) so fast modes are representable. Re-running with Fourier features gave curvy plots sitting right on the reference (data MSE 1.4e-5).
- **Second, deeper realization:** that "forward PINN" was **fake as a solver** — it was fed **3000 dense labels** of the already-solved trajectory, so it was just supervised interpolation; the physics residual did nothing.
- **Fix:** built `forward-pinn-train-hybrid/` — the *honest* forward solve: only **40 random sparse observations + IC anchor + physics residual**, so the ODE must actually fill the gaps. Also here the **tmux workflow** was established (an earlier run died mid-L-BFGS when the shell dropped).

## Phase 2 — First inverse PINN, 2 params (June 25)

Built `PINN-inverse-solve/`. Flip the problem: trajectory known → **recover unknown parameters**. Started with just `W` and `thetaP`.

- **Result:** `W` recovered cleanly in *every* regime; `thetaP` recovered only in **low-WNT** regimes (Normal 0.6% err → Strong APC-mutant **142% err**).
- **Key finding (the through-line of the whole project):** this is **genuine structural non-identifiability, not a bug**. At high `W`, β-catenin saturates and the APC sub-dynamics go insensitive to `thetaP` — the loss landscape is flat along it. First statement of the "high-WNT `thetaP` wall."

## Phase 3 — Scale to all 36 params (June 26)

Generalized the inverse solver from 2 → **all 36 identifiable parameters** at once (derived `UNKNOWN` set, data-driven `InverseParams`, param-major reporting). Deliberately over-parameterized and ill-posed — most params expected to recover poorly, *by design*, as the input to a future sensitivity-based pruning.

## Phase 4 — "Better" inverse: the fixes that partly worked (June 29)

Built `PINN-inverse-better/` after reading Engl et al. (*Inverse Problems in Systems Biology*) and Peifer & Timmer (multiple-shooting). Threw four fixes at it:

1. **Multiple experimental conditions** (3: ctrl/noATRA/earlyATRA) — one state-net per condition, **shared** params → the identifiability lever.
2. **Log/relative parametrization** (scale-invariant across the 0.08–3.5 span).
3. **Decoupled LR schedules** — fixed a real **LR-freeze bug** where one cosine schedule drove the params to 1e-6 and froze them.
4. **Adaptive grad-norm loss weighting**.

Also added the user-requested **error-vs-iteration plots** and a **per-parameter identifiability verdict** (YES/MARGINAL/NO + csv/json).

- **Result: disappointing — only 6/36 under 10%,** and `W`/`thetaP` were *worse* than the 2-param solver. The textbook signature appeared: **low total loss at the wrong parameters.**

## Phase 5 — Root-cause proof + the ODE-fit that actually works (June 30)

`PINN-inverse-multicond/`. This is the pivotal diagnostic session. Proved with controlled experiments **why** the PINN tops out:

- `W` freezes within 100 epochs while loss keeps falling 100× → **gradient starvation** (the huge state net drives the residual to zero at wrong θ, starving the θ-gradient).
- Denser/cleaner data didn't fix it. Gradient-matching floored at **autodiff-derivative noise** (lower residual gave a *worse* `W`=0.50).
- **Conclusion: the PINN has its own ceiling ~8/36**, from net-derivative bias + gradient starvation, on top of the fundamental ill-posedness.

**The fix that worked — drop the neural net entirely:** classical **multiple-condition ODE-fit** (`recover_odefit.py`, scipy `least_squares` trust-region over 6 conditions). No net-derivative noise → **18/17/13/13 under 10%** (Normal→Strong), ~2× the PINN. `W`/`thetaP` now excellent in low-WNT, still non-identifiable in high-WNT. Established the realistic ceiling: **~13–18/36 by any method; only more *information* lifts it.**

## Phase 6 — Two failed/successful attacks on the levers (July 1–2)

- **`-multicond-better` (July 1): architecture lever — REJECTED.** Smaller weight-decayed nets + alternating two-timescale optimization, hoping to raise the *PINN* count. Result **9/10/2/4** — flat-to-worse. **Net architecture is not the lever.**
- **`-multicond-excite` (July 1): information lever — CONFIRMED.** Added 4 conditions that directly perturb the WNT & MYC nodes (exogenous pulses) to "excite the dark half of the network." ODE-fit rose to **24/23/21/14** vs 18/17/13/13 baseline (+6/+6/+8/+1). Information works; Strong APC-mutant still stuck (the `thetaP` wall).
- **`pinn-boost` (July 2): break the PINN's *derivative* ceiling — CONFIRMED.** The key idea: swap the biased autodiff `dz/dt` for a **derivative-free trapezoidal/integral residual** (multiple-shooting), plus relative-error weighting, deterministic collocation, multi-start, and a SIREN option. Result: the **PINN itself** jumped to **17/16/10/7** vs the 10/4/5/7 baseline (+24 total, ~2×). The integral residual breaks the autodiff half of the PINN ceiling. Strong APC-mutant unmoved — structural, not derivative-limited.

## Phase 7 — Rigorous identifiability verdicts (July 5–6)

Having a point estimate + ad-hoc verdict wasn't enough. Ran three independent identifiability analyses, all pointing at the same wall:

- **Profile-likelihood** (Raue-style, `identifiability` pkg) across the excite/multicond/pinn-boost folders.
- **Fisher Information Matrix** (`PINN-fisher-matrix/`): each regime has exactly 1 hard-null direction, whose dominant param marches `deltaP1` → `thetaP`+`deltaP1` → **`thetaP`** as WNT rises; `|R|>0.99` correlated pairs climb 0→2→7→10 with WNT drive. cond(FIM) 1e7–1e17 (severely sloppy).
- **Reduced 8-param FIM** (`-top8/`): restricting to the 8 most-identifiable params gives a clean well-posed contrast — cond 1e2–1e3, 0 null directions, all 8 IDENT. Proves the sloppiness is about *which* params, not the method.

## Phase 8 — The Bayesian / UQ turn (July 7–8, ongoing)

Move from point estimates to **posteriors**. SciML Project 1.

- **`PINN-bayesian` (inverse):** HMC over the 36 params with the pinn-boost integral-residual nets **frozen** — the Bayesian version of Stage-3 refine. Deliverable: per-param posterior marginals; a marginal reverting to prior = honest non-identifiability. The smoke test auto-discovered the `deltaP1`–`thetaP` degeneracy (corr 0.998).
  - ⚠️ **First full run (July 7) is BROKEN — do not trust the "36/36 IDENT" headline.** Miscalibration: the physics likelihood summed the residual over ~210k collocation terms with tiny σ → absurdly sharp, biased posterior; ESS≈3, truth 30–50σ out. Honest recovery is only 19/20/11/8. **Being reran (`pinn_bayes_fix`)** with a grid-invariant likelihood (mean residual × fixed effective N) + an ESS/coverage honesty gate.
- **`PINN-forward-bayesian` (forward twin):** params known, HMC over the network **weights** → posterior-predictive trajectory band (UQ on the forward solve). First runs under-covered (0.73→0.56); relaunched with more warmup/draws/leapfrog.

---

## The one-sentence throughline

**Forward PINN works once you fix spectral bias (Fourier features) and stop cheating with dense labels; the inverse problem is fundamentally ill-posed (~13–18/36 ceiling), the PINN adds its own lower ceiling (~8/36) from autodiff-derivative bias + gradient starvation, the *integral residual* breaks the derivative half and *more exciting conditions* is the only lever that raises the fundamental ceiling — and no method moves the high-WNT `thetaP` wall, which three independent identifiability analyses (profile-likelihood, FIM, and now Bayesian HMC) confirm is a real structural/sloppiness limit, not a tuning failure.**
