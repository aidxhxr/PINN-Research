# new-theme — Niche construction by a single ecosystem engineer in a three-species competition model

**Status (2026-09-02): prior-art and realism check done. Verdict: NOT a duplicate; proceed, but reshape before writing.** Unrelated to the PINN work in the rest of this repo.

## What the idea is

Three species. `u` is an ecosystem engineer whose own carrying capacity rises with its own density and saturates; `v`, `w` are ordinary logistic competitors. All interspecific competition is saturating (Beddington–DeAngelis form). Source document: `combined_report.pdf` (25 pp, dated 2026-08-10).

```
du/dt = r1 u [1 − u/K_u(u)] − u ( a12 v/(1+h12 v) + a13 w/(1+h13 w) ),   K_u(u) = K1 + βu/(1+γu)
dv/dt = r2 v (1 − v/K2)     − v ( a21 u/(1+h21 u) + a23 w/(1+h23 w) )
dw/dt = r3 w (1 − w/K3)     − w ( a31 u/(1+h31 u) + a32 v/(1+h32 v) )
```

Spatial version: add `D_i Δ` to each equation, 1D/2D, no-flux boundaries.

The report's five results (β = construction strength is the bifurcation parameter):

| # | result | where in report |
|---|--------|-----------------|
| R1 | Engineer's invasion growth rate from rarity is independent of β, γ (construction enters `du/dt` at O(u³)), so the model never gives Chesson-style protected coexistence, only priority effects | §A.1, §A.7 |
| R2 | Four-regime cascade: exclusion → fold to bistable 3-species coexistence → symmetry-breaking pitchfork excludes one competitor → transcritical to engineer monopoly, with closed forms (β2 = 247/48) | §A.2–A.5 |
| R3 | Bistable fronts with a Maxwell point β_M ∈ (4.00, 4.15) | §B.1 |
| R4 | Supercritical Hopf (β_H = 0.303) under cyclic competition; subcritical Hopf (β_H = 2.376) in a general asymmetric set | Parts C, D.2 |
| R5 | Stationary Turing instability in the asymmetric set at β = 1.5 with a 14,860:40:1 diffusivity spread; 1D and 2D patterns | §D.3–D.4 |

## What the check found

Three literature agents ran in parallel (exact-equation prior art; empirical realism; prior art on R1–R5), plus an independent scipy re-verification of every number in the report. Synthesis in `prior-art/00_verdict_and_refinement.md`.

1. **No exact duplicate.** Nobody has published this `K_u(u)` combined with saturating competition among three species. Every ingredient exists separately.
2. **One paper must be read by hand before committing:** Cuddington, Wilson & Hastings 2009, *Am Nat* 173:488. Its fast-environment quasi-steady-state limit may reduce to our `K_u(u)`. Agents could not get past the paywall.
3. **Four of the five results are known in spirit.** R1: Gonzalez, Lambert & Ricciardi 2008; Cuddington 2009; Ke & Letten 2018. R3: Keitt, Lewis & Holt 2001; Bel, Hagberg & Meron 2012. R4: Zeeman 1993 and unfoldings of May–Leonard's degenerate Hopf — saturation is not the active ingredient, symmetry breaking is. R2's closed forms and R5 are the novel parts.
4. **Ecological realism: plausible with caveats.** Self-facilitating K is well documented (beaver, mussels, Spartina, seagrass, oysters) but is usually threshold-shaped, not hyperbolic. Two assumptions fight the literature: engineer-only benefit (contradicted in beaver, mussel–cordgrass, cheatgrass, coral; only Vibrio biofilm EPS is clean) and instantaneous construction (every engineer model since Gurney & Lawton 1996 uses a lagged habitat variable; Arroyo-Esquivel & Hastings 2020 show the lag creates a delayed Allee effect, which breaks R1).
5. **The report's own motivating examples are wrong.** Beaver and muskrat do not compete for food; mussel–cordgrass is a mutualism. Use beaver / elk / moose, Spartina / Salicornia / Scirpus (Schwarz et al. 2018 Nat Geosci saw the patch expansion-or-collapse R3 predicts), or EPS producer / non-producers.
6. **All numbers reproduce** (`verification/outputs.txt`), but **the Part-D Turing mechanism is misattributed.** The engineer's Jacobian diagonal is always negative for this K(u), so patterning needs a negative 2×2 principal minor. In Part D the v–w minor is −0.0067: v and w alone are a founder-control pair held together by the engineer, and a fast-diffusing engineer cannot hold them together everywhere. Swap which species diffuses fast and the instability vanishes. Part A has all minors positive, which is why no diffusivity ratio ever patterned it. The near-Hopf complex pair is coincidental, not a precursor.

## Refinement plan (ordered by payoff)

1. Add a legacy variable `S`: `dS/dt = c(u) − δS`, `K_u = K1 + βS/(1+γS)`. Test which of R1–R3 survive the lag.
2. Shared benefit: `K_v, K_w = K_i + ε_i βS/(1+γS)`. Engineer-only is ε = 0. Map where the priority effect disappears.
3. Rebuild R5 on the principal-minor criterion, check against Piskovsky 2025's necessary-and-sufficient inequalities, choose a parameter set where u is not 5% of K1.
4. Reframe R4 as an unfolding of May–Leonard; compute ℓ1 for a fully Z3-symmetric saturating system.
5. Replace the motivating systems; test robustness to the joint-denominator (Hassell / Law–Watkinson) competition form used empirically.
6. State and prove R1's corollary: construction lowers v, w's invasion rates monotonically, so it can only move outcomes toward founder control or monopoly.

## Files

```
new-theme/
├── README.md                              ← this file (context + status)
├── combined_report.pdf                    ← the original 25-page report
├── references.md                          ← consolidated bibliography, tagged [V] verified / [U] unverified / [!] must-read
├── prior-art/
│   ├── 00_verdict_and_refinement.md       ← synthesis (same text as notes/2026-09-02-niche-construction-prior-art.md)
│   ├── 01_exact_equation_prior_art.md     ← agent 1: has this equation been published?
│   ├── 02_ecological_realism.md           ← agent 2: real 3-entity systems, per-ingredient support, reviewer objections
│   └── 03_math_results_prior_art.md       ← agent 3: prior art on R1–R5, corrected Turing mechanism
└── verification/
    ├── verify.py                          ← Part A: V0, λ_invade, β2 = 247/48, regimes, rare-engineer runs
    ├── verify2.py                         ← Part C/D Hopf points, Part D Turing band
    ├── minors.py                          ← principal minors of the Jacobians; which species must diffuse fast
    └── outputs.txt                        ← captured output of all three (scipy)
```

`python3 verification/<script>.py` needs numpy + scipy only.

## Open items

- Read Cuddington, Wilson & Hastings 2009 in full (Swarthmore library). Decide whether its QSS limit is our model.
- Verify every `[U]` entry in `references.md` before citing.
- Decide venue after refinement: Theoretical Ecology / J Theor Biol / Bull Math Biol fit the current shape; Am Nat or Ecol Lett would need the legacy-variable and shared-benefit extensions plus a named empirical system.
