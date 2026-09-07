# new-theme — Niche construction by a single ecosystem engineer in a three-species competition model

**Status (2026-09-07): publication is the goal; the current report is not submission-ready. Develop a focused theoretical ecology paper after correcting claims and completing targeted checks.** Unrelated to the PINN work in the rest of this repo.

## Current publication context — read first

The user is a sophomore and wants clear, plain-language explanations suitable for discussing the project with their professor. The source proposal remains [combined_report.pdf](combined_report.pdf).

Full, dated assessments:

- [Publication-readiness assessment](prior-art/09_publication_readiness.md): modeling expectations, current strengths and gaps, the growth-rate sensitivity calculation, numerical checks, journal scope, and a proposed pre-submission package.
- [Literature and ecological-plausibility report](prior-art/08_literature_and_ecological_plausibility.md): four close theoretical comparisons, four empirical studies, equation-level differences, supported mechanisms, limitations, and source links.

These assessments supersede conflicting claims in the earlier summary and refinement plan retained below. Their published copies are tracked in `prior-art/`; the original dated working notes remain local. This section records the operational conclusions.

### Conclusions to retain

- **Publication route:** a theoretical mechanism paper does not automatically require fitting a real ecosystem. It does require a useful new ecological insight, appropriate assumptions, correct analysis, and reproducibility. No universal checklist certifies all ecological models, and publication cannot be promised.
- **Novelty:** similar work exists; no exact duplicate was identified in the review, which is not proof of absence. Compare directly with Gonzalez et al. (2008), Vera et al. (2024), Han et al. (2016), and Moreno-Spiegelberg and Gomila (2024). Different equations or additional bifurcation plots alone are not sufficient novelty.
- **Interpretation:** the engineered capacity reduces self-crowding; it does not create an isolated-population Allee effect. The competition response is Holling-II-shaped, not the full Beddington–DeAngelis response. Its half-saturation density is `1/h_ij`.
- **Invasion scope:** construction cannot change the engineer's invasion exponent when the other parameters are fixed and baseline capacity is positive. Its negative sign is specific to the Part-A setting: changing only `r1` from `1.000` to `1.005` changes the exponent from `-0.0043086289` to `+0.0006913711`. This is direct algebra, not a new simulation, and does not establish full three-species persistence after the change.
- **Ecological evidence:** seagrass habitat modification, mussel spatial organization, and localized biofilm benefits support selected ingredients; they do not validate the complete competitive triad. The spatial example's chosen population-diffusion ratio of approximately `14860:40:1` has no matched empirical justification. It is not a proved minimum ratio or proof of ecological impossibility.
- **Existing strengths:** retain credit for the two-parameter sweep, multiple starting states, time-step spot checks, diffusion verification, and noise-initialized 2D simulation. These were reported previously; the September 7 assessment did not rerun them or certify every figure.
- **Important corrections:** the 2020 competition response is not equation-identical to ours; the 2024 paper gives the closer special-case comparison. A habitat lag confined to positive carrying capacity does not automatically change invasion growth or create an Allee effect. The mussel mortality law is not our capacity law. A critical patch size is not a Maxwell parameter. A near-Hopf pair is not a general prerequisite for stationary Turing instability. Part-A engineer monopoly is not global: the engineer-free attractor remains locally stable.

### Next work, in priority order

1. Correct the ecological explanations, scope of claims, parameter meanings, and units or nondimensionalization.
2. Focus on establishment, persistence, and spread: when does environmental improvement help an established population survive, and when can it establish from rarity or expand spatially?
3. Test growth and competition parameters, departures from exact symmetry, and a justified alternative assumption or response law. Do not assume every omitted ecological process must be added.
4. Compare no construction, the engineered model, and a constant-capacity control matched at the equilibrium of interest to separate increased capacity from density-dependent feedback.
5. Resolve branch-classification issues and document convergence of the main thresholds, front speeds, and pattern properties; assemble a reproducible all-figure package with accurate data/code and AI-use disclosures under the chosen journal's policy.

No new experiment is authorized merely by this context update. Follow the repository's tmux and session-recording rules when experiments are subsequently requested.

## What the idea is

Three species. `u` is an ecosystem engineer whose own carrying capacity rises with its own density and saturates; `v`, `w` are ordinary logistic competitors. Interspecific competition uses Holling-II-shaped responses, a restricted case of the broader Beddington–DeAngelis family. Source document: `combined_report.pdf` (25 pp, dated 2026-08-10).

```
du/dt = r1 u [1 − u/K_u(u)] − u ( a12 v/(1+h12 v) + a13 w/(1+h13 w) ),   K_u(u) = K1 + βu/(1+γu)
dv/dt = r2 v (1 − v/K2)     − v ( a21 u/(1+h21 u) + a23 w/(1+h23 w) )
dw/dt = r3 w (1 − w/K3)     − w ( a31 u/(1+h31 u) + a32 v/(1+h32 v) )
```

Spatial version: add `D_i Δ` to each equation, 1D/2D, no-flux boundaries.

The original report's five claimed results (β = construction strength; read with the corrections above):

| # | result | where in report |
|---|--------|-----------------|
| R1 | Engineer's invasion growth rate from rarity is independent of β, γ, so never Chesson-style protected coexistence, only priority effects | §A.1, §A.7 |
| R2 | Four-regime cascade: exclusion → fold to bistable coexistence → pitchfork excludes one competitor → transcritical to monopoly; closed forms (β2 = 247/48) | §A.2–A.5 |
| R3 | Bistable fronts with a Maxwell point β_M ∈ (4.00, 4.15) | §B.1 |
| R4 | Supercritical Hopf (β_H = 0.303) under cyclic competition; subcritical Hopf (β_H = 2.376) in an asymmetric set | Parts C, D.2 |
| R5 | Stationary Turing instability at β = 1.5 with a 14,860:40:1 diffusivity spread; 1D and 2D patterns | §D.3–D.4 |

## Earlier check findings — historical, superseded where noted above

Earlier synthesis: `prior-art/00_verdict_and_refinement.md`.

1. **No exact duplicate**, after two passes: ecology venues (report 01) and applied-math venues with ~70 bibliographic-API queries (report 05). Closest: Castillo-Alvino & Marvá 2020 (our competition term, 2 species), Velasco-Hernández et al. 2017 (constructed variable K, linear competition, 2 species), Li, Liu & Yuan 2019 (competition Turing via cross-diffusion).
2. **The K_u(u) form is one parameter from published theory.** Gonzalez, Lambert & Ricciardi 2008's habitat model (Gurney–Lawton lineage) reduces exactly to `K(I) = βI/(1+γI)` in the fast-habitat limit with linear engineering. That is our form with K1 = 0 (obligate engineer). K1 > 0 and the reduction itself are unclaimed (report 04).
3. **Cuddington, Wilson & Hastings 2009 is still unread.** No open copy exists; ~45 citing works checked. Must be obtained through the library before writing.
4. **Two real systems write the engineer term in our form with parameters:** van de Koppel 2005 mussel beds (full sourced table) and van der Heide 2012 seagrass (saturating engineering driving a lagged sediment variable, two diffusivities, bistability). Name the mussel bed (report 07).
5. **Ecological realism: plausible with caveats** (report 02). Engineer-only benefit and instantaneous construction both fight the literature. The report's motivating examples are wrong: beaver–muskrat is not documented as food competition, mussel–cordgrass is a mutualism.
6. **All numbers reproduce** (`verification/outputs.txt`), **but the Part-D Turing mechanism is misattributed.** The engineer's Jacobian diagonal is always negative for this K(u), so by Satnoianu et al. 2000 patterning needs a negative 2×2 principal minor: a founder-control v–w pair held together by a fast-diffusing engineer. Part D has vw minor −0.0067; Part A is s-stable (no D ever patterns it, proven); a constructed example with the engineer at 110% of its baseline K patterns at σ* = 0.013. Piskovsky 2025's necessary-and-sufficient test confirms pure Turing, not Turing–Hopf. Symmetric competitors can never pattern because the antisymmetric mode decouples from the engineer. The near-Hopf pair is coincidental.
7. **Round-2 reference verification** (report 06) corrected five citations and removed one bad DOI; see `[X]` tags in `references.md`. Piskovsky's inequalities and Satnoianu's classification are transcribed there and implemented in `verification/piskovsky_check.py`.

## Earlier refinement plan — historical, not the current action plan

1. Legacy variable `S`: `dS/dt = c(u) − δS`, `K_u = K1 + βS/(1+γS)`. Show the report is the fast-S limit; find the δ at which the delayed Allee effect appears.
2. Shared benefit: `K_v, K_w = K_i + ε_i βS/(1+γS)`. Map where the priority effect disappears.
3. Rebuild R5 on the negative-minor criterion with Piskovsky classification and the symmetry argument; engineer near its baseline K.
4. Reframe R4 as an unfolding of May–Leonard; compute ℓ1 for the Z3-symmetric saturating system.
5. Name the mussel bed (van de Koppel 2005 Table 1); Michaels 2020 for the critical-patch prediction; Schmitt 2019 / Adam 2022 for the founder-control pair; test the joint-denominator competition form.
6. State and prove R1's corollary; state that K1 = 0 recovers Gonzalez 2008.

## Files

```
new-theme/
├── README.md                              ← this file
├── combined_report.pdf                    ← the original 25-page report
├── references.md                          ← consolidated bibliography, ~130 entries, tagged [V]/[U]/[!]/[X]
├── prior-art/
│   ├── 00_verdict_and_refinement.md       ← synthesis of both rounds (also in notes/2026-09-02-niche-construction-prior-art.md)
│   ├── 01_exact_equation_prior_art.md     ← round 1: has this equation been published? (ecology venues)
│   ├── 02_ecological_realism.md           ← round 1: real systems, per-ingredient support, reviewer objections
│   ├── 03_math_results_prior_art.md       ← round 1: prior art on R1–R5
│   ├── 04_engineer_model_equations.md     ← round 2: the actual equations of Cuddington/Gonzalez/Krakauer/Kylafis/Gross/Wright/UNSW; the QSS reduction
│   ├── 05_applied_math_second_pass.md     ← round 2: duplicate hunt in applied-math venues
│   ├── 06_reference_verification.md       ← round 2: 24 references checked; Piskovsky + Satnoianu conditions transcribed
│   ├── 07_parameterisable_triads.md       ← round 2: which real system can supply numbers
│   ├── 08_literature_and_ecological_plausibility.md ← September 7: four comparisons and four empirical studies
│   └── 09_publication_readiness.md         ← September 7: modeling standards and pre-submission priorities
└── verification/
    ├── verify.py, verify2.py, outputs.txt ← every number in the report reproduced with scipy
    ├── minors.py                          ← principal minors; which species must diffuse fast
    ├── mechanism_test.py                  ← symmetric Part-A family: no founder-control pair possible (0 hits)
    ├── mechanism_test2.py, mechanism_hits.json, mechanism_outputs.txt ← random asymmetric search; constructed example (trial 3868)
    └── piskovsky_check.py, piskovsky_outputs.txt ← Piskovsky 2025 Turing / Turing–Hopf test + Satnoianu s-stability
```

Scripts need numpy + scipy only.

## Earlier open items — consult the current priorities above

- Obtain Cuddington, Wilson & Hastings 2009 (Swarthmore library / ILL) and check whether its fast-environment limit is our model.
- Obtain Kishimoto 1982 to see the pure-competition coefficient matrix.
- Verify every `[U]` in `references.md` before citing.
- Venue after refinement: Theoretical Ecology / J Theor Biol / Bull Math Biol for the current shape; Am Nat or Ecol Lett would need steps 1, 2 and 5.
