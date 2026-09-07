# Verdict and refinement — niche-construction 3-species model (2026-09-02, after two research rounds)

Source: `new-theme/combined_report.pdf` (25 pp, "Niche Construction by a Single Ecosystem Engineer in a Three-Species Competition Model", dated 2026-08-10). Seven literature agents in two rounds (reports 01–07 in this folder) plus my own scipy re-verification and two new computations (`verification/`).

## Bottom line

**Not a duplicate. Proceed. But the paper that comes out should be a different paper from the report.**

1. **No one has published this model.** Two full passes (ecology venues; applied-math venues with ~70 bibliographic-API queries) found nothing combining a self-referential saturating K with saturating competition among three species. The closest are Castillo-Alvino & Marvá 2020 (our competition term, two species, constant K), Velasco-Hernández et al. 2017 (constructed variable K, linear competition, two species, zero citations), and Li, Liu & Yuan 2019 (competition Turing, but cross-diffusion).
2. **The K_u(u) form is one parameter away from published theory.** The Gurney & Lawton 1996 → Gonzalez, Lambert & Ricciardi 2008 → Watt et al. 2021 habitat model, in the fast-habitat limit with linear engineering, reduces *exactly* to `dI/dt = r I (1 − I/K(I))` with `K(I) = βI/(1+γI)`, β = Tb/δ, γ = b(δ+ρ)/(ρδ). Triple-verified by the agent (equilibrium, invasion threshold, O(ε) convergence). That is our form with **K1 = 0** (obligate engineer). Our K1 > 0 is the facultative generalisation, and the reduction itself appears nowhere in print. Kylafis & Loreau 2008 independently use `cP/(h+P)` in own density for the same anti-runaway reason.
3. **Cuddington, Wilson & Hastings 2009 remains unread.** No open copy exists on any of ~15 routes and none of ~45 citing OA works reproduces the equations. The abstract's "runaway growth" suggests a non-saturating K, which would make our hyperbolic form the fix rather than a copy. Get it through the library before writing.
4. **Two real systems already write the engineer term in our functional form, with parameters.** van de Koppel et al. 2005 (mussel beds: mortality `d_M k_M/(k_M+M)`, full sourced table, k_M = 150 g m⁻², D = 5×10⁻⁴ m² h⁻¹, 6 m bands) and van der Heide et al. 2012 (seagrass: `s Z/(h₂+Z)` driving a lagged sediment variable, two diffusivities, bistability). The seagrass structure is literally refinement step 1 below. **Name the mussel bed.**
5. **The report's own Turing mechanism is wrong, and the correct one is better.** Details below.

## All numbers in the report re-verified

V0 = 1.29472709, λ_invade = −0.00430863, β2 = 247/48, all four Part-A regimes; Part C Hopf β_H = 0.30308; Part D Hopf β_H = 2.375971; Turing band q ∈ (1.625, 3.99), q* = 2.5475, σ* = 0.0025606. Scripts `verify.py`, `verify2.py`, outputs in `verification/outputs.txt`.

## The corrected Turing mechanism

The engineer's Jacobian diagonal is always negative for this K(u): J_uu = −r1 u (K − uK′)/K² with K − uK′ = K1 + βγu²/(1+γu)² > 0. So no species is a self-activator, and by Satnoianu, Menzinger & Maini 2000 a steady Turing bifurcation needs a **negative 2×2 principal minor** (type p = 2): a two-species sub-block that is a saddle when isolated, i.e. a founder-control pair, with the third species diffusing fast.

| equilibrium | minors (uv, uw, vw) | s-stable? | Piskovsky class, engineer fast | v or w fast |
|---|---|---|---|---|
| Part D, β = 1.5 (report) | +0.0027, +0.0119, **−0.0067** | no | **Turing** (not Turing–Hopf) | none |
| Part A, β = 4 node | +0.0119, +0.0119, +0.0012 | **yes** | none, for any D (Satnoianu Thm 1) | none |
| constructed, trial 3868 | +0.0241, +0.0156, **−0.0117** | no | **Turing**, σ* = 0.013 | none |

(`minors.py`, `piskovsky_check.py`, `mechanism_test2.py`.) The near-marginal complex pair at q = 0 in Part D is coincidental. The complex→real collision along k is generic for any stable focus and is not a mechanism. The report's Part-A Routh–Hurwitz check is superseded by s-stability, which is a proof for *all* D at once.

**Why Part A can never pattern, structurally:** in the v↔w symmetric family the antisymmetric mode (0, 1, −1) is an eigenvector of J with eigenvalue J_vv − J_vw that does not involve the engineer. A founder-control v–w pair would make that eigenvalue positive with nothing to stabilise it. Symmetric competitors are therefore incompatible with the mechanism; v ≠ w is required.

**The ecological statement:** a mobile engineer holds two locally mutually-exclusive sessile competitors in coexistence, but cannot do so everywhere at once, so they segregate into out-of-phase spots. The constructed example has the engineer at 110% of its baseline K1 (the report's Part D had it at 5%), so the pattern is not an artefact of a nearly-absent engineer. Empirical import for the founder-control pair: Schmitt et al. 2019 PNAS and Adam et al. 2022 Ecology (coral / macroalgae order-dependent exclusion).

## Result-by-result verdicts

| # | claim | verdict | key prior art |
|---|---|---|---|
| R1 | invasion rate independent of β, γ ⇒ only priority effects | **Trivial as math** (any finite K(0) gives it); known in spirit. Holds only in the quasi-static limit. | Gonzalez 2008 (linear vs quadratic engineering ⇒ threshold vs Allee, proven in the un-reduced model); Cuddington 2009; MacDougall, Gilbert & Levine 2009; Ke & Letten 2018; Koffel et al. 2021 "Allee niche"; Arroyo-Esquivel & Hastings 2020 "delayed Allee effect" once the environment lags. **State the corollary:** construction lowers v, w's invasion rates monotonically, so it can only move outcomes toward founder control or monopoly. |
| R2 | fold → pitchfork → transcritical, closed forms | Closed forms new; phenomenology pre-claimed | Krakauer, Page & Erwin 2009; Cuddington 2009; Moreno-Spiegelberg & Gomila 2023 (symmetric seagrass competitors: same bifurcations, region-dependent ordering) |
| R3 | bistable fronts + Maxwell point | Textbook technique, new application | Keitt, Lewis & Holt 2001; Bel, Hagberg & Meron 2012; Kenne 2026; Lequin et al. 2026 (2D curvature moves fronts even at the Maxwell point). **Michaels et al. 2020 Ecology formalises the Maxwell point as a critical patch radius with field values.** Non-variational: define by zero speed. |
| R4 | Hopf in cyclic competition | Largely known; framing misleading | Zeeman 1993; Gilpin 1975; Jaramillo et al. 2023 (unfolding May–Leonard's degenerate Hopf). Symmetry breaking, not saturation, is the active ingredient. |
| R5 | Turing in pure competition, self-diffusion only | **Most novel — with the corrected mechanism** | Kishimoto 1982 (one pure-competition example exists, matrix paywalled); Manna et al. 2021; Satnoianu et al. 2000; Piskovsky 2025. Kishimoto & Weinberger 1985: two species never pattern — the negative-minor criterion explains why three can. |

## Ecological realism (report 02 + report 07)

Plausible with caveats. Self-facilitating K is well documented but measured as threshold/sigmoidal more often than hyperbolic (Bouma 2009; Bertness & Grosholz 1985). Two assumptions fight the literature: engineer-only benefit (only Vibrio EPS is clean; Nadell 2015) and instantaneous construction (every engineer model since Gurney & Lawton 1996 uses a lagged habitat variable). The report's motivating examples must be replaced: beaver–muskrat is a lodge-sharing paper not a food-competition one (Mott 2013 verified: says nothing about food), and mussel–cordgrass is a mutualism (Bertness 1984). No real triad has all three ingredients; the mussel bed has the engineer term and parameters, coral reefs have the founder-control pair, beaver willow has the strongest alternative-stable-state evidence (Wolf, Cooper & Hobbs 2007).

## Refinement plan (ordered by payoff)

1. **Legacy variable S** — `dS/dt = c(u) − δS`, `K_u = K1 + βS/(1+γS)` with c saturating. This is van der Heide 2012's structure and Gonzalez 2008's un-reduced structure. Show the report's model is the fast-S limit, then test which of R1–R3 survive finite δ. Arroyo-Esquivel & Hastings 2020 predict a delayed Allee effect; finding the δ at which it appears is a result.
2. **Shared benefit** — `K_v, K_w = K_i + ε_i βS/(1+γS)`; engineer-only is ε = 0. Map where the priority effect disappears.
3. **Rebuild R5 on the negative-minor criterion.** Report Satnoianu type-2, Piskovsky Turing-vs-Turing–Hopf classification, the symmetry argument for why symmetric competitors cannot pattern, and a parameter set with the engineer near its baseline K. Drop the near-Hopf language.
4. **Reframe R4** as an unfolding of May–Leonard's degenerate Hopf; compute ℓ1 for a fully Z3-symmetric saturating system.
5. **Name the mussel bed** with van de Koppel 2005 Table 1; use Michaels 2020 for the critical-patch prediction; import the founder-control pair from Schmitt 2019 / Adam 2022. Test robustness to the joint-denominator competition form (Law & Watkinson; Hart et al. 2018).
6. **State and prove R1's corollary** (monotonicity of u* in β ⇒ competitor invasion rates decrease), and state explicitly that K1 = 0 recovers Gonzalez 2008's obligate case.

## Still unverified

Cuddington 2009 equations `[!]`; Kishimoto 1982 competition matrix; van der Heide 2012 rate parameters; Bouma 2009 threshold densities; Krakauer 2009 Appendix B; Arroyo-Esquivel & Hastings 2020 and Lutscher et al. 2020 full texts. Every `[U]` in `references.md`.
