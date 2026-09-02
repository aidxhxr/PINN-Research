# 2026-09-02 — New theme: niche-construction 3-species model — prior-art and realism check

Source: `new-theme/combined_report.pdf` (25 pp, "Niche Construction by a Single
Ecosystem Engineer in a Three-Species Competition Model", dated 2026-08-10).
Three literature agents (exact-equation prior art / empirical realism / prior
art on the five mathematical results) + my own scipy re-verification.

## Bottom line

- **No exact duplicate.** Nobody has published `K_u(u) = K1 + βu/(1+γu)` combined
  with per-species saturating (Beddington–DeAngelis-form) competition among three
  species. Every *ingredient* is published; four of the five headline results are
  known in spirit.
- **Biggest duplication risk (must check by hand, paywalled):** Cuddington, Wilson &
  Hastings 2009 Am Nat 173:488 (doi:10.1086/597216). Engineer + environment state
  variable E. If their engineering rate saturates in engineer density, the
  quasi-steady-state limit of their model *is* our K_u(u). Also check Gonzalez,
  Lambert & Ricciardi 2008 Oikos 117:1247 (open PDF): linear engineering rate ⇒
  invader establishes from any density; quadratic ⇒ Allee threshold. That is R1
  in spirit.
- **Ecological realism: plausible with caveats.** Two assumptions run against the
  engineering literature: (a) benefit is engineer-only (contradicted in beaver,
  mussel–cordgrass, cheatgrass, coral; supported only for Vibrio EPS biofilms,
  Nadell & Bassler 2011 PNAS); (b) construction is an instantaneous function of
  density (every engineer model since Gurney & Lawton 1996 uses a separate
  habitat/structure variable with decay; beaver meadows persist >70 yr, Hastings
  et al. 2007). Measured self-facilitation is more often threshold/sigmoidal than
  smoothly saturating (Bouma et al. 2009 Spartina; Bertness & Grosholz 1985
  Geukensia).
- **The report's own motivating examples are wrong**: beaver–muskrat is not a
  food competition (Mott et al. 2013); mussel–cordgrass is a facultative
  mutualism (Bertness 1984). Better triads: beaver / elk / moose (Baker et al.
  2005, 2012); Spartina / Salicornia / Scirpus (Schwarz et al. 2018 Nat Geosci —
  patch expansion/collapse observed = the Maxwell-point phenomenon); Vibrio EPS
  producer / non-producers.

## All numbers re-verified with scipy (Radau/RK45, fsolve, brentq)

V0 = 1.29472709, λ_invade = −0.00430863, β2 = 247/48, regimes at β = 2, 3.7, 3.8,
4.0, 4.2, 4.3, 4.8, 5.2, 5.6 all as reported; Part C Hopf β_H = 0.30308; Part D
Hopf β_H = 2.375971; Turing band q ∈ (1.625, 3.99), q* = 2.5475, σ* = 0.0025606.
Scripts: scratchpad `verify.py`, `verify2.py`, `minors.py`.

## Result-by-result verdicts

| # | claim | verdict | key prior art |
|---|-------|---------|---------------|
| R1 | invasion rate independent of β,γ ⇒ only priority effects | **trivial as math** (any K(0) finite gives it); known in spirit | Cuddington 2009; Gonzalez 2008; MacDougall, Gilbert & Levine 2009 J Ecol; Ke & Letten 2018 NEE. **Holds only in the quasi-static limit** — Arroyo-Esquivel & Hastings 2020 BMB find a "delayed Allee effect" once E has dynamics. Unstated corollary worth stating: construction *lowers* v,w's invasion rates, so it can only move outcomes toward founder control / monopoly, never toward mutual invasibility. |
| R2 | fold → pitchfork → transcritical cascade, closed forms | closed forms new; phenomenology pre-claimed | Krakauer, Page & Erwin 2009 Am Nat ("monopolies of niche construction"); Cuddington 2009 (bistability/runaway); Han, Chen & Hui 2016 AMC (CA, not ODE) |
| R3 | bistable fronts + Maxwell point | textbook technique, new application | Keitt, Lewis & Holt 2001 (pinning); Bel, Hagberg & Meron 2012; Lutscher, Fink & Zhu 2020 BMB (engineer fronts, no competitors). Caveat: 3-component system is non-variational — define Maxwell point by zero speed; in 2D curvature moves fronts (Lequin, Biroli & Scalliet 2026 arXiv:2608.05251). |
| R4 | Hopf in cyclic competition | **largely known**; framing misleading | Zeeman 1993 (classes 26–31 admit Hopf), Gilpin 1975, Jaramillo et al. 2023 (unfolding May–Leonard's degenerate Hopf). What produces a nondegenerate Hopf is breaking the α+β=2 degeneracy, not saturation per se. |
| R5 | Turing in pure competition, self-diffusion only | **most novel — but the report's mechanism is wrong** | Kishimoto 1982 JMB (3-species LV can have stable non-constant states); Manna, Volpert & Banerjee 2021 BMB (cyclic LV Turing); Piskovsky 2025 AML (3-species necessary+sufficient conditions — check ours against it); Satnoianu, Menzinger & Maini 2000. Kishimoto & Weinberger 1985: 2-species competition–diffusion on convex domains never patterns — a referee will ask why 3 species escapes. |

## The corrected Turing mechanism (my check, `minors.py`)

The engineer's Jacobian diagonal is always negative: J_uu = −r1 u (K − uK')/K²
with K − uK' = K1 + βγu²/(1+γu)² > 0. With all diagonals negative, det(J − k²D)
can only change sign if a 2×2 principal minor is negative, i.e. a two-species
sub-block is a saddle (founder-control pair) and the third species diffuses fast.

- Part D, β=1.5: minors uv = +0.0027, uw = +0.0119, **vw = −0.0067**. v and w are a
  mutually-exclusive pair held together by the engineer. Engineer fast, v,w
  sessile ⇒ σ* = +0.00256 (report's D); with v,w at equal tiny D and u fast ⇒
  σ* = +0.0073 at q = 5.6. Swapping so v or w is the fast species ⇒ no
  instability.
- Part A, β=4: all minors positive (0.0119, 0.0119, 0.0012) ⇒ no D ever patterns
  it, which is the §B.2 Routh–Hurwitz finding, now explained.
- So: the near-marginal complex pair at q=0 is *coincidental*, not a precursor.
  The complex→real collision along k is generic for any stable focus. The real
  story is ecological and better: **a mobile engineer stabilising two locally
  mutually-exclusive sessile competitors cannot hold them together everywhere,
  so they segregate into out-of-phase spots.** Note u ≈ 0.023 (5% of K1) at that
  equilibrium — the engineer is nearly absent, so the pattern is effectively a
  v–w pattern brokered by u.

## Refinement recommendations (ordered by payoff)

1. Add a structure/legacy variable S: dS/dt = c(u) − δS, K_u = K1 + βS/(1+γS).
   Test whether R1 (priority-effect-only), the Maxwell point and the cascade
   survive the lag; Cuddington 2009 predicts new cycles. This is what separates
   the work from the quasi-static literature and what reviewers will demand.
2. Shared benefit: K_v, K_w = K_i + ε_i βS/(1+γS). Engineer-only is ε=0. Map where
   the priority effect disappears as ε grows (facilitation cascade).
3. Rebuild R5 around the founder-control sub-block (principal-minor criterion),
   verify against Piskovsky 2025's inequalities, and pick a parameter set where u
   is not 5% of K1. Drop the "novel route via near-Hopf" language.
4. Reframe R4 as an unfolding of May–Leonard's degenerate Hopf; compute ℓ1 for a
   fully Z3-symmetric saturating system to see whether saturation alone
   de-degenerates it.
5. Replace the motivating systems (see above); check robustness to the
   joint-denominator Hassell / Law–Watkinson competition form used empirically.
6. State R1's corollary explicitly and prove it (monotonicity of u* in β).

## Full agent reports

Saved verbatim in `new-theme/prior-art/` (three markdown files).
