# Agent report 4 — the engineer-model equations: does any reduce to a hyperbolic K(u)? (2026-09-02, round 2)

## 1. Cuddington, Wilson & Hastings (2009) Am Nat 173:488–498 — **[ABSTRACT ONLY]**

**No open copy exists.** Confirmed exhaustively: OpenAlex `best_oa_location: null`; Semantic Scholar `openAccessPdf: CLOSED`; Google Scholar's "all versions" cluster (`cluster=8148984741996928311`) lists nine versions, all paywalled abstracts (uchicago, JSTOR, PubMed, EuropePMC, ADS, CiNii, PASCAL-FRANCIS). Cuddington's own lab pages (ecotheory.ca/EcoEngineering.htm) list the paper with no PDF. ~45 OA citing works checked via OpenAlex; none reproduces the equations (Schreiber's robust-permanence paper arXiv:1612.06554 and the 2021 aquatic-insect bioRxiv test both cite it only in prose).

Structure, from the OpenAlex abstract: "model an engineer altering environmental conditions at rates dependent on engineer density, with environmental decay back to baseline", determining when feedback yields "altered equilibrium densities, bistability, runaway growth, or changes in density-dependent versus density-independent controls." Basiri, Lutscher & Moameni 2025 (MBE 22:152) cite it as the source for obligate engineers ("species who can only survive after engineering"), example: coastal redwoods.

**Reduces to hyperbolic K(u)? UNKNOWN — equations not obtained.** Note the abstract's "runaway growth" is the signature of a K(u) growing without saturation (linear K(u) = K1 + βu gives unbounded growth when β > 1); a hyperbolic K(u) is exactly what removes it. That is a hypothesis, not a reading of the paper.

## 2. Gonzalez, Lambert & Ricciardi (2008) Oikos 117:1247 — **[READ FULL]** — ★ the hit

PDF: https://redpath-staff.mcgill.ca/ricciardi/Gonzalez_etal2008.pdf. Explicitly "augmented the single-species ecosystem engineer model of Gurney and Lawton (1996)". Verbatim:

```
dR/dt  = r1 R (1 − R/H1)
dI/dt  = r2 I (1 − I/H2)
dH1/dt = ρ(T − H1 − H2) − f(I)H1
dH2/dt = f(I)H1 − δH2
```

R resident, I invading engineer, H1 native habitat, H2 engineered habitat, D = T − H1 − H2 degraded, T total (constant), δ decay, ρ recovery. "Habitat quantity Hi is expressed in units of carrying capacity for species i." Two cases: linear f(I) = bI, quadratic f(I) = cI².

**QSS derivation (agent's, not in the paper).** Setting the two habitat equations to zero: H1 = δH2/f(I), giving

```
K(I) ≡ H2*(I) = ρ T f(I) / [ (δ+ρ) f(I) + ρδ ]
```

**Linear case f(I) = bI** — exactly the target form:

```
K(I) = (Tb/δ)·I / (1 + [b(δ+ρ)/(ρδ)]·I)     ⇒  K1 = 0,  β = Tb/δ,  γ = b(δ+ρ)/(ρδ)
```

so dI/dt = r2 I (1 − I/K(I)) with K hyperbolic in the engineer's own density. Three confirmations: (i) K(I) = I gives I* = ρ(Tb−δ)/(b(δ+ρ)), identical to the paper's H2* = (T−δ/b)/(1+δ/ρ); (ii) K′(0) = Tb/δ > 1 reproduces the paper's invasion threshold b > δ/T; (iii) numerically, max|I_full − I_reduced| = 1.99e-1 / 1.87e-2 / 1.89e-3 as the habitat is sped up by ε = 1 / 0.1 / 0.01 — clean O(ε) convergence.

**Quadratic case f(I) = cI²** gives K(I) ∝ I²/(1+γI²) — Holling III, K′(0) = 0, hence no invasion from low density. Paper's words: "invasion is possible when the invader (I) and its habitat H2 are above some threshold. This is an Allee effect (Taylor and Hastings 2005) that emerges from the assumption that habitat engineering is facilitative." Linear case: "as soon as persistence is possible, it will occur regardless of the initial invader's abundance." Exclusion criteria: δ/bT < 0.1 (linear); δ(1−δ)/ε < 0.18 (nonlinear).

**Reduces to hyperbolic K(u)? YES (linear case) — exactly, but with K1 = 0** (obligate engineer). PARTIAL/NO for the quadratic case (sigmoid → Allee).

## 3. Krakauer, Page & Erwin (2009) Am Nat 173:26–40 — **[READ FULL]**

Open on Karen Page's UCL homepage: https://www.homepages.ucl.ac.uk/~ucackmp/Publications_files/niche.pdf.

```
ẋ = c_x x − x(x + b_yx y)/(k_x z)
ẏ = c_y y − y(y + b_xy x)/(k_y z)
ż = p + (1−c_x)e·x/(x+y+z) + (1−c_y)e·y/(x+y+z) − d z
```

The niche z is a state variable, deliberately: "the niche is not represented indirectly through an algebraic viability function but directly through a differential equation." With y = 0, ẋ = cx(1 − x/K) with K = c·k·z — linear in z. "Monopoly" is formal: each individual monopolizes its own niche with probability m; ESS construction c* = 1/(1+m). The "dilemma" is a continuous-strategy public-goods game: eq. (7) ẇ/w = ẋ/x + (c′−c), so the non-constructor always invades. Three-species version: only in Appendix B, UNVERIFIED (uchicago suppl 403).

**Reduces to hyperbolic K(u)? PARTIAL.** QSS on z gives a quadratic, K(x) = ck·z*(x) saturating from ckp/d to ck[p+(1−c)e]/d — right shape, wrong function. Paper assumes the opposite timescale ordering (niche decays slower than organisms).

## 4. Kylafis & Loreau (2008) Ecol Lett 11:1072–1081 — **[READ FULL, via thesis]**

Title: "Ecological and evolutionary consequences of niche construction for its agent." Obtained verbatim as Ch. 1 of Kylafis' McGill thesis (mcgill.scholaris.ca).

```
u(N,P) = uNP/(k+P)          c(P) = cP/(h+P)
dP/dt = uNP/(k+P) − e_P·P − r·P + a·cP/(h+P)
dN/dt = I − e_N·N − uNP/(k+P) + r·P + (1−a)·cP/(h+P)
```

The construction term c(P) = cP/(h+P) is exactly Michaelis–Menten in the engineer's own density, chosen because density-independent construction "creates the possibility of a boundless autocatalytic process leading to unlimited growth."

**PARTIAL.** MM in own density, yes — but it enters as an additive input flux, not a logistic K. (Kylafis & Loreau 2011 makes construction explicitly linear — NO.)

## 5. Gross (2008) Ecol Lett 11:929 — **[READ FULL]** (https://krgross.github.io/files/publications/Gross-2008-ELE.pdf)

Facilitation enters mortality, not K: m_i = m_i° − d_i(1 − exp{−θ_ij n_j}), saturating in other species' density. "Per capita growth and mortality rates are not subject to intraspecific density dependence." **NO** — a poor citation for this purpose.

## 6. Wright, Gurney & Jones (2004) Oikos 105:336 — **[READ FULL]** (Cary Institute reprint)

dA/dt = nA(1−A−D) − δA, dD/dt = δA − ρD. **PARTIAL:** fast-D QSS gives an exact logistic with K a constant, hyperbolic in the parameter ratio δ/ρ, not in density.

## 7. UNSW Canberra (Safuan / Jovanoski / Towers / Sidhu / Watt) — **[READ FULL, 8 PDFs]**

Every K form catalogued in Safuan's thesis (unsworks.unsw.edu.au): dK/dt = bK − cKN; dK/dt = −α(K−Ks); dK/dt = K(c−dK−eX−fY); dK/dt = −bN; K(t) = a + b sin(ct+φ). **NO in every case** — K is either explicit in time or a state variable coupled bilinearly to N. The thesis raises K = K(N) as a possibility (p. 8) and never uses it. Watt et al. 2021 MODSIM restates the Gonzalez model exactly and derives b_cr = δ/T.

## 8. Meyer & Ausubel (1999) — **[READ FULL]**

K varies with time, not N: κ(t) = κ1 + κ2/(1+exp(−α_κ(t−t_mκ))). **NO.**

## Overall verdict

**Not as written — but one is one parameter away.** The Gurney & Lawton (1996) → Gonzalez et al. (2008) → Watt et al. (2021) lineage, in the fast-habitat QSS limit with linear engineering, is exactly dI/dt = r2 I(1 − I/K(I)) with K(I) = βI/(1+γI), β = Tb/δ, γ = b(δ+ρ)/(ρδ) — triple-verified. **But K1 = 0**: an obligate engineer. K1 > 0 is the non-obligate generalization, and no published model writes it. The QSS reduction itself appears nowhere (all three papers analyse the 3–4-D system directly), and the K1 > 0 interpolation between "obligate" and "facultative" is unclaimed.

Closest independent second: Kylafis & Loreau (2008), whose cP/(h+P) is Monod in own density for the same anti-runaway reason, but acts additively.

Supporting result: hyperbolic K ⇒ threshold in β only (K′(0) > 1); sigmoid K (quadratic engineering) ⇒ Allee effect. Gonzalez et al. prove exactly this dichotomy in the un-reduced system.

## UNVERIFIED / not obtained

1. Cuddington, Wilson & Hastings (2009) equations — the priority target. Needs library/ILL access.
2. Gurney & Lawton (1996) — not fetched directly; structure known via Gonzalez, Franco & Fontanari, Watt.
3. Arroyo-Esquivel & Hastings (2020) BMB 82:149 and Lutscher, Fink & Zhu (2020) BMB 82:138 — closed.
4. Krakauer et al. Appendices A–C (incl. the multi-species extension).
5. Safuan Ecol Modelling 2013 and AMC 2016 — via thesis only.
