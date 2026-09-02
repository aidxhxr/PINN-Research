# Agent report 1 — exact-equation prior art (2026-09-02)

## 1. VERDICT

**PARTIAL OVERLAP — no exact duplicate found, but every individual ingredient is published and claims 1, 3, 4 and 5 are substantially anticipated.**

I found no paper combining a hyperbolic self-enhanced carrying capacity `K_u(u) = K1 + βu/(1+γu)` with saturating (Beddington–DeAngelis / Holling-II) *competition* among three species. The engineer-with-density-dependent-K literature (Gurney & Lawton 1996 → Cuddington et al. 2009 → Franco & Fontanari 2017) universally routes the feedback through an explicit habitat/environment state variable rather than a closed-form `K(u)`, and almost always studies one species. The strongest conceptual overlap is Cuddington, Wilson & Hastings (2009), whose obligate/non-obligate distinction is essentially claim 1, and whose results are "altered equilibria, bistability, or runaway growth" — i.e. claim 2's phenomenology, without the three-species cascade.

## 2. Closest papers

| # | Citation | URL/DOI | K(N) or competition form | Contains which of the 5 claims | Score /5 |
|---|---|---|---|---|---|
| 1 | Cuddington, Wilson & Hastings 2009, *Am Nat* 173:488–498 | https://www.journals.uchicago.edu/doi/10.1086/597216 | Engineer + environment state E; engineer alters E at rate ∝ engineer density, E decays exponentially; growth depends on E (**exact form UNVERIFIED**) | **1** (obligate engineers cannot persist without prior modification → alternative states), partial **2** (bistability, runaway) | **4** |
| 2 | Gurney & Lawton 1996, *Oikos* 76:273–283 | https://ui.adsabs.harvard.edu/abs/1996Oikos..76..273G/abstract | Habitat compartments virgin/usable/degraded; K = usable habitat → density-dependent K | none of 2–5; implicit 1 | 3 |
| 3 | Franco & Fontanari 2017, *Math Biosci* | https://arxiv.org/abs/1611.09283 · 10.1016/j.mbs.2017.10.006 | Ricker, "density-dependent carrying capacity given by the number of modified habitats"; lattice + diffusive dispersal | spatial engineer dynamics; no competitors, no Turing/Maxwell | 3 |
| 4 | Watt, Jovanoski, Towers, Saifuddin & Sidhu 2021, MODSIM24 | https://mssanz.org.au/modsim2021/papers/F3/watt2.pdf | **Engineer + one resident competitor**, both logistic with K = own habitat size; habitat recycling | closest to the B-category structure; no saturating competition, no cascade | 3 |
| 5 | Liautaud, Barbier & Loreau 2020, *Ecography* 43:1–12 | https://nsojournals.onlinelibrary.wiley.com/doi/full/10.1111/ecog.04902 | `dN_i/dt = r_iN_i(1−N_i/K_i(E))`, `K_i(E)` Gaussian in E; E driven toward engineer's optimum | **3** (sharp fronts/ecotones), bistability, alternative stable states, multi-species engineers | **4** |
| 6 | Vera, Marvá, García-Garrido & Escalante 2024, *Mathematics* 12(4):562 | https://www.mdpi.com/2227-7390/12/4/562 | **BD competitive response in the classical competition model — claimed as first time**; 2 species, constant K | multistability reinterpreted via interference | **3** (category C) |
| 7 | Krakauer, Page & Erwin 2009, *Am Nat* 173:26–40 | https://www.journals.uchicago.edu/doi/abs/10.1086/593707 | Construction as shared public good; tragedy of the commons | **2** ("monopolies of niche construction"; monopolization stabilizes coexistence) | 3 |
| 8 | Piskovsky 2024, *Appl Math Lett* | https://arxiv.org/abs/2405.14682 | general 3-species reaction–diffusion | **5** — necessary & sufficient conditions, Turing vs Turing–Hopf separated | **4** (method) |
| 9 | Manna, Volpert & Banerjee 2021, *Bull Math Biol* 83:52 | https://link.springer.com/article/10.1007/s11538-021-00886-4 | diffusive May–Leonard cyclic competition | **4**, **5** (stationary/periodic/chaotic patterns) | 3 |
| 10 | Zeeman 1993, Hopf bifurcations in competitive 3D Lotka–Volterra | https://www.semanticscholar.org/paper/0bf035d0e500722ee7f80e7bae93c70cdbfd2a16 | 3D competitive LV | **4** (Hopf is classical here) | 3 |
| 11 | Kylafis & Loreau 2008 *Ecol Lett* 11:1072–1081; 2011 *Ecol Lett* 14:82–90 | 10.1111/j.1461-0248.2008.01220.x · 10.1111/j.1461-0248.2010.01551.x | consumer–resource(+predator) with construction; **not** K(N) | construction modifies competition & coexistence | 2 |
| 12 | Yukalov, Yukalova & Sornette 2009/2012, *Physica D* 238:1752; 241 | 10.1016/j.physd.2009.05.011 · https://arxiv.org/abs/1003.2092 | **explicit functional K(N)**, e.g. `K(N)=A+BN(t−τ)`; carrying capacities as polynomials in populations | literal "K depends on own N" family — but linear/delayed, physics venue | 3 (form) |
| 13 | Cuddington & Hastings 2004, *Ecol Model* 178:335–347 | 10.1016/j.ecolmodel.2004.03.010 | invasive engineer + habitat modification + decay | two-phase invasion (slow then explosive) ≈ claim 1's consequence | 3 |
| 14 | Arroyo-Esquivel & Hastings 2020, *Bull Math Biol* 82:149 | https://link.springer.com/article/10.1007/s11538-020-00833-9 | spatial two-patch extension **of Cuddington et al.** | spatial engineer spread | 3 |
| 15 | Gross 2008, *Ecol Lett* 11:929–936 | https://pubmed.ncbi.nlm.nih.gov/18485001/ | facilitation among competitors raising carrying capacity (**form UNVERIFIED**) | positive interactions + coexistence | 2 |

## 3. Claim-by-claim: known vs new

- **(1) Engineering never rescues the engineer from rarity → only priority effects, never Chesson coexistence.** *Known in substance, new as a theorem for this K(u).* Cuddington et al. 2009's obligate engineer — "net growth rate is negative unless they modify the environment" — plus the alternative-states result is the same statement. MacDougall, Gilbert & Levine 2009 (*J Ecol*, 10.1111/j.1365-2745.2009.01514.x) states the ecological version: positive-frequency-dependent invader feedbacks "have little role in their initial establishment". The clean O(u²) argument and the explicit β,γ-independence of the invasion growth rate is your contribution; frame it as a proof, not a discovery.
- **(2) Four-regime β cascade with closed-form thresholds.** *Genuinely new as stated.* No paper found with this sequence for an engineer + two competitors. But bistability/exclusion/monopoly outcomes under construction are pre-claimed qualitatively by Krakauer et al. 2009 and Cuddington et al. 2009.
- **(3) Bistable fronts with a Maxwell point.** *Known technique, new application.* Maxwell points / front pinning are standard in bistable ecological PDEs (vegetation and forest models; invasion-reversal work, 10.1016/j.chaos.2022.112899; Lequin, Biroli & Scalliet arXiv:2608.05251 derives nucleation theory for bistable LV competition fronts). Claim novelty only for the engineer system.
- **(4) Hopf under cyclic May–Leonard competition with β as parameter.** *Largely known.* Hopf in 3D competitive LV is Zeeman 1993; "using carrying capacity as the bifurcation parameter, models undergo sequences of Hopf bifurcations" is standard. Novelty reduces to β being the knob.
- **(5) Turing instability from a near-Hopf equilibrium.** *Method now published.* Piskovsky 2024 gives necessary and sufficient conditions for 3-species Turing and separates Turing from Turing–Hopf — cite it, and check your instability against it.

## 4. Red flags

- **I found no paper with literally `K = K0 + bN/(1+cN)`.** The nearest literal `K(N)` family is Yukalov/Sornette's `K(N)=A+BN(t−τ)` (linear, delayed) — different form, different field, but it will be raised in review as prior art for the concept.
- **Kishimoto & Weinberger**: the classical Lotka–Volterra competition–diffusion system admits **no stable non-constant positive steady states** without cross-diffusion. Your claim 5 needs an explicit argument for why the saturating terms and the self-K escape this — otherwise a referee kills it in one line. Note also that a "Turing instability requiring very disparate diffusivities" in a competition system is exactly the classic objection to Turing mechanisms in ecology; be ready to defend biological plausibility.
- **Vera et al. 2024 claim BD competition in a competition model is "the first time"** (2024, two species). That is good news for your novelty on ingredient C but means you must cite them and cannot claim the BD-competition idea itself.
- Cuddington et al. 2009's model, in the **fast-environment quasi-steady-state limit**, collapses to a logistic with K a function of engineer density. If their engineering rate saturates in engineer density, their reduced model **is** your `K_u(u)`. This is the single biggest duplication risk.

## 5. UNVERIFIED — check by hand

1. **Cuddington, Wilson & Hastings 2009 exact equations** — paywalled. Check: (a) the functional form linking engineer density to environmental modification rate, (b) whether the QSS reduction yields a hyperbolic `K(E(u))`, (c) whether any competitor is added in a discussion section.
2. **Gross 2008 *Ecol Lett*** — whether facilitation enters as an increasing (saturating?) carrying capacity, and whether any species facilitates *itself*.
3. **Kylafis & Loreau 2008/2011 model equations** — abstracts elided by publishers on all mirrors tried.
4. **Yukalov, Yukalova & Sornette (Physica D 2012; EPJ ST 2012)** — whether any of their `K(N)` forms is hyperbolic rather than linear/polynomial.
5. **Wright, Gurney & Jones 2004 *Oikos* 105:336–348** — patch-dynamic engineer model; only the abstract was reachable.
6. **Vera et al. 2024 full equations** (MDPI returned 403) — confirm they are 2-species and do not treat a variable K.
7. Whether any **Hastings-lab follow-up** (Arroyo-Esquivel & Hastings 2020, both *Bull Math Biol* papers) adds competitor species to the Cuddington framework — abstracts only.
