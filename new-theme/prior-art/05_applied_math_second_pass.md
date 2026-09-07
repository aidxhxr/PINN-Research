# Agent report 5 — second-pass duplicate hunt, applied-math venues (2026-09-02, round 2)

## 1. VERDICT

**NONE FOUND IN THIS PASS** (highest single-paper overlap = 3/5, i.e. PARTIAL on one ingredient each).

Across ~70 distinct queries no paper combines the three defining ingredients: (a) a self-referential saturating carrying capacity K_u(u) = K1 + βu/(1+γu), (b) Holling-II/Beddington–DeAngelis *competition* (not predation) terms, and (c) a three-species system analysed for fold/pitchfork/transcritical/Hopf-with-first-Lyapunov-coefficient plus Turing instability and bistable fronts. Each ingredient exists separately; the combination does not appear to exist, and the one paper that gets closest to (a)+(c) with three species and diffusion *proves patterns do not form* in its setting.

**Methodology.** WebSearch budget exhausted after 16 queries; switched to Crossref `query.bibliographic` (~20), OpenAlex `title_and_abstract.search` with boolean phrase filters (~35, exact hit counts), arXiv API (7), Semantic Scholar (2), plus forward-citation sweeps on the three closest papers.

Telling null results (OpenAlex title+abstract, exact-phrase): `"competition model" AND "variable carrying capacity"` → 0; `"competition" AND "saturating" AND "carrying capacity" AND "bifurcation"` → 0; `"carrying capacity" AND "Turing instability" AND "Hopf" AND competition` → 0; `"three species" AND competition AND "first Lyapunov coefficient"` → 0; `"dynamic carrying capacity" AND competition` → 0; `"habitat modification" AND competition AND bistability` → 0; `"carrying capacity" AND "self-facilitation"` → 1 (irrelevant); `"population-dependent carrying capacity"` → 1 (turtle field study); `"state-dependent carrying capacity"` → 4.

## 2. CANDIDATE TABLE

| # | Citation | DOI | Form of K | Competition term | Spp. | Analyses | Ov. |
|---|---|---|---|---|---|---|---|
|1|Castillo-Alvino & Marvá, *J. Biol. Dyn.* 14 (2020)|10.1080/17513758.2020.1742392|constant|**Holling II competitive response** a v/(1+hv), derived from interfering time|2|equilibria, bi-/tri-stability, priority effects|3|
|2|Velasco-Hernández, Núñez-López, Ramírez-Santiago, Hernández-Rosales, *DCDS-B* 22 (2017) 1099|10.3934/dcdsb.2017054|**variable, niche-constructed** (extra state variable)|classical linear LV|2 + niche var.|equilibria, eigenvalue stability|3|
|3|Li, Liu & Yuan, *Appl. Math. Comput.* 347 (2019)|10.1016/j.amc.2018.10.071|constant|competition with saturation effect|2|Turing bifurcation, amplitude equations (needs **cross**-diffusion)|3|
|4|Liu, Wang, Zhang & Li, *DCDS-B* (2019)|10.3934/dcdsb.2019028|state-dependent (vegetation K reduced by pika)|predator–prey|2|saddle-node, transcritical, Hopf, bistability|3|
|5|Cao, Geng, Li & Qu, *Int. J. Biomath.* (2024)|10.1142/s1793524524500979|variable, ∝ biotic resource|intraguild predation|3 + diffusion|**proves non-existence of non-constant steady states**|3|
|6|Vera, Marvá, García-Garrido & Escalante, *Mathematics* 12 (2024) 562|10.3390/math12040562|constant|Beddington–DeAngelis competitive response|2|coexistence regions, multistability|3|
|7|Safuan & Musa, *AIP Conf. Proc.* (2016)|10.1063/1.4954551|shared biotic-resource state variable|linear LV|2 + resource|stability, bifurcation diagrams|3|
|8|Abdullah, Safuan, Nor & Jamaian, *AIP Conf. Proc.* (2018)|10.1063/1.5041609|as above + harvesting|linear LV|2 + resource|steady states, eigenvalues|2|
|9|Safuan, Towers, Jovanoski & Sidhu, *ANZIAM J.* 53 (2012) 172|10.21914/anziamj.v53i0.4972|K as coupled state variable|none|1|stability, simulation|2|
|10|Yukalov, Yukalova & Sornette, *IJBC* 24 (2014) 1450021|10.1142/s0218127414500217|nonlinear K(N) with delay|self only|1|regime classification, noise|2|
|11|Yukalov, Yukalova & Sornette, *Physica D* 241 (2012) 1270|arXiv:1003.2092|mutual K_i(N_j) (symbiosis)|mutualistic|2|regimes, finite-time singularities|2|
|12|Yukalov, Yukalova & Sornette, *Physica D* 238 (2009)|10.1016/j.physd.2009.05.011|delayed K(N)|—|1|punctuated dynamics|1|
|13|Ganguli, Kar & Mondal, *Int. J. Biomath.* 10 (2017)|10.1142/s1793524517500693|variable (time-dep.) K|IGP + competition, linear|3|Routh–Hurwitz, Hopf, harvest|2|
|14|Ang & Safuan, *Chaos Solitons Fractals* 126 (2019)|10.1016/j.chaos.2019.06.004|variable K, shared resource|IGP|3|stability, bifurcation|2|
|15|Jana & Roy, *Comp. Appl. Math.* 41 (2022)|10.1007/s40314-022-02099-4|variable K, toxicant, delay|IGP|3|stability, delay-Hopf|2|
|16|Franco & Fontanari, *Math. Biosci.* 292 (2017) 76|10.1016/j.mbs.2017.08.002|density-dependent K = modified habitats|—|1 engineer|Ricker chaos, lattice dispersal|2|
|17|Lopes & Fontanari, *MBE* 16 (2019)|10.3934/mbe.2019173|density-dependent K ∝ usable habitats|—|1 engineer|feast/famine cycles|2|
|18|Lutscher, Fink & Zhu, *Bull. Math. Biol.* 82 (2020) 138|10.1007/s11538-020-00818-8|engineered habitat|—|engineer + habitat|spreading fronts, free boundary|2|
|19|Panja, Gayen, Kar & Jana, *Results Control Optim.* (2022)|10.1016/j.rico.2022.100153|constant|nonlinear LV competition|3|Hopf + transcritical|2|
|20|Chang & Wu, *JDE* (2025); Guo, *J. Biol. Dyn.* (2026)|10.1016/j.jde.2025.01.019; 10.1080/17513758.2026.2672784|constant|linear LV|3 + diffusion|bistable wavefronts, wave speeds|2|

Also screened, scored ≤1: Hui/Liu/Zhao *Mathematics* 10 (2022) 2382 and Li *IJB* (2024) — "carrying-capacity-driven diffusion", not K(u); Yue & Chen *Axioms* (2026) — wind-dependent K; Lumi et al. 10.1155/2014/120624 — size-dependent K, single species; Işık *IJBC* (2025) — fractional, discrete, commensal; Manna, Volpert & Banerjee *BMB* 83 (2021) 52 — three-species cyclic competition patterns; Wang Bijun, *Pure Mathematics* 12(1) 2022, doi:10.12677/pm.2022.121009 — Hopf of a predator–prey model with "degenerate carrying capacity", Holling-II, 2 species.

## 3. THE CLOSEST THREE

1. **Castillo-Alvino & Marvá 2020 (J. Biol. Dyn.)** — has exactly the saturating competition term a v/(1+hv), mechanistically derived from interference time, and finds bi-/tri-stability. Differs: two species, constant K, no spatial component, no Hopf/Turing. Follow-ups (Holling IV, 2022; Beddington–DeAngelis, 2024) stay at two species; a 16-paper forward-citation sweep found no three-species or variable-K extension.
2. **Velasco-Hernández et al. 2017 (DCDS-B)** — the only paper found that puts a variable, self-constructed carrying capacity into a competition model. Differs: K is a separate metapopulation "constructed niche" state variable, not a closed-form K(u); competition is linear LV; two species; only equilibria and eigenvalue stability. Zero forward citations in OpenAlex.
3. **Li, Liu & Yuan 2019 (Appl. Math. Comput.)** — closest on the spatial side: a competition model with saturation effect taken to Turing bifurcation with amplitude equations. Differs: two species, constant K, and the instability is cross-diffusion-driven.

Runner-up: Cao et al. 2024 (IJB) — three species, diffusion, variable K tied to a biotic resource — but intraguild predation, and the main theorem is non-existence of non-constant steady states.

## 4. UNVERIFIED / COVERAGE GAPS

- arXiv full text (q-bio.PE/nlin.PS/math.DS) not searched — no public full-text index.
- Chinese-language venues (CNKI/Wanfang) unreachable; coverage partial.
- OpenAlex lacks abstracts for several Elsevier/Springer titles (rows 3, 5, 16, 18, 20 scored from title + venue).
- Thieme/Hsu/Ruan-lineage competition-with-nonlinear-response papers not separately confirmed for a three-species saturating-competition Turing analysis.
