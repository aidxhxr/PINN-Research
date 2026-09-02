# Agent report 3 — prior art on the five mathematical results (2026-09-02)

Note (added by the session, not the agent): the agent's "order-counting error" under R1 was against the paraphrase in its prompt, not the report. The report's Taylor expansion (β enters du/dt at O(u³), per-capita at O(u²)) is correct.

## R1 — Invasion growth rate independent of construction ⇒ only founder control

**Verdict: KNOWN IN SPIRIT; the mathematics is elementary.**

- The invariance itself is trivial: any per-capita growth of the form r(1 − u/K(u)) with 0 < K(0) < ∞ has per-capita growth r at u = 0, whatever K does. What is *not* trivial (and the paper should state) is the corollary: construction cannot raise u's invasion rate but *can lower* v's and w's (u* increases with β, and their invasion rates r_v − a_vu u*/(1+h_vu u*) are monotone decreasing in u*), so construction can only move outcomes from "competitor wins" or "coexistence" toward founder control / engineer monopoly — never toward mutual invasibility.
- That priority effects are exactly the positive-frequency-dependence sector of coexistence theory is Ke & Letten 2018, Nat Ecol Evol, doi:10.1038/s41559-018-0679-z; see also Grainger et al. 2019 PNAS doi:10.1073/pnas.1803122116 and Schreiber, Yamamichi & Strauss 2019 Ecology doi:10.1002/ecy.2664.
- That engineering feedback yields alternative states and that the low-density fate depends on whether the engineer is obligate: Cuddington, Wilson & Hastings 2009 Am Nat 173:488 doi:10.1086/597216; Gurney & Lawton 1996 Oikos doi:10.2307/3546200; Kéfi, Holmgren & Scheffer 2016 Funct Ecol doi:10.1111/1365-2435.12601 (positive interactions ⇒ alternative stable states). The engineer-as-facilitator framing: Bruno, Stachowicz & Bertness 2003 TREE doi:10.1016/S0169-5347(02)00045-9; Koffel, Daufresne & Klausmeier 2021 Ecol Monogr "From competition to facilitation and mutualism" (https://hal.inrae.fr/hal-03282294v1) — could not verify that the latter states the priority-effect corollary explicitly (UNVERIFIED).
- **Important gap the paper should acknowledge**: the independence hinges on K being an *instantaneous* function of u. When the engineered environment is a state variable with its own dynamics (Cuddington et al. 2009; Kylafis & Loreau 2008 Ecol Lett doi:10.1111/j.1461-0248.2008.01220.x), Arroyo-Esquivel & Hastings 2020 Bull Math Biol doi:10.1007/s11538-020-00833-9 report a "delayed Allee effect", i.e. engineering *does* shape low-density dynamics. So "self-facilitation via carrying capacity cannot rescue from rarity" is a statement about the quasi-static limit, not about niche construction in general.
- Also note g'(0) = −r/K1 < 0 always, so there is never an Allee effect at low density; and u = K(u) has exactly one positive root, so single-species bistability is impossible — bistability needs the competitors.

## R2 — Fold → symmetry-breaking pitchfork → transcritical cascade

**Verdict: KNOWN IN SPIRIT / standard exercise; the closed-form thresholds in this model appear new.**

- Symmetry-breaking pitchforks of a Z2-symmetric coexistence state are the generic equivariant scenario and appear in every symmetric two-competitor model; the closest ecological analog with facilitation + symmetric competitors is Moreno-Spiegelberg & Gomila 2023 (seagrass, arXiv:2304.09693) with ten bifurcation regions — whether it contains this exact fold→pitchfork→transcritical sequence is UNVERIFIED.
- "Symmetry breaking in cyclic competition by niche construction" (Han, Chen & Hui 2016, Appl Math Comput 284:66, doi:10.1016/j.amc.2016.02.056) sounds like prior art but is a cellular-automaton metapopulation study, not an ODE bifurcation analysis — the gap is real.
- Bistability from engineering feedback: Cuddington et al. 2009; Gurney & Lawton 1996; Kéfi et al. 2016. Krakauer, Page & Erwin 2009 Am Nat doi:10.1086/593707 treat niche construction as a constructed-K public good in LV competition — check it for overlap.

## R3 — Bistable fronts and a Maxwell point

**Verdict: KNOWN IN SPIRIT (the mathematics is textbook); no prior paper combines an engineer, competitors and a Maxwell point.**

- Zero-speed fronts / speed sign in bistable systems: Lewis & Kareiva 1993 Theor Popul Biol 43:141 doi:10.1006/tpbi.1993.1007; Keitt, Lewis & Holt 2001 Am Nat 157:203 doi:10.1086/318633 (pinning; their continuous-space zero-velocity solution is structurally unstable — precisely the Maxwell-point statement); for competition-diffusion: Gardner 1982 J Diff Eq 44:343, Kan-on 1995 SIAM J Math Anal 26:340, and the 2026 preprint on the symmetric LV speed sign (arXiv:2608.16845).
- Maxwell point as an ecological concept: Bel, Hagberg & Meron 2012 Theor Ecol doi:10.1007/s12080-011-0149-6; Zelnik & Meron 2018 Ecol Indic doi:10.1016/j.ecolind.2018.01.020; Wolbachia analog: Nadin, Strugarek & Vauchelet 2018 J Math Biol doi:10.1007/s00285-017-1181-y.
- Engineer spread models: Lutscher, Fink & Zhu 2020 Bull Math Biol doi:10.1007/s11538-020-00818-8 (free-boundary travelling waves, obligate engineers, no competitors); Arroyo-Esquivel & Hastings 2020 (two-patch); Cuddington & Hastings 2004 Ecol Modell 178:335 (spatially implicit); Franco & Fontanari 2017 Math Biosci 292:76 (lattice).
- **Caveat to state**: the 3-component system is non-variational, so the "Maxwell point" must be *defined* by zero speed, not by equal energy; and in 2D, curvature moves fronts even at the Maxwell point (critical-droplet/nucleation — Lequin, Biroli & Scalliet 2026, arXiv:2608.05251, for two-species LV).

## R4 — Hopf bifurcation in a May–Leonard system with saturation/density-dependent K

**Verdict: ALREADY KNOWN in substance; the framing "saturation converts the heteroclinic cycle to a genuine Hopf" is misleading.**

- Limit cycles in 3-species *linear* LV competition: Gilpin 1975 Am Nat 109:51 doi:10.1086/282973; Zeeman 1993 Dyn Stab Syst 8:189 (classes 26–31 admit Hopf; the May–Leonard-type class 27 has both a heteroclinic cycle and Hopf); Hofbauer & So 1994 Appl Math Lett 7:65; four limit cycles in class 27 (Gyllenberg & Yan) and class 28 (arXiv:2603.24612). Generic perturbations of May–Leonard unfold its degenerate Hopf: Jaramillo, Mrad & Stepien 2023 (arXiv:2210.04342, linear mutation term); Mohd 2019 Appl Math Comput doi:10.1016/j.amc.2019.02.007 (population flow, supercritical Hopf).
- So the ingredient producing a nondegenerate Hopf is *breaking the symmetric α+β=2 degeneracy*, not saturation per se. The paper should compute the first Lyapunov coefficient and check whether a fully Z3-symmetric saturating system remains degenerate.

## R5 — Stationary Turing instability in pure 3-species competition, self-diffusion only

**Verdict: KNOWN IN SPIRIT; the most interesting of the five if framed correctly.**

- Two species: Kishimoto & Weinberger 1985 J Diff Eq 58:15 doi:10.1016/0022-0396(85)90020-8 — for competition-diffusion on *convex* domains all stable equilibria are spatially constant (the theorem needs the convexity caveat). Three species: Kishimoto 1982 J Math Biol doi:10.1007/BF00275163 shows a diffusive 3-species LV system (including a competitive example) *can* have a stable non-constant equilibrium; Kishimoto, Mimura & Yoshida 1983 J Math Biol doi:10.1007/BF00276088 (stable spatio-temporal oscillations, ≥3 species). Recent: Manna, Volpert & Banerjee 2021 Bull Math Biol doi:10.1007/s11538-021-00886-4 (Turing patterns in 3-species cyclic LV competition, self-diffusion, for one cyclic ordering only); Li, Mergia & Patidar 2026 arXiv:2604.12215. General n-species conditions: Satnoianu, Menzinger & Maini 2000 J Math Biol doi:10.1007/s002850000056; Villar-Sepúlveda & Champneys 2023 J Math Biol doi:10.1007/s00285-023-01870-3; Piskovsky 2025 Appl Math Lett doi:10.1016/j.aml.2024.109269 (3-species necessary and sufficient inequalities). Note Levin 1974 Am Nat doi:10.1086/282900 is about founder effects in discrete patches, not a Turing theorem.
- **The near-Hopf precursor is not a necessity.** With this K(u), u·K'(u) − K(u) = −βγu²/(1+γu)² − K1 < 0, so the engineer's diagonal Jacobian entry is *always negative* — no species is a self-activator. With all diagonals negative, det(J − k²D) can only change sign if some 2×2 principal minor M_i < 0, i.e. a two-species sub-block is a saddle (founder-control pair), with the *third* species diffusing fast. That is Satnoianu's "p = 2 activator subsystem" class, and it ties R5 directly to the R2 bistability, not to the Hopf. The complex→real eigenvalue collision along k is the generic dispersion-relation behaviour whenever the homogeneous state is a stable focus (routine in Brusselator/Schnakenberg and in Turing–Hopf work, e.g. Baurmann, Gross & Feudel 2007 J Theor Biol doi:10.1016/j.jtbi.2006.09.036); no paper names it as a distinct route, but it is not novel and not required. Ecosystem-engineer pattern literature (van de Koppel et al. 2005 Am Nat doi:10.1086/428362; Rietkerk & van de Koppel 2008 TREE doi:10.1016/j.tree.2007.10.013; Liu et al. 2013 PNAS doi:10.1073/pnas.1222339110) uses resource or density-dependent-movement mechanisms, not pure competition — a genuine gap.
- "Niche construction" + Turing/reaction-diffusion: no hits 2010–2026 beyond Hui, Li & Yue 2004 Ecol Modell 177:107 (metapopulation "self-organized environmental heterogeneity") and the ecotone paper (Ecography 2020, doi:10.1111/ecog.04902).

## Overall

Genuine novelty: R5 (Turing in pure competition via a bistable founder-control sub-block, self-diffusion only) and, secondarily, the R2 closed forms and the R1 corollary that construction can only *remove* coexistence. Standard exercises in a new model: R3 and R4 (R4 should be reframed as unfolding May–Leonard's degenerate Hopf).

**UNVERIFIED**: Koffel 2021 wording; Kylafis & Loreau 2008 model details; Cuddington 2009 abstract text (paywalled, used secondary sources); Kishimoto 1982 content beyond title; Girardin 2018 three-component KPP–LV claim (search snippet only); an AMC paper reporting RPS + logistic ⇒ subcritical/generalized/zero-Hopf (source not identified); Moreno-Spiegelberg & Gomila pitchfork.
