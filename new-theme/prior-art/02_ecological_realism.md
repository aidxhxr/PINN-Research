# Agent report 2 — empirical / ecological realism (2026-09-02)

## VERDICT: PLAUSIBLE WITH CAVEATS

Every ingredient has an empirical analogue, but no documented three-species system matches the full structure, and two assumptions run against the weight of the engineering literature. (1) Density-dependent self-facilitation with a hard upper cap is well documented (beaver pond creation saturates geomorphologically; mussel/cordgrass/seagrass/oyster feedbacks), but the measured form is more often **threshold/sigmoidal** than smoothly saturating. (2) The "engineer-only benefit" assumption is contradicted in the two motivating systems themselves (beaver-created wetlands raise plant richness 33%; ribbed mussels and cordgrass are a *mutualism*), so the model describes an idealised limit. (3) The central prediction (engineering cannot help invasion from rarity; founder control) matches the existing engineer-population theory exactly and is consistent with restoration evidence, but the prediction is model-structure-dependent: it holds for facultative engineers (K(0)>0) and fails for obligate ones.

## 1. Self-facilitating K — what is documented

| Engineer | Evidence that own K rises with own density | Measured shape | Source |
|---|---|---|---|
| Beaver | Pond creation tracked colony numbers 1940–61, then rate collapsed; ponds established by 1961 were 75% of number and 90% of area by 1986; later creation "constrained by geomorphology" — a watershed-set ceiling, as the model's K1+β/γ assumes. BRAT quantifies the dam-capacity ceiling. | Saturating over decades; ceiling is geomorphic. Higher-density populations occupy ponds storing less water. | Johnston & Naiman 1990, doi:10.2307/1938297; Macfarlane et al. 2017 (BRAT); Ronnquist & Westbrook 2021, sciencedirect S0048969721024049 |
| Ribbed mussel (Geukensia) | Aggregation cut mortality from ice and crabs; "population would be reduced by 90% in only five years and no juveniles would survive their second year without an aggregated distribution"; growth reduced at high density. | Positive on survival, negative on growth — net hump/threshold, not pure saturation. | Bertness & Grosholz 1985 Oecologia 67:192, doi:10.1007/BF00384283 |
| Blue mussel (Mytilus edulis) beds | Short-range facilitation (wave protection) vs long-range competition; net facilitation only in winter and only at small scale; clumping raises persistence. | Scale- and season-dependent; sign reverses. | van de Koppel et al. 2005 Am Nat, doi:10.1086/428362; Gascoigne et al. 2005 Oecologia 145:371; de Paoli et al. 2017 PNAS doi:10.1073/pnas.1619203114 |
| Cordgrass (Spartina anglica) | Flume: local sediment-accretion feedback and long-range erosion-trough feedback are both "strongly density-dependent with clear thresholds". Tussocks below a size/density threshold erode. | **Threshold**, not saturating. | Bouma et al. 2009 Oikos 118:260, doi:10.1111/j.1600-0706.2008.16892.x; van Hulzen et al. 2007 Estuaries & Coasts 30:3 |
| Seagrass | SEM on large-scale data: seagrass density → lower turbidity → light (standardised indirect effect 0.19); authors state SEM gives "no evidence for alternative stable states". Restoration meta-analysis (1786 trials): large-scale planting needed to reach "critical mass for stress amelioration". | Positive feedback confirmed; shape not resolved. | van der Heide et al. 2011 PLoS One doi:10.1371/journal.pone.0016504; van Katwijk et al. 2016 J Appl Ecol doi:10.1111/1365-2664.12562 |
| Oyster reef | High-relief reefs: 4x oyster density (~1000/m²) vs low-relief; shell accretion is the feedback. | Threshold in reef height. | Schulte, Burke & Lipcius 2009 Science |
| Cheatgrass | Grass–fire cycle raises post-fire cheatgrass cover; response climate-contingent. | Positive feedback; benefit is mostly via *removing* competitors. | Taylor et al. 2014 Ecosystems doi:10.1007/s10021-014-9771-7; Pilliod et al. 2017 PMC5632665 |
| Sphagnum | Acidification, water retention; bistability reported; benefit again largely via suppressing vascular plants. | Not quantified as K(u). | Eppinga et al. 2008 Plant Ecol doi:10.1007/s11258-007-9309-6 |
| Vibrio biofilm matrix | EPS producers "selectively benefit their clonemates" — the one clean case of a privatised (engineer-only) benefit. | Not measured as K(u). | Nadell & Bassler 2011 PNAS doi:10.1073/pnas.1111147108 |

Counter-evidence: in Mytilus californianus, groups of contiguous mussels "collectively have a lower tenacity than when force is applied to a single individual" (PubMed 25216503), so self-facilitation is not universal even within mussels.

## 2. Three-entity systems

| System | Engineer | Competitors | K self-raised & saturating? | Engineer-only benefit? | Competition structure | Mobility | Fit 0–5 |
|---|---|---|---|---|---|---|---|
| Rocky Mountain riparian willow | Beaver | Elk, moose (muskrat does **not** compete for food; Mott et al. 2013) | Yes (Johnston & Naiman 1990) | **No**: beaver raise plant richness 33% (Wright et al. 2002 Oecologia 132:96) and elk browse beaver-cut willow; interaction is hierarchical and elk-dominated (beaver persist only ≤20 elk/km²: Baker et al. 2012 Ecosphere doi:10.1890/ES12-00058.1; Baker et al. 2005 Ecol Appl doi:10.1890/03-5237) | Hierarchical, asymmetric | Mobile engineer | 3 |
| New England salt marsh | Geukensia (or S. alterniflora) | S. patens, Juncus gerardi | Yes for mussel (Bertness & Grosholz 1985) | **No**: mussel–cordgrass is a facultative mutualism (Bertness 1984 Ecology 65:1794) | Strict hierarchy set by stress tolerance (Bertness 1991 Ecology doi:10.2307/1938909) | Sessile | 3 |
| Wadden Sea intertidal | Mytilus edulis | Barnacles, macroalgae | Yes, scale-dependent | Partly (bed structure hosts many taxa) | Hierarchy (mussels dominate) | Sessile; pattern-forming | 3 |
| Sagebrush steppe | Cheatgrass | Bunchgrasses (Pseudoroegneria, Poa secunda), Artemisia | Feedback yes; saturation not shown | No: fire *harms* competitors, which is the mechanism | Hierarchy, context-dependent | Mobile seed | 2 |
| Scheldt pioneer marsh | Spartina | Salicornia, Scirpus | Yes, threshold | Partly | Colonisation-trait driven | Sessile; patch expansion/collapse observed | 3 (best spatial analogue: Schwarz et al. 2018 Nat Geosci doi:10.1038/s41561-018-0180-y) |
| Vibrio biofilm | EPS producer | Non-producer(s) | Not as K(u) | **Yes** (privatised) | Hierarchy | Sessile | 3 |
| Caribbean reef | Coral | Macroalgae, turf | Feedbacks on both sides; macroalgae self-facilitate too | No | Hierarchy with hysteresis | Sessile | 2 (Mumby et al. 2007; van de Leemput et al. 2016 Coral Reefs doi:10.1007/s00338-016-1439-7) |

No system gives one self-facilitating engineer whose construction benefits only itself, flanked by two non-engineers. Tumour subclones: not assessed (UNVERIFIED).

## 3. Cyclic competition with an engineer

Real RPS systems: colicin E. coli (Kerr et al. 2002 Nature doi:10.1038/nature00823), Uta lizards (Sinervo & Lively 1996 doi:10.1038/380240a0), cryptic reef encrusters (Buss & Jackson 1979 Am Nat doi:10.1086/283381), and intransitivity is widespread in plant communities (Soliveres et al. 2015 Ecol Lett, PubMed 26032242). The closest to "RPS plus habitat modification" is Brassica nigra: high-sinigrin genotypes kill mycorrhizae of heterospecific competitors, creating an intransitive loop among genotypes and species (Lankau & Strauss 2007 Science doi:10.1126/science.1147455; Lankau 2011 J Ecol). But the modification harms rivals rather than raising the constructor's own K. The only "niche construction + RPS" treatment is theoretical (Han, Chen & Hui 2016 Appl Math Comput 284:66, doi:10.1016/j.amc.2016.02.056, cellular automata). No empirical engineer-RPS system found.

## 4. Saturating competition

Nonlinear isoclines are old news (Ayala, Gilpin & Ehrenfeld 1973 TPB 4:331, PubMed 4747658; Schoener 1976 TPB 10:309, PubMed 1013908, where heterospecific interference yields priority effects). Plant ecology standardly uses hyperbolic forms, N/(1+ΣaN) (Law & Watkinson 1987 J Ecol 75:871; Inouye 2001 Ecology 82:2696; Hart, Freckleton & Levine 2018 J Ecol doi:10.1111/1365-2745.12954; Stouffer 2022 MEE doi:10.1111/2041-210X.13965), and non-additive/saturating per-capita effects are argued to be generic (Letten & Stouffer 2019 Ecol Lett doi:10.1111/ele.13211). Caveat: those are **fecundity-divisor** forms with a *joint* denominator over all competitors; the model's *additive, per-species* subtractive terms a_ij N_j/(1+h_ij N_j) are a Beddington–DeAngelis-style construction found in theory (Mathematics 2024, doi:10.3390/math12040562) but no dataset fitted to that exact form was found (UNVERIFIED).

## 5. "Engineering cannot help invasion from rarity"

The engineer-population theory says the same thing, with the same mechanism. Gonzalez, Lambert & Ricciardi 2008 (Oikos 117:1247, doi:10.1111/j.0030-1299.2008.16419.x, open PDF at redpath-staff.mcgill.ca): with engineering rate ∝ I the invader establishes "regardless of the (possibly low) initial abundance"; with rate ∝ I² there is "a threshold value for the initial abundance of the exotic, below which invasion fails" — "an Allee effect ... that emerges from the assumption that habitat engineering is facilitative". Franco & Fontanari (arXiv 1611.09283) state the identical O(u²) condition ("obligate cooperators"). Cuddington, Wilson & Hastings 2009 find bistability and runaway growth from engineer–environment feedback. Empirically, foundation-species restoration repeatedly needs critical mass (van Katwijk 2016; Schulte 2009; Schotanus et al. 2020 Restor Ecol doi:10.1111/rec.13168; Balke et al. 2011 MEPS 440:1). The Spartina Allee effect in Willapa Bay is pollen limitation, not engineering (Davis et al. 2004 PNAS doi:10.1073/pnas.0405230101; Taylor et al. 2004 Ecology doi:10.1890/03-0640) — do not cite it as engineering support.

## Per-ingredient support

| Ingredient | Support | Note |
|---|---|---|
| Self-facilitating K | Strong | Shape mostly threshold/hump, not hyperbolic saturation |
| Saturating competition | Moderate | Divisor/joint form is what is fitted; per-species subtractive form UNVERIFIED |
| Engineer-only benefit | Weak | Contradicted in beaver, mussel–cordgrass, cheatgrass, coral; supported only for biofilm EPS |
| Instantaneous construction | Weak | Beaver active ~4 yr, meadows persist >70 yr (Hastings et al. 2007 Ecol Lett doi:10.1111/j.1461-0248.2006.00997.x); all engineer models since Gurney & Lawton 1996 use a separate habitat variable with decay |
| Cyclic competition with engineer | None empirical | Theory only |

## Strongest reviewer objections and fixes

1. **Engineering is a legacy variable, not K(u).** Add a structure variable S with dS/dt = c(u) − δS and K_u = K1 + βS/(1+γS); then check whether the priority-effect and Maxwell-point results survive the lag (Cuddington 2009 shows lags can generate cycles).
2. **Benefit is shared.** Let K_v, K_w depend on S with coefficients ε_v, ε_w (facilitation cascade); the "engineer-only" case becomes ε=0 and the paper can state where the priority effect disappears.
3. **The invasion claim is structural, not general.** Make explicit that it requires a facultative engineer (K(0)=K1>0); an obligate engineer (Gurney–Lawton, Gonzalez linear case) can invade only via engineering. State both regimes.
4. **Functional form.** Either justify the additive saturating loss from interference theory (Schoener 1976) or re-run with the Hassell/Law–Watkinson joint-denominator form used in the empirical literature and show the results are robust.
5. **Motivating examples contradict assumptions** (beaver–muskrat is not a food competition; mussel–cordgrass is a mutualism). Replace with beaver/elk/moose and Spartina/Salicornia/Scirpus or Vibrio EPS/non-producers.

## UNVERIFIED

- Exact functional form in the MDPI Beddington–DeAngelis competition paper (abstract only).
- Han, Chen & Hui 2016 dynamics (from search summary; abstract elided).
- Bull Math Biol 2020 "two-patch" claim that Cuddington's delayed Allee effect extends to space (abstract elided).
- Cuddington 2009 obligate/non-obligate wording (from search summary; OpenAlex abstract confirms bistability/runaway only).
- Any empirical fit of additive per-species saturating competition terms; tumour niche-construction analogy.
