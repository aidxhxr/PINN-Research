# References — niche-construction 3-species model

Consolidated from the three prior-art reports (2026-09-02). Every entry is tagged
by **role** and by **status**:

- `[V]` — DOI/URL resolved and abstract or full text read by an agent.
- `[U]` — cited from a search snippet or secondary source only; **verify before citing**.
- `[!]` — must be read in full by hand before the project proceeds.

Roles: `PA` prior art on the model / equations · `EMP` empirical grounding ·
`R1..R5` prior art on the five mathematical results (R1 invasion / priority
effect, R2 bifurcation cascade, R3 Maxwell point, R4 Hopf, R5 Turing) ·
`MCT` modern coexistence theory framing · `BG` background already cited in the
report.

## Must-read (duplication risk)

| status | role | citation |
|---|---|---|
| `[!]` | PA, R1, R2 | Cuddington K, Wilson WG, Hastings A (2009). Ecosystem engineers: feedback and population dynamics. *Am Nat* 173:488–498. doi:10.1086/597216. **Check whether the fast-environment quasi-steady-state limit reduces to K(u) = K1 + βu/(1+γu).** Paywalled; agents could not read the equations. |
| `[V]` | PA, R1 | Gonzalez A, Lambert A, Ricciardi A (2008). When does ecosystem engineering cause invasion and species replacement? *Oikos* 117:1247–1257. doi:10.1111/j.0030-1299.2008.16419.x. Open PDF at redpath-staff.mcgill.ca. Linear engineering rate ⇒ invasion from any density; quadratic ⇒ Allee threshold. |
| `[V]` | PA (competition form) | Vera J, Marvá M, García-Garrido VJ, Escalante R (2024). The Beddington–DeAngelis competitive response: intra-species interference enhances coexistence in species competition. *Mathematics* 12(4):562. doi:10.3390/math12040562. Two species, constant K; claims priority on BD competition. Full equations `[U]` (MDPI 403). |
| `[U]` | PA, R2 | Krakauer DC, Page KM, Erwin DH (2009). Diversity, dilemmas, and monopolies of niche construction. *Am Nat* 173:26–40. doi:10.1086/593707. Construction as a public good in LV competition; "monopolies" ≈ Regime IV. |

## Prior art on the model (PA)

| status | citation |
|---|---|
| `[V]` | Gurney WSC, Lawton JH (1996). The population dynamics of ecosystem engineers. *Oikos* 76:273–283. doi:10.2307/3546200. Compartmental habitat model (virgin/usable/degraded). *Already cited in the report.* |
| `[V]` | Cuddington K, Hastings A (2004). Invasive engineers. *Ecol Model* 178:335–347. doi:10.1016/j.ecolmodel.2004.03.010. Two-phase invasion (slow, then explosive). |
| `[V]` | Arroyo-Esquivel J, Hastings A (2020). Spatial dynamics and spread of ecosystem engineers: two-patch analysis. *Bull Math Biol* 82:149. doi:10.1007/s11538-020-00833-9. "Delayed Allee effect" once the environment has dynamics — **this breaks R1 outside the quasi-static limit.** |
| `[V]` | Lutscher F, Fink J, Zhu Y (2020). Pushing the boundaries: models for the spatial spread of ecosystem engineers. *Bull Math Biol* 82. doi:10.1007/s11538-020-00818-8. Free-boundary travelling waves, no competitors. |
| `[V]` | Franco C, Fontanari JF (2017). The spatial dynamics of ecosystem engineers. *Math Biosci* 292:76. arXiv:1611.09283. Lattice model; states the O(u²) "obligate cooperator" condition. |
| `[V]` | Watt S, Jovanoski Z, Towers I, Saifuddin M, Sidhu H (2021). MODSIM24 paper on an engineer plus one resident competitor with K = own habitat size. https://mssanz.org.au/modsim2021/papers/F3/watt2.pdf |
| `[V]` | Liautaud K, Barbier M, Loreau M (2020). Ecological ecotones and engineering feedbacks. *Ecography* 43:1–12. doi:10.1111/ecog.04902. K_i(E) Gaussian in an engineered environment E; sharp fronts, alternative states. |
| `[U]` | Kylafis G, Loreau M (2008). Ecological and evolutionary consequences of niche construction for its agent. *Ecol Lett* 11:1072–1081. doi:10.1111/j.1461-0248.2008.01220.x. Consumer–resource, not K(N). *Already cited in the report.* Model equations not seen. |
| `[U]` | Kylafis G, Loreau M (2011). Niche construction in the light of niche theory. *Ecol Lett* 14:82–90. doi:10.1111/j.1461-0248.2010.01551.x. *Already cited in the report.* |
| `[U]` | Wright JP, Gurney WSC, Jones CG (2004). Patch dynamics in a landscape modified by ecosystem engineers. *Oikos* 105:336–348. Abstract only. |
| `[U]` | Yukalov VI, Yukalova EP, Sornette D (2009). Punctuated evolution due to delayed carrying capacity. *Physica D* 238:1752. doi:10.1016/j.physd.2009.05.011; and (2012) arXiv:1003.2092. Explicit K(N) = A + BN(t−τ) — linear/delayed, not hyperbolic. |
| `[U]` | Gross K (2008). Positive interactions among competitors can produce species-rich communities. *Ecol Lett* 11:929–936. PubMed 18485001. Whether facilitation enters as a saturating K, and whether self-facilitation is included, unknown. |
| `[V]` | Han X, Chen B, Hui C (2016). Symmetry breaking in cyclic competition by niche construction. *Appl Math Comput* 284:66. doi:10.1016/j.amc.2016.02.056. Cellular automaton, not ODE bifurcation analysis; the only "niche construction + RPS" paper found. |
| `[U]` | Hui C, Li Z, Yue D (2004). Metapopulation dynamics and distribution, and environmental heterogeneity induced by niche construction. *Ecol Model* 177:107. |

## Modern coexistence theory framing (MCT, R1)

| status | citation |
|---|---|
| `[V]` | Ke P-J, Letten AD (2018). Coexistence theory and the frequency-dependence of priority effects. *Nat Ecol Evol* 2:1691. doi:10.1038/s41559-018-0679-z. Priority effects = positive frequency dependence sector of MCT. |
| `[V]` | Grainger TN, Levine JM, Gilbert B (2019). The invasion criterion: a common currency for ecological research. *TREE* 34:925–935. *Already cited in the report.* |
| `[V]` | Grainger TN et al. (2019). Applying modern coexistence theory to priority effects. *PNAS*. doi:10.1073/pnas.1803122116. |
| `[V]` | Schreiber SJ, Yamamichi M, Strauss SY (2019). When rarity has costs: coexistence under positive frequency-dependence and environmental stochasticity. *Ecology* 100:e02664. doi:10.1002/ecy.2664. |
| `[V]` | MacDougall AS, Gilbert B, Levine JM (2009). Plant invasions and the niche. *J Ecol* 97:609–615. doi:10.1111/j.1365-2745.2009.01514.x. Positive-frequency-dependent feedbacks "have little role in initial establishment". |
| `[V]` | Ranjan R, Koffel T, Klausmeier CA (2024). The three-species problem. *Ecol Lett* 27:e14426. *Already cited in the report.* |
| `[V]` | Kéfi S, Holmgren M, Scheffer M (2016). When can positive interactions cause alternative stable states in ecosystems? *Funct Ecol* 30:88–97. doi:10.1111/1365-2435.12601. |
| `[V]` | Bruno JF, Stachowicz JJ, Bertness MD (2003). Inclusion of facilitation into ecological theory. *TREE* 18:119–125. doi:10.1016/S0169-5347(02)00045-9. |
| `[U]` | Koffel T, Daufresne T, Klausmeier CA (2021). From competition to facilitation and mutualism: a general theory of the niche. *Ecol Monogr* 91:e01458. https://hal.inrae.fr/hal-03282294v1. |

## Empirical grounding (EMP)

### Self-facilitating carrying capacity
| status | citation |
|---|---|
| `[V]` | Johnston CA, Naiman RJ (1990). Aquatic patch creation in relation to beaver population trends. *Ecology* 71:1617–1621. doi:10.2307/1938297. Pond creation saturates; geomorphic ceiling. |
| `[V]` | Macfarlane WW et al. (2017). Modeling the capacity of riverscapes to support beaver dams (BRAT). *Geomorphology* 277:72–99. |
| `[V]` | Ronnquist AL, Westbrook CJ (2021). Beaver dams: how structure, flow state, and landform control their hydraulic effects. *Sci Total Environ* (S0048969721024049). |
| `[V]` | Bertness MD, Grosholz E (1985). Population dynamics of the ribbed mussel, Geukensia demissa: the costs and benefits of an aggregated distribution. *Oecologia* 67:192–204. doi:10.1007/BF00384283. Hump/threshold, not saturating. |
| `[V]` | van de Koppel J, Rietkerk M, Dankers N, Herman PMJ (2005). Scale-dependent feedback and regular spatial patterns in young mussel beds. *Am Nat* 165:E66–E77. doi:10.1086/428362. |
| `[V]` | de Paoli H et al. (2017). Behavioral self-organization underlies the resilience of a coastal ecosystem. *PNAS* 114:8035. doi:10.1073/pnas.1619203114. |
| `[U]` | Gascoigne JC et al. (2005). Density dependence, spatial scale and patterning in sessile biota. *Oecologia* 145:371–381. |
| `[V]` | Bouma TJ et al. (2009). Density-dependent linkage of scale-dependent feedbacks: a flume study on the intertidal macrophyte Spartina anglica. *Oikos* 118:260–268. doi:10.1111/j.1600-0706.2008.16892.x. **Threshold**, not saturating. |
| `[U]` | van Hulzen JB et al. (2007). Morphological variation and habitat modification are strongly correlated for the autogenic ecosystem engineer Spartina anglica. *Estuaries & Coasts* 30:3–11. |
| `[V]` | van der Heide T et al. (2011). Positive feedbacks in seagrass ecosystems: evidence from large-scale empirical data. *PLoS One* 6:e16504. doi:10.1371/journal.pone.0016504. |
| `[V]` | van Katwijk MM et al. (2016). Global analysis of seagrass restoration: the importance of large-scale planting. *J Appl Ecol* 53:567–578. doi:10.1111/1365-2664.12562. |
| `[U]` | Schulte DM, Burke RP, Lipcius RN (2009). Unprecedented restoration of a native oyster metapopulation. *Science* 325:1124. |
| `[V]` | Taylor K et al. (2014). Climate and fire history interact with cheatgrass invasion. *Ecosystems* 17:1017. doi:10.1007/s10021-014-9771-7. |
| `[U]` | Pilliod DS et al. (2017). Refining the cheatgrass–fire cycle in the Great Basin. *Ecol Evol* (PMC5632665). |
| `[V]` | Eppinga MB et al. (2008). Regular surface patterning of peatlands: confronting theory with field data. *Plant Ecol* / *Ecosystems*. doi:10.1007/s11258-007-9309-6. |
| `[V]` | Nadell CD, Bassler BL (2011). A fitness trade-off between local competition and dispersal in Vibrio cholerae biofilms. *PNAS* 108:14181. doi:10.1073/pnas.1111147108. **The one clean engineer-only benefit.** |
| `[U]` | Mytilus californianus group tenacity counter-example. PubMed 25216503. |
| `[V]` | Hastings A et al. (2007). Ecosystem engineering in space and time. *Ecol Lett* 10:153–164. doi:10.1111/j.1461-0248.2006.00997.x. Legacy/persistence — the argument against instantaneous K(u). |

### Candidate three-species systems
| status | citation |
|---|---|
| `[V]` | Baker BW et al. (2005). Interaction of beaver and elk herbivory reduces standing crop of willow. *Ecol Appl* 15:110–118. doi:10.1890/03-5237. |
| `[V]` | Baker BW et al. (2012). Why aren't there more beaver in Rocky Mountain National Park? *Ecosphere* 3:1–15. doi:10.1890/ES12-00058.1. Beaver persist only ≤20 elk/km². |
| `[U]` | Mott CL et al. (2013). Muskrat–beaver interactions (muskrat is not a food competitor). |
| `[V]` | Wright JP, Jones CG, Flecker AS (2002). An ecosystem engineer, the beaver, increases species richness at the landscape scale. *Oecologia* 132:96–101. Benefit is shared. |
| `[V]` | Bertness MD (1984). Ribbed mussels and Spartina alterniflora production in a New England salt marsh. *Ecology* 65:1794–1807. **Mutualism, not competition.** |
| `[V]` | Bertness MD (1991). Zonation of Spartina patens and Spartina alterniflora in a New England salt marsh. *Ecology* 72:138–148. doi:10.2307/1938909. |
| `[V]` | Schwarz C et al. (2018). Self-organization of a biogeomorphic landscape controlled by plant life-history traits. *Nat Geosci* 11:672–677. doi:10.1038/s41561-018-0180-y. Spartina / Salicornia / Scirpus; patch expansion vs collapse. **Best spatial analogue for R3.** |
| `[U]` | Mumby PJ, Hastings A, Edwards HJ (2007). Thresholds and the resilience of Caribbean coral reefs. *Nature* 450:98–101. |
| `[V]` | van de Leemput IA et al. (2016). Multiple feedbacks and the prevalence of alternate stable states on coral reefs. *Coral Reefs* 35:857. doi:10.1007/s00338-016-1439-7. |

### Cyclic competition
| status | citation |
|---|---|
| `[V]` | Kerr B, Riley MA, Feldman MW, Bohannan BJM (2002). Local dispersal promotes biodiversity in a real-life game of rock–paper–scissors. *Nature* 418:171. doi:10.1038/nature00823. |
| `[V]` | Sinervo B, Lively CM (1996). The rock–paper–scissors game and the evolution of alternative male strategies. *Nature* 380:240. doi:10.1038/380240a0. |
| `[V]` | Buss LW, Jackson JBC (1979). Competitive networks: nontransitive competitive relationships in cryptic coral reef environments. *Am Nat* 113:223. doi:10.1086/283381. |
| `[V]` | Soliveres S et al. (2015). Intransitive competition is widespread in plant communities. *Ecol Lett* 18:790. PubMed 26032242. |
| `[V]` | Lankau RA, Strauss SY (2007). Mutual feedbacks maintain both genetic and species diversity in a plant community. *Science* 317:1561. doi:10.1126/science.1147455. |

### Saturating competition
| status | citation |
|---|---|
| `[V]` | Ayala FJ, Gilpin ME, Ehrenfeld JG (1973). Competition between species: theoretical models and experimental tests. *Theor Popul Biol* 4:331. PubMed 4747658. |
| `[V]` | Schoener TW (1976). Alternatives to Lotka–Volterra competition: models of intermediate complexity. *Theor Popul Biol* 10:309. PubMed 1013908. Interference ⇒ priority effects. |
| `[V]` | Law R, Watkinson AR (1987). Response-surface analysis of two-species competition. *J Ecol* 75:871–886. |
| `[V]` | Inouye BD (2001). Response surface experimental designs for investigating interspecific competition. *Ecology* 82:2696. |
| `[V]` | Hart SP, Freckleton RP, Levine JM (2018). How to quantify competitive ability. *J Ecol* 106:1902. doi:10.1111/1365-2745.12954. |
| `[V]` | Letten AD, Stouffer DB (2019). The mechanistic basis for higher-order interactions and non-additivity in competitive communities. *Ecol Lett* 22:423. doi:10.1111/ele.13211. |
| `[V]` | Stouffer DB (2022). A critical examination of models of annual-plant population dynamics and density-dependent fecundity. *MEE* 13:2437. doi:10.1111/2041-210X.13965. |

### Invasion thresholds / restoration (support for R1's prediction)
| status | citation |
|---|---|
| `[V]` | Schotanus J et al. (2020). Promoting self-facilitating feedback processes in coastal ecosystem engineers to increase restoration success. *Restor Ecol* 28:1105. doi:10.1111/rec.13168. |
| `[U]` | Balke T et al. (2011). Windows of opportunity: thresholds to mangrove seedling establishment on tidal flats. *MEPS* 440:1–9. |
| `[V]` | Davis HG, Taylor CM, Lambrinos JG, Strong DR (2004). Pollen limitation causes an Allee effect in a wind-pollinated invasive grass. *PNAS* 101:13804. doi:10.1073/pnas.0405230101. **Do not cite as engineering support** — pollen limitation. |
| `[V]` | Taylor CM et al. (2004). Consequences of an Allee effect in the invasion of a Pacific estuary by Spartina alterniflora. *Ecology* 85:3254. doi:10.1890/03-0640. Same caveat. |

## Prior art on the mathematical results

### R2 — bifurcation cascade
| status | citation |
|---|---|
| `[U]` | Moreno-Spiegelberg P, Gomila D (2023). Seagrass facilitation model with ten bifurcation regions. arXiv:2304.09693. |

### R3 — bistable fronts, Maxwell point
| status | citation |
|---|---|
| `[V]` | Lewis MA, Kareiva P (1993). Allee dynamics and the spread of invading organisms. *Theor Popul Biol* 43:141. doi:10.1006/tpbi.1993.1007. |
| `[V]` | Keitt TH, Lewis MA, Holt RD (2001). Allee effects, invasion pinning, and species' borders. *Am Nat* 157:203. doi:10.1086/318633. |
| `[V]` | Bel G, Hagberg A, Meron E (2012). Gradual regime shifts in spatially extended ecosystems. *Theor Ecol* 5:591. doi:10.1007/s12080-011-0149-6. Maxwell point as an ecological concept. |
| `[V]` | Zelnik YR, Meron E (2018). Regime shifts by front dynamics. *Ecol Indic* 94:544. doi:10.1016/j.ecolind.2018.01.020. |
| `[V]` | Nadin G, Strugarek M, Vauchelet N (2018). Hindrances to bistable front propagation: application to Wolbachia invasion. *J Math Biol* 76:1489. doi:10.1007/s00285-017-1181-y. |
| `[U]` | Gardner RA (1982). Existence and stability of travelling wave solutions of competition models. *J Diff Eq* 44:343. |
| `[U]` | Kan-on Y (1995). Parameter dependence of propagation speed of travelling waves for competition–diffusion equations. *SIAM J Math Anal* 26:340. |
| `[V]` | Lequin, Biroli, Scalliet (2026). Nucleation theory for bistable LV competition fronts. arXiv:2608.05251. **Curvature moves fronts in 2D even at the Maxwell point.** |
| `[U]` | arXiv:2608.16845 (2026) on the sign of the front speed in symmetric LV competition. |
| `[U]` | Invasion-reversal / Maxwell point paper, *Chaos Solitons Fractals* 2022. doi:10.1016/j.chaos.2022.112899. |

### R4 — Hopf in cyclic competition
| status | citation |
|---|---|
| `[V]` | May RM, Leonard WJ (1975). Nonlinear aspects of competition between three species. *SIAM J Appl Math* 29:243. *Already cited in the report.* Heteroclinic cycle, degenerate Hopf. |
| `[V]` | Gilpin ME (1975). Limit cycles in competition communities. *Am Nat* 109:51. doi:10.1086/282973. |
| `[V]` | Zeeman ML (1993). Hopf bifurcations in competitive three-dimensional Lotka–Volterra systems. *Dyn Stab Syst* 8:189. Classes 26–31 admit Hopf. |
| `[U]` | Hofbauer J, So JW-H (1994). Multiple limit cycles for three-dimensional Lotka–Volterra equations. *Appl Math Lett* 7:65. |
| `[V]` | Jaramillo G, Mrad L, Stepien TL (2023). Dynamics of a linearly perturbed May–Leonard competition model. arXiv:2210.04342. Unfolds the degenerate Hopf. |
| `[V]` | Mohd MH (2019). Diversity in interaction strength promotes rich dynamical behaviours in a three-species ecological system. *Appl Math Comput*. doi:10.1016/j.amc.2019.02.007. |
| `[U]` | Four limit cycles in Zeeman class 28. arXiv:2603.24612. |

### R5 — Turing in pure competition
| status | citation |
|---|---|
| `[V]` | Kishimoto K, Weinberger HF (1985). The spatial homogeneity of stable equilibria of some reaction–diffusion systems on convex domains. *J Diff Eq* 58:15. doi:10.1016/0022-0396(85)90020-8. **Two-species competition never patterns on convex domains** — a referee will ask why three does. |
| `[U]` | Kishimoto K (1982). The diffusive Lotka–Volterra system with three species can have a stable non-constant equilibrium solution. *J Math Biol* 16:103. doi:10.1007/BF00275163. |
| `[U]` | Kishimoto K, Mimura M, Yoshida K (1983). Stable spatio-temporal oscillations of diffusive Lotka–Volterra system with three or more species. *J Math Biol* 18:213. doi:10.1007/BF00276088. |
| `[V]` | Manna K, Volpert V, Banerjee M (2021). Pattern formation in a three-species cyclic competition model. *Bull Math Biol* 83:52. doi:10.1007/s11538-021-00886-4. Self-diffusion only, cyclic LV. |
| `[V]` | Piskovsky V (2024/2025). Necessary and sufficient conditions for Turing and Turing–Hopf in three-species reaction–diffusion. *Appl Math Lett* 160:109269. doi:10.1016/j.aml.2024.109269; arXiv:2405.14682. **Check our instability against these inequalities.** |
| `[V]` | Satnoianu RA, Menzinger M, Maini PK (2000). Turing instabilities in general systems. *J Math Biol* 41:493. doi:10.1007/s002850000056. The "p = 2 activator subsystem" class our mechanism falls into. |
| `[V]` | Villar-Sepúlveda E, Champneys AR (2023). General conditions for Turing and wave instabilities in reaction–diffusion systems. *J Math Biol* 86:39. doi:10.1007/s00285-023-01870-3. |
| `[V]` | Baurmann M, Gross T, Feudel U (2007). Instabilities in spatially extended predator–prey systems. *J Theor Biol* 245:220. *Already cited in the report.* |
| `[U]` | Li, Mergia, Patidar (2026). arXiv:2604.12215. Three-species competition–diffusion patterns. |
| `[V]` | Rietkerk M, van de Koppel J (2008). Regular pattern formation in real ecosystems. *TREE* 23:169. doi:10.1016/j.tree.2007.10.013. |
| `[V]` | Liu Q-X et al. (2013). Phase separation explains a new class of self-organized spatial patterns in ecological systems. *PNAS* 110:11905. doi:10.1073/pnas.1222339110. |
| `[V]` | Levin SA (1974). Dispersion and population interactions. *Am Nat* 108:207. doi:10.1086/282900. Founder effects in discrete patches, **not** a Turing theorem. |
