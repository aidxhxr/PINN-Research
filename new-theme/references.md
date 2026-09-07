# References — niche-construction 3-species model

Consolidated from seven prior-art reports (rounds 1 and 2, 2026-09-02). Tags:

- `[V]` — DOI/URL resolved and abstract or full text read by an agent.
- `[U]` — cited from a search snippet or secondary source only; **verify before citing**.
- `[!]` — must be read in full by hand before the project proceeds.
- `[X]` — round-2 verification found the citation wrong; corrected or removed as noted.

Roles: `PA` prior art on the model / equations · `EMP` empirical grounding · `R1..R5` prior art on the five mathematical results (R1 invasion / priority effect, R2 bifurcation cascade, R3 Maxwell point, R4 Hopf, R5 Turing) · `MCT` modern coexistence theory framing.

## Must-read (duplication risk)

| status | role | citation |
|---|---|---|
| `[!]` | PA, R1, R2 | Cuddington K, Wilson WG, Hastings A (2009). Ecosystem engineers: feedback and population dynamics. *Am Nat* 173:488–498. doi:10.1086/597216. **No open copy exists anywhere** (nine paywalled versions, ~45 citing OA works checked, none reproduces the equations). Needs library/ILL. Abstract: engineer alters environment at density-dependent rates, decay to baseline; outcomes "altered equilibrium densities, bistability, runaway growth". "Runaway growth" suggests a non-saturating K, which our hyperbolic form removes — hypothesis, not a reading. |
| `[V]` | PA, R1 | Gonzalez A, Lambert A, Ricciardi A (2008). When does ecosystem engineering cause invasion and species replacement? *Oikos* 117:1247–1257. doi:10.1111/j.0030-1299.2008.16419.x. Open PDF: redpath-staff.mcgill.ca/ricciardi/Gonzalez_etal2008.pdf. Gurney–Lawton habitat model + resident + invading engineer. **Fast-habitat QSS with linear engineering f(I)=bI gives exactly K(I) = βI/(1+γI), β = Tb/δ, γ = b(δ+ρ)/(ρδ), i.e. our form with K1 = 0** (obligate engineer). Quadratic f = cI² gives a sigmoid K and an Allee effect. The QSS reduction is not in the paper. |
| `[V]` | PA (competition form) | Castillo-Alvino H, Marvá M (2020). The competition model with Holling type II competitive response to interfering time. *J Biol Dyn* 14:222–244. doi:10.1080/17513758.2020.1742392. Exactly our a v/(1+hv) competition term, mechanistically derived; two species, constant K; bi-/tri-stability and priority effects. No 3-species or variable-K follow-up in a 16-paper forward sweep. |
| `[V]` | PA | Velasco-Hernández JX, Núñez-López M, Ramírez-Santiago G, Hernández-Rosales M (2017). On carrying-capacity construction, metapopulations and density-dependent mortality. *DCDS-B* 22:1099. doi:10.3934/dcdsb.2017054. The only paper found with a self-constructed variable K inside a competition model; K is a separate state variable, linear LV competition, two species, equilibria only. Zero forward citations. |
| `[V]` | PA (competition form) | Vera J, Marvá M, García-Garrido VJ, Escalante R (2024). The Beddington–DeAngelis competitive response: intra-species interference enhances coexistence in species competition. *Mathematics* 12(4):562. doi:10.3390/math12040562. Two species, constant K_i = r_i/a_ii; eq. (8) x_i′/x_i = r_i − a_ii x_i − a_ij x_j/(1 + a_i x_i + ã_j(x_j − 1)); Theorem 2: monotone convergence. |
| `[V]` | PA, R2 | Krakauer DC, Page KM, Erwin DH (2009). Diversity, dilemmas, and monopolies of niche construction. *Am Nat* 173:26–40. doi:10.1086/593707. Open PDF: homepages.ucl.ac.uk/~ucackmp/Publications_files/niche.pdf. Niche z is a state variable; K = c·k·z linear in z; QSS on z gives a saturating but quadratic K(x). "Monopoly" = probability m of privatising own niche; ESS c* = 1/(1+m). Three-species version only in Appendix B `[U]`. |

## Prior art on the model (PA)

| status | citation |
|---|---|
| `[V]` | Gurney WSC, Lawton JH (1996). The population dynamics of ecosystem engineers. *Oikos* 76:273–283. doi:10.2307/3546200. Compartmental habitat model. *Already cited in the report.* Structure known via Gonzalez 2008, Franco & Fontanari 2017, Watt 2021; not fetched directly. |
| `[V]` | Cuddington K, Hastings A (2004). Invasive engineers. *Ecol Model* 178:335–347. doi:10.1016/j.ecolmodel.2004.03.010. |
| `[U]` | Arroyo-Esquivel J, Hastings A (2020). Spatial dynamics and spread of ecosystem engineers: two-patch analysis. *Bull Math Biol* 82:149. doi:10.1007/s11538-020-00833-9. "Delayed Allee effect" once the environment has dynamics — **breaks R1 outside the quasi-static limit.** Closed; abstract only. |
| `[U]` | Lutscher F, Fink J, Zhu Y (2020). Pushing the boundaries: models for the spatial spread of ecosystem engineers. *Bull Math Biol* 82:138. doi:10.1007/s11538-020-00818-8. Free-boundary travelling waves, no competitors. Closed. |
| `[V]` | Basiri M, Lutscher F, Moameni A (2025). Traveling waves in a free boundary model of ecosystem engineers with strong Allee effect. *Math Biosci Eng* 22:152–184. Open PDF at aimspress.com. Cites Cuddington 2009 for obligate engineers. |
| `[V]` | Franco C, Fontanari JF (2017). The spatial dynamics of ecosystem engineers. *Math Biosci* 292:76. doi:10.1016/j.mbs.2017.08.002; arXiv:1611.09283. Lattice; states the O(u²) "obligate cooperator" condition. |
| `[V]` | Lopes AV, Fontanari JF (2019). Influence of technology on the population dynamics of ecosystem engineers. *Math Biosci Eng* 16. doi:10.3934/mbe.2019173. K ∝ usable habitats; feast/famine cycles. |
| `[V]` | Watt S, Jovanoski Z, Towers I, Saifuddin M, Sidhu H (2021). MODSIM24. https://mssanz.org.au/modsim2021/papers/F3/watt2.pdf. Restates the Gonzalez model exactly; derives b_cr = δ/T. |
| `[V]` | Safuan HM (thesis, UNSW Canberra) and Safuan, Towers, Jovanoski, Sidhu series 2012–2016 (*ANZIAM J* 53:172 doi:10.21914/anziamj.v53i0.4972; *Ecol Model* 2013; *AMC* 2016). Every K form is explicit in time or a state variable coupled bilinearly to N. **None is K(N).** |
| `[V]` | Liautaud K, Barbier M, Loreau M (2020). Ecological ecotones and engineering feedbacks. *Ecography* 43:1–12. doi:10.1111/ecog.04902. K_i(E) Gaussian in an engineered environment E; sharp fronts, alternative states. |
| `[V]` | Kylafis G, Loreau M (2008). Ecological and evolutionary consequences of niche construction for its agent. *Ecol Lett* 11:1072–1081. doi:10.1111/j.1461-0248.2008.01220.x. **Single-species** plant–soil-nutrient model. Construction term c(P) = cP/(h+P), **Michaelis–Menten in own density**, chosen to avoid "boundless autocatalytic" growth; enters as an additive nutrient flux, not a K. Full equations via Kylafis' McGill thesis. |
| `[V]` | Kylafis G, Loreau M (2011). Niche construction in the light of niche theory. *Ecol Lett* 14:82–90. doi:10.1111/j.1461-0248.2010.01551.x. Two consumers + predator; construction linear; "can either facilitate coexistence or intensify competition". |
| `[V]` | Wright JP, Gurney WSC, Jones CG (2004). Patch dynamics in a landscape modified by ecosystem engineers. *Oikos* 105:336–348. doi:10.1111/j.0030-1299.2004.12654.x. Cary Institute reprint. Fast-D QSS gives a logistic with constant K. |
| `[X]→[V]` | Yukalov VI, Yukalova EP, Sornette D (2009). Punctuated evolution due to delayed carrying capacity. *Physica D* 238:1752–1767. doi:10.1016/j.physd.2009.05.011; **arXiv:0901.4714** (round 1 had the wrong arXiv number). K(t) = A + B N(t−τ). And (2012) Modeling symbiosis by interactions through species carrying capacities. *Physica D* 241:1270–1289; arXiv:1003.2092. |
| `[V]` | Yukalov VI, Yukalova EP, Sornette D (2014). *IJBC* 24:1450021. doi:10.1142/s0218127414500217. Nonlinear delayed K(N), single species. |
| `[X]→[V]` | Gross K (2008). Positive interactions among competitors can produce species-rich communities. *Ecol Lett* 11:929–936. **doi:10.1111/j.1461-0248.2008.01204.x** (PMID 18485001). Facilitation enters **mortality**, m_i = m_i° − d_i(1 − exp{−θ_ij n_j}), in other species' density; no intraspecific density dependence at all. **Not a K-raising model; poor citation for our purpose.** Author PDF at krgross.github.io. |
| `[V]` | Hui C, Li Z, Yue D (2004). Metapopulation dynamics and distribution, and environmental heterogeneity induced by niche construction. *Ecol Model* 177:107–118. doi:10.1016/j.ecolmodel.2003.11.016. |
| `[X]→[V]` | Han X, Chen B, Hui C (2016). Symmetry breaking in cyclic competition by niche construction. *Appl Math Comput* 284:66–78. doi:10.1016/j.amc.2016.02.056. Three cyclically competing **metapopulations** (RPS); damped oscillation, periodic fluctuation, stage equilibrium. **Cellular-automaton status NOT confirmed** (round 1 called it a CA); treat as a Hui-style metapopulation model. |
| `[V]` | Meyer PS, Ausubel JH (1999). Carrying capacity: a model with logistically varying limits. *Technol Forecast Soc Change* 61:209. K varies with time, not N. |
| `[V]` | Li, Liu & Yuan (2019). *Appl Math Comput* 347. doi:10.1016/j.amc.2018.10.071. Competition with saturation effect, Turing bifurcation — but **cross-diffusion**, two species. |
| `[V]` | Cao, Geng, Li & Qu (2024). *Int J Biomath*. doi:10.1142/s1793524524500979. Three species + diffusion, variable K ∝ biotic resource, intraguild predation; proves **non-existence** of non-constant steady states. |
| `[V]` | Liu, Wang, Zhang & Li (2019). *DCDS-B*. doi:10.3934/dcdsb.2019028. State-dependent K (vegetation reduced by pika); predator–prey; saddle-node, transcritical, Hopf. |

## Modern coexistence theory framing (MCT, R1)

| status | citation |
|---|---|
| `[V]` | Ke P-J, Letten AD (2018). Coexistence theory and the frequency-dependence of priority effects. *Nat Ecol Evol* 2:1691. doi:10.1038/s41559-018-0679-z. |
| `[V]` | Grainger TN, Levine JM, Gilbert B (2019). The invasion criterion: a common currency for ecological research. *TREE* 34:925–935. *Already cited in the report.* |
| `[V]` | Grainger TN et al. (2019). Applying modern coexistence theory to priority effects. *PNAS*. doi:10.1073/pnas.1803122116. |
| `[V]` | Schreiber SJ, Yamamichi M, Strauss SY (2019). When rarity has costs. *Ecology* 100:e02664. doi:10.1002/ecy.2664. |
| `[V]` | MacDougall AS, Gilbert B, Levine JM (2009). Plant invasions and the niche. *J Ecol* 97:609–615. doi:10.1111/j.1365-2745.2009.01514.x. |
| `[V]` | Ranjan R, Koffel T, Klausmeier CA (2024). The three-species problem. *Ecol Lett* 27:e14426. *Already cited in the report.* |
| `[V]` | Kéfi S, Holmgren M, Scheffer M (2016). When can positive interactions cause alternative stable states in ecosystems? *Funct Ecol* 30:88–97. doi:10.1111/1365-2435.12601. |
| `[V]` | Bruno JF, Stachowicz JJ, Bertness MD (2003). Inclusion of facilitation into ecological theory. *TREE* 18:119–125. doi:10.1016/S0169-5347(02)00045-9. |
| `[V]` | Koffel T, Daufresne T, Klausmeier CA (2021). From competition to facilitation and mutualism: a general theory of the niche. *Ecol Monogr* 91:e01458. doi:10.1002/ecm.1458. Introduces the **"Allee niche"**: conditions where a species can persist but not invade at low density — our R1 regime by definition. |

## Empirical grounding (EMP)

### Self-facilitating carrying capacity — the two with the model's functional form
| status | citation |
|---|---|
| `[V]` | **van de Koppel J, Rietkerk M, Dankers N, Herman PMJ (2005).** Scale-dependent feedback and regular spatial patterns in young mussel beds. *Am Nat* 165:E66–E77. doi:10.1086/428362. Mussel equation ∂M/∂t = ecAM − d_M·M·k_M/(k_M+M) + D∇²M: per-capita mortality falls hyperbolically with own density. **Full sourced parameter table** (k_M = 150 g m⁻², d_M = 0.02 g g⁻¹ h⁻¹, D = 5×10⁻⁴ m² h⁻¹, V = 10 cm s⁻¹; k_M and D are estimated). Band wavelength ≈ 6 m. **Recommended named system.** |
| `[V]` | **van der Heide T et al. (2012).** Ecosystem engineering by seagrasses interacts with grazing to shape an intertidal landscape. *PLoS One* 7:e42060. doi:10.1371/journal.pone.0042060. dH/dt = s·Z/(h₂+Z) − eH + d_H∇²H: engineering acts through a **lagged sediment variable driven by a saturating function of own density** — literally refinement step 1. Two diffusivities, bistability. K = 5976 shoots m⁻², s = 0.00035 m day⁻¹; r, m, e, h₁, h₂, d_Z, d_H `[U]` (Table 1 is an image). |

### Self-facilitating carrying capacity — other systems
| status | citation |
|---|---|
| `[V]` | Johnston CA, Naiman RJ (1990). Aquatic patch creation in relation to beaver population trends. *Ecology* 71:1617–1621. doi:10.2307/1938297. |
| `[V]` | Macfarlane WW et al. (2017). BRAT. *Geomorphology* 277:72–99. |
| `[V]` | Ronnquist AL, Westbrook CJ (2021). *Sci Total Environ* (S0048969721024049). |
| `[V]` | Bertness MD, Grosholz E (1985). *Oecologia* 67:192–204. doi:10.1007/BF00384283. Hump/threshold. |
| `[V]` | de Paoli H et al. (2017). *PNAS* 114:8035. doi:10.1073/pnas.1619203114. Densities, band widths, resilience contrast. |
| `[V]` | Gascoigne JC, Beadman HA, Saurel C, Kaiser MJ (2005). *Oecologia* 145:371–381. doi:10.1007/s00442-005-0137-x. |
| `[V]` | Liu Q-X et al. (2012). *Proc R Soc B*. doi:10.1098/rspb.2012.0157. Sediment-accumulation mussel variant. |
| `[V]` | Sherratt JA (2021) PMC8384834; Bennett JJR, Sherratt JA (2019) PMC6510835. Nondimensional mussel RD analyses; m/(1+ξm) form. |
| `[V]` | Bouma TJ et al. (2009). *Oikos* 118:260–268. doi:10.1111/j.1600-0706.2008.16892.x. Spartina thresholds; numeric threshold densities `[U]` (closed). |
| `[V]` | van Hulzen JB, van Soelen J, Bouma TJ (2007). *Estuaries and Coasts* 30:3–11. doi:10.1007/BF02782962. |
| `[V]` | van der Heide T et al. (2011). *PLoS One* 6:e16504. doi:10.1371/journal.pone.0016504. |
| `[V]` | van Katwijk MM et al. (2016). *J Appl Ecol* 53:567–578. doi:10.1111/1365-2664.12562. |
| `[V]` | Schulte DM, Burke RP, Lipcius RN (2009). *Science* 325:1124–1128. doi:10.1126/science.1176516. |
| `[V]` | Taylor K et al. (2014). *Ecosystems* 17:1017. doi:10.1007/s10021-014-9771-7. |
| `[V]` | Pilliod DS, Welty JL, Arkle RS (2017). *Ecol Evol* 7:8126–8151. doi:10.1002/ece3.3414. |
| `[V]` | Eppinga MB et al. (2008). doi:10.1007/s11258-007-9309-6. |
| `[V]` | Nadell CD, Bassler BL (2011). *PNAS* 108:14181. doi:10.1073/pnas.1111147108. EPS⁺ μ = 0.139 vs EPS⁻ 0.183 h⁻¹; selection +0.144 h⁻¹ in biofilm. |
| `[V]` | Nadell CD et al. (2015). *ISME J*. PMC4511925. RbmA benefit private, RbmC shared; clean priority effect. |
| `[V]` | Yan J et al. (2017). *Nat Commun*. PMC5569112. |
| `[V]` | Hastings A et al. (2007). Ecosystem engineering in space and time. *Ecol Lett* 10:153–164. doi:10.1111/j.1461-0248.2006.00997.x. |
| `[V]` | Temmink RJM et al. (2020). Mimicry of emergent traits amplifies coastal restoration success. *Nat Commun*. PMC7376209. Establishment structures: seagrass survival 100% vs 20%, cordgrass 100% vs 0%. |

### Candidate three-species systems
| status | citation |
|---|---|
| `[V]` | Wolf EC, Cooper DJ, Hobbs NT (2007). Hydrologic regime and herbivory stabilize an alternative state in Yellowstone. *Ecol Appl*. doi:10.1890/06-2042.1. Explicit two-state ASS with beaver as engineer. |
| `[V]` | Marshall KN, Hobbs NT, Cooper DJ (2013). Stream hydrology limits recovery of riparian ecosystems after wolf reintroduction. *Proc R Soc B*. PMC3574379. Dammed+unbrowsed willow 248 cm vs control 117 cm; water table +33 cm. |
| `[V]` | Baker BW et al. (2012). *Ecosphere* 3:1–15. doi:10.1890/ES12-00058.1. Beaver persist at ≤20 elk km⁻². |
| `[U]` | Baker BW et al. (2005). *Ecol Appl* 15:110–118. doi:10.1890/03-5237. |
| `[X]` | Mott CL, Bloomquist CK, Nielsen CK (2013). Within-lodge interactions between two ecosystem engineers, beavers and muskrats. *Behaviour* 150:1325–1344. doi:10.1163/1568539X-00003097. **About lodge sharing; does NOT verify "no food competition".** Do not cite for that claim. |
| `[V]` | Wright JP, Jones CG, Flecker AS (2002). *Oecologia* 132:96–101. Beaver raise richness — benefit shared. |
| `[V]` | Bertness MD (1984). *Ecology* 65:1794–1807. Mussel–cordgrass mutualism. |
| `[V]` | Bertness MD (1991). *Ecology* 72:138–148. doi:10.2307/1938909. |
| `[V]` | Schwarz C et al. (2018). *Nat Geosci* 11:672–677. doi:10.1038/s41561-018-0180-y. Paywalled; traits and model `[U]`. |
| `[V]` | Li B et al. (2011). PMC3187781. Spartina × Scirpus RNE flips to facilitation at high N — not a stable founder-control pair. |
| `[V]` | Schmitt RJ et al. (2019). Experimental support for alternative attractors on coral reefs. *PNAS*. PMC6410839. **Order-dependent exclusion, hysteresis test.** |
| `[V]` | Adam TC et al. (2022). *Ecology*. PMC10078572. "Whichever colonizes first can suppress the competitor." |
| `[V]` | Mumby PJ, Hastings A, Edwards HJ (2007). *Nature* 450:98–101. doi:10.1038/nature06252. |
| `[V]` | van de Leemput IA et al. (2016). *Coral Reefs* 35:857. doi:10.1007/s00338-016-1439-7. |
| `[V]` | Briggs CJ et al. (2018). *PLoS One*. doi:10.1371/journal.pone.0202273. Coral bistability with parameter table. |
| `[V]` | Sura SA et al. (2025). *PLoS Comput Biol*. doi:10.1371/journal.pcbi.1013221. Holling-II herbivory, same a N/(1+hN) form. |
| `[U]` | Fung T et al. (2011). *Ecology*. doi:10.1890/10-0378.1. |
| `[V]` | Michaels TK et al. (2020). Nucleation and alternative stable states. *Ecology* 101:e03099. doi:10.1002/ecy.3099. **Critical patch radius = our Maxwell point;** Sphagnum r = 0.035 m declines, 0.07 m grows. |
| `[V]` | Angelini C et al. (2016). *Nat Commun*. PMC4992128. Cordgrass survival 64% with mussels vs 1% without. |

### Cyclic competition
| status | citation |
|---|---|
| `[V]` | Kerr B et al. (2002). *Nature* 418:171. doi:10.1038/nature00823. |
| `[V]` | Sinervo B, Lively CM (1996). *Nature* 380:240. doi:10.1038/380240a0. |
| `[V]` | Buss LW, Jackson JBC (1979). *Am Nat* 113:223. doi:10.1086/283381. |
| `[V]` | Soliveres S et al. (2015). *Ecol Lett* 18:790. PubMed 26032242. |
| `[V]` | Lankau RA, Strauss SY (2007). *Science* 317:1561. doi:10.1126/science.1147455. |

### Saturating competition
| status | citation |
|---|---|
| `[V]` | Ayala FJ, Gilpin ME, Ehrenfeld JG (1973). *Theor Popul Biol* 4:331. PubMed 4747658. |
| `[V]` | Schoener TW (1976). *Theor Popul Biol* 10:309. PubMed 1013908. |
| `[V]` | Law R, Watkinson AR (1987). *J Ecol* 75:871–886. |
| `[V]` | Inouye BD (2001). *Ecology* 82:2696. |
| `[V]` | Hart SP, Freckleton RP, Levine JM (2018). *J Ecol* 106:1902. doi:10.1111/1365-2745.12954. |
| `[V]` | Letten AD, Stouffer DB (2019). *Ecol Lett* 22:423. doi:10.1111/ele.13211. |
| `[V]` | Stouffer DB (2022). *MEE* 13:2437. doi:10.1111/2041-210X.13965. |

### Invasion thresholds / restoration
| status | citation |
|---|---|
| `[V]` | Schotanus J et al. (2020). *Restor Ecol* 28:1105. doi:10.1111/rec.13168. |
| `[V]` | Balke T et al. (2011). *MEPS* 440:1–9. doi:10.3354/meps09364. |
| `[V]` | Davis HG et al. (2004). *PNAS* 101:13804. doi:10.1073/pnas.0405230101. Pollen-limitation Allee — **not** engineering. |
| `[V]` | Taylor CM et al. (2004). *Ecology* 85:3254. doi:10.1890/03-0640. Same caveat. |

## Prior art on the mathematical results

### R2 — bifurcation cascade
| status | citation |
|---|---|
| `[V]` | Moreno-Spiegelberg P, Gomila D (2023). A model for seagrass species competition: dynamics of the symmetric case. arXiv:2304.09693. Ten regions; saddle-node, symmetric saddle-node, transcritical, pitchfork (super/subcritical by region) and Hopf all present; **no single canonical fold→pitchfork→transcritical ordering**, and a codim-2 point where T, Pitch and Hopf coincide at 2α = (1+β)². |

### R3 — bistable fronts, Maxwell point
| status | citation |
|---|---|
| `[V]` | Lewis MA, Kareiva P (1993). *Theor Popul Biol* 43:141. doi:10.1006/tpbi.1993.1007. |
| `[V]` | Keitt TH, Lewis MA, Holt RD (2001). *Am Nat* 157:203. doi:10.1086/318633. |
| `[V]` | Bel G, Hagberg A, Meron E (2012). *Theor Ecol* 5:591. doi:10.1007/s12080-011-0149-6. |
| `[V]` | Zelnik YR, Meron E (2018). *Ecol Indic* 94:544. doi:10.1016/j.ecolind.2018.01.020. |
| `[V]` | Nadin G, Strugarek M, Vauchelet N (2018). *J Math Biol* 76:1489. doi:10.1007/s00285-017-1181-y. |
| `[V]` | Gardner RA (1982). *J Diff Eq* 44:343–364. doi:10.1016/0022-0396(82)90001-8. |
| `[V]` | Kan-on Y (1995). *SIAM J Math Anal* 26:340–363. doi:10.1137/S0036141093244556. |
| `[V]` | Lequin, Biroli, Scalliet (2026). Nucleation beyond equilibrium: fronts control invasion in bistable ecosystems. arXiv:2608.05251. |
| `[V]` | Kenne C (2026). Complete characterization of the sign of the wave speed in the symmetric Lotka-Volterra system under strong competition. arXiv:2608.16845. |
| `[V]` | Chang, Chen, Wang (2023). Propagating direction in the two species Lotka–Volterra competition–diffusion system. *DCDS-B* 28:5998–6014. doi:10.3934/dcdsb.2023052. |
| `[V]` | Chang & Wu (2025) *JDE* doi:10.1016/j.jde.2025.01.019; Guo (2026) *J Biol Dyn* doi:10.1080/17513758.2026.2672784. Three-species LV bistable wavefronts. |
| `[X]` | ~~doi:10.1016/j.chaos.2022.112899~~ — resolves to a memristor neural-network paper. **Removed.** |

### R4 — Hopf in cyclic competition
| status | citation |
|---|---|
| `[V]` | May RM, Leonard WJ (1975). *SIAM J Appl Math* 29:243. *Already cited in the report.* |
| `[V]` | Gilpin ME (1975). *Am Nat* 109:51. doi:10.1086/282973. |
| `[V]` | Zeeman ML (1993). *Dyn Stab Syst* 8:189. |
| `[V]` | Hofbauer J, So JW-H (1994). *Appl Math Lett* 7:65–70. doi:10.1016/0893-9659(94)90095-7. |
| `[V]` | Jaramillo G, Mrad L, Stepien TL (2023). arXiv:2210.04342. |
| `[V]` | Mohd MH (2019). *Appl Math Comput*. doi:10.1016/j.amc.2019.02.007. |
| `[V]` | Hu, Lu, Luo (2026). Four limit cycles in class 28. arXiv:2603.24612. |
| `[V]` | Panja, Gayen, Kar & Jana (2022). *Results Control Optim*. doi:10.1016/j.rico.2022.100153. Nonlinear LV competition, three species, Hopf + transcritical. |

### R5 — Turing in pure competition
| status | citation |
|---|---|
| `[V]` | Kishimoto K, Weinberger HF (1985). *J Diff Eq* 58:15. doi:10.1016/0022-0396(85)90020-8. Two-species competition never patterns on convex domains. |
| `[V]` | **Kishimoto K (1982).** The diffusive Lotka-Volterra system with three species can have a stable non-constant equilibrium solution. *J Math Biol* 16:103–112. doi:10.1007/BF00275163. Abstract: three examples, "**one example is competitive**". Coefficient matrix only in the paywalled PDF. |
| `[V]` | Kishimoto K, Mimura M, Yoshida K (1983). *J Math Biol* 18:213–221. doi:10.1007/BF00276088. |
| `[V]` | Manna K, Volpert V, Banerjee M (2021). *Bull Math Biol* 83:52. doi:10.1007/s11538-021-00886-4. |
| `[X]→[V]` | **Piskovsky V (2025). Turing instabilities for three interacting species.** *Appl Math Lett* **159**:109269. doi:10.1016/j.aml.2024.109269; arXiv:2405.14682. (Round 1 had a paraphrased title and vol. 160.) Full inequalities transcribed in `prior-art/06_reference_verification.md`; implemented in `verification/piskovsky_check.py`. |
| `[V]` | Satnoianu RA, Menzinger M, Maini PK (2000). Turing instabilities in general systems. *J Math Biol* 41:493–512. doi:10.1007/s002850000056. s-stability; type-p classification; **p = 2 requires a negative 2×2 principal minor.** Transcribed in `06_reference_verification.md`. |
| `[V]` | Villar-Sepúlveda E, Champneys AR (2023). *J Math Biol* 86:39. doi:10.1007/s00285-023-01870-3. |
| `[V]` | Baurmann M, Gross T, Feudel U (2007). *J Theor Biol* 245:220. *Already cited in the report.* |
| `[V]` | Li, Mergia, Patidar (2026). arXiv:2604.12215. Numerical-scheme paper; droplet, banded, spiral, glider patterns. |
| `[V]` | Rietkerk M, van de Koppel J (2008). *TREE* 23:169. doi:10.1016/j.tree.2007.10.013. |
| `[V]` | Liu Q-X et al. (2013). *PNAS* 110:11905. doi:10.1073/pnas.1222339110. |
| `[V]` | Levin SA (1974). *Am Nat* 108:207. doi:10.1086/282900. Not a Turing theorem. |
