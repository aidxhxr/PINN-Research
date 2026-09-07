# Literature report: novelty and ecological plausibility of the single-engineer competition model

Prepared 7 September 2026. Proposal assessed: [new-theme/combined_report.pdf](../combined_report.pdf), dated 10 August 2026, all 25 pages. The earlier `new-theme` notes supplied search leads; their conclusions were checked against primary literature rather than adopted as evidence.

## 1. Answer to your two questions

**Has similar work been done? Yes, extensively at the level of mechanisms and outcomes. I did not identify an exact published duplicate of your complete equations and analysis.** That is a bounded search conclusion, not proof that nobody has published them. In particular, niche construction combined with three cyclic competitors and spatial patterning predates your proposal: Han, Chen and Hui studied that combination in 2016. Your competition response is also contained in a limiting case of Vera and colleagues’ 2024 model. These are central comparisons, not peripheral citations. [Han et al.](https://doi.org/10.1016/j.amc.2016.02.056), [Vera et al.](https://www.mdpi.com/2227-7390/12/4/562)

**Is it ecologically viable? It is defensible as an idealized theoretical model, with empirical support for several ingredients. It is not yet a validated model of any identified three-species community.** The four empirical works below support habitat modification, localized benefits, spatially dependent persistence, and alternative attractors. None establishes your exact carrying-capacity law, all six competition responses, or the Part-D diffusion ratios in one community.

My recommendation is to continue, but make the ecological question more precise: **when can capacity-enhancing engineering sustain an established population without allowing it to recover from rarity, and when does spatial transport change that outcome?** Parts A and B provide the clearest starting point. Parts C and D are additional dynamical possibilities obtained under different interaction parameters; they should not be presented as one observed ecological progression.

The requested four close comparisons are Gonzalez et al. (2008), Vera et al. (2024), Han et al. (2016), and Moreno-Spiegelberg and Gomila (2024). The requested four empirical studies are van der Heide et al. (2012), de Paoli et al. (2017), Nadell et al. (2015), and Schmitt et al. (2019). Supporting references appear where they resolve a specific issue.

## 2. What your proposal actually assumes

Writing $x=(u,v,w)$, the report uses

$$
\dot x_i=r_i x_i\left(1-\frac{x_i}{K_i(x_i)}\right)
-x_i\sum_{j\ne i}\frac{\alpha_{ij}x_j}{1+h_{ij}x_j},
$$

with

$$
K_u(u)=K_1+\frac{\beta u}{1+\gamma u},\qquad
K_v=K_2,\qquad K_w=K_3.
$$

The spatial extension adds $D_i\Delta x_i$, with constant diagonal diffusion and no-flux boundaries. Throughout the assessment below, $K_i,r_i,\gamma>0$, and construction and competition parameters are nonnegative.

Biologically, this is a **facultative engineer**: its baseline habitat supports growth even without engineering. Construction immediately reduces its self-crowding penalty. Competitors receive no direct engineering benefit; construction has no explicit energetic cost; habitat has no independent memory; and each interspecific competitive effect saturates separately with the density of the species causing it. These assumptions are more restrictive than the general statement that organisms modify their environment.

Two terminology corrections matter:

- The response $\alpha_{ij}x_j/(1+h_{ij}x_j)$ is a Holling-II-shaped competitive response. A full Beddington–DeAngelis denominator also contains dependence on the other participant’s density. Your response can arise as a restricted case of that broader family; calling it the full response obscures what is actually assumed. See the comparison with Vera et al. below.
- In your parameterization, the competitor density at half maximum is **$1/h_{ij}$**, and the maximum per-capita effect is **$\alpha_{ij}/h_{ij}$** when $h_{ij}>0$. The parameter $h_{ij}$ itself is not the half-saturation density or necessarily a handling time. Similarly, construction has initial slope $\beta$, asymptotic capacity increase $\beta/\gamma$, and half-saturation density $1/\gamma$.

The report explicitly says its numerical parameters were chosen for mathematical exploration rather than fitted to an ecosystem. That qualification should remain prominent.

## 3. Four closely related works, and how yours differs

### P1. Gonzalez, Lambert and Ricciardi (2008): the closest habitat-engineering ancestry

**Citation:** Andrew Gonzalez, Amaury Lambert and Anthony Ricciardi. “When does ecosystem engineering cause invasion and species replacement?” *Oikos* 117:1247–1257. [Primary paper, full PDF](https://redpath-staff.mcgill.ca/ricciardi/Gonzalez_etal2008.pdf). DOI as printed in the paper: `10.1111/j.2008.0030-1299.16419.x`.

**Published overlap.** A resident and an invading engineer occupy different habitat states. Engineer abundance controls habitat conversion; engineered habitat decays and degraded habitat recovers. Population carrying capacities are habitat quantities. The authors compare linear and quadratic engineering, obtaining different establishment thresholds and replacement outcomes. This already establishes the ecological link between engineering, invasion, coexistence, and exclusion.

**Our mathematical comparison, derived from their habitat equations.** Let $T$ be total habitat, $H_1,H_2$ the native and engineered habitat, $\rho$ recovery, and $\delta$ decay. For engineering $f(u)=bu$, setting their habitat derivatives to zero gives

$$
H_2^{\mathrm{qs}}(u)
=\frac{\rho Tbu}{(\delta+\rho)bu+\rho\delta}
=\frac{(Tb/\delta)u}{1+[b(\delta+\rho)/(\rho\delta)]u}.
$$

This is the same hyperbolic shape as your capacity increment. **The reduction is an inference here; it is not a published novelty claim on the authors’ behalf.** It also requires habitat adjustment to be fast enough to justify eliminating its dynamics.

**Difference that matters.** Your baseline $K_1>0$, two non-engineering competitors, explicit competition losses, and spatial population diffusion are substantial structural changes. Their full model also changes the resident’s available habitat, so the entire system is not recovered merely by setting your $K_1=0$.

**Implication for positioning.** Present the hyperbolic feedback as a tractable reduction inspired by established habitat-engineering theory. The new question is what its facultative, three-competitor version implies. Do not claim the density-dependent habitat feedback itself is new.

### P2. Vera, Marvá, García-Garrido and Escalante (2024): your competition response is a special case

**Citation:** María Carmen Vera, Marcos Marvá, Víctor José García-Garrido and René Escalante. “The Beddington–DeAngelis Competitive Response: Intra-Species Interference Enhances Coexistence in Species Competition.” *Mathematics* 12(4):562. [Primary article](https://www.mdpi.com/2227-7390/12/4/562), DOI `10.3390/math12040562`.

**Published overlap.** This two-species model introduces density-dependent interference into competition and analyzes coexistence and multiple stable outcomes. In their Eq. (9), the loss in the first equation has denominator

$$
1+a_1K_1u_1+\widetilde a_2(K_2u_2-1).
$$

**Our algebraic comparison.** In their restriction $a_1=0$, for $0\le\widetilde a_2<1$, that loss becomes

$$
\frac{r_1c_{12}u_1u_2}{1-\widetilde a_2+\widetilde a_2K_2u_2}
=u_1\frac{A_{12}u_2}{1+H_{12}u_2},
$$

where $A_{12}=r_1c_{12}/(1-\widetilde a_2)$ and $H_{12}=\widetilde a_2K_2/(1-\widetilde a_2)$. This is your pairwise loss form after parameter relabeling; the second equation has the analogous reduction. It does not identify their full model with yours.

**Difference that matters.** Your distinctive addition is one density-dependent carrying capacity in a three-species system, followed by invasion and spatial analyses. The saturating competition ingredient is established.

**Important correction to the previous notes.** Castillo-Alvino and Marvá’s 2020 paper is related, but its Eq. (5) contains $u_1u_2/(1+c_1u_1)$ in the first species’ equation. Your denominator there depends on $u_2$. The difference changes which density weakens the competitive effect. The earlier claim of exact identity with the 2020 model is incorrect. [Castillo-Alvino and Marvá, primary article](https://www.tandfonline.com/doi/full/10.1080/17513758.2020.1742392)

### P3. Han, Chen and Hui (2016): three cyclic competitors plus niche construction already exists

**Citation:** Xiaozhuo Han, Baoying Chen and Cang Hui. “Symmetry breaking in cyclic competition by niche construction.” *Applied Mathematics and Computation* 284:66–78. [Primary publisher record and article preview](https://www.sciencedirect.com/science/article/abs/pii/S0096300316301758), DOI `10.1016/j.amc.2016.02.056`.

**Published overlap.** The authors study three cyclically competing metapopulations with niche construction using cellular automata. Their abstract reports damped and persistent oscillations, changes in occupancy as construction strengthens, and spatial structure. Consequently, the broad combination of engineering, three competitors, cyclic dominance, and pattern formation was already investigated a decade before your report.

**Difference that matters.** Your report specifies continuous population equations and a particular instantaneous $K_u(u)$, enabling explicit invasion rates and equilibrium formulas. Its PDE calculations distinguish a moving boundary between alternative states from a finite-wavelength instability of a homogeneous equilibrium. Those distinctions should carry the comparison; having three species or producing attractive spatial plots cannot.

**Evidence limit.** I verified the publisher’s abstract and introduction and the author’s publication record, but did not obtain the full 13-page article. Therefore I do not claim to have excluded every equation-level or theorem-level overlap with Han et al. The abstract itself explicitly identifies cellular automata, resolving that uncertainty in the previous notes.

**Implication for positioning.** This should be a leading citation for Parts C and D. Phrase your contribution as an analytically tractable continuous model and a precisely classified instability, while leaving priority of the broad ecological idea with the earlier literature.

### P4. Moreno-Spiegelberg and Gomila (2024): a close precedent for the bifurcation catalogue

**Citation:** Pablo Moreno-Spiegelberg and Damià Gomila. “A model for seagrass species competition: dynamics of the symmetric case.” *Mathematical Modelling of Natural Phenomena* 19:2. [Published paper](https://doi.org/10.1051/mmnp/2023033); [accessible author preprint](https://arxiv.org/pdf/2304.09693). The journal publication is **2024**; the preprint is from 2023.

**Published overlap.** The paper combines density-dependent facilitation and competition for two seagrasses. Its reduced local reaction is

$$
\dot n_1=n_1[-\omega+n_1+\alpha n_2-(n_1+\beta n_2)^2],
$$

with the exchanged equation for $n_2$. Tables 1–2 give equilibria and saddle-node, transcritical, pitchfork, and Hopf thresholds; the parameter plane contains ten distinct bifurcation regimes. Spatial terms also include nonlinear clonal spread. Their $\beta$ is not your construction parameter.

**Difference that matters.** Your model singles out one engineer among three otherwise competing populations, uses a bounded hyperbolic capacity, and keeps spatial diffusion linear and diagonal. Their polynomial feedback can directly improve low-density per-capita growth; your capacity law does not do that in monoculture, as shown below.

**Implication for positioning.** A fold–pitchfork–transcritical catalogue is not an unprecedented ecological result, and exact thresholds alone are not a distinguishing research program. Your particular branch structure may still be new, but its significance must come from the ecological distinction it reveals. Their paper does not establish the exact sequence and parameter thresholds of your Part A.

### Comparison at a glance

| Work | Main overlap | Defensible distinction in your report |
|---|---|---|
| Gonzalez et al. 2008 | Population–habitat feedback and engineering-dependent establishment | Facultative capacity law, explicit competition, three populations |
| Vera et al. 2024 | Pairwise saturating competition included as a special case | One engineered capacity and three-species spatial dynamics |
| Han et al. 2016 | Engineering, three cyclic competitors, oscillations and spatial structure | Continuous specified equations, invasion formulas, instability classification |
| Moreno-Spiegelberg & Gomila 2024 | Facilitation–competition bifurcations | A single engineer, rational capacity feedback, different local density dependence |

### Other prior work that constrains a novelty claim

These are supporting comparisons, not additional members of the requested four.

**Explicit constructed carrying capacity is established.** Velasco-Hernández et al. (2017), *DCDS-B* 22:1099–1110, use an independently evolving constructed niche and two competing patch occupants. The full PDF confirms capacities $K_i=\kappa_i z$ and construction contributions from both occupants. Describing the paper merely as two ordinary Lotka–Volterra equations misses its patch structure. [Primary paper](https://www.aimsciences.org/article/doi/10.3934/dcdsb.2017054)

**Engineer–environment bistability is established.** Cuddington, Wilson and Hastings (2009), *American Naturalist* 173:488–498, already report altered equilibria, bistability, and runaway growth from engineer–environment feedback. I could verify its abstract, not the equations. Its exact relationship to your reduction remains an access gap; the previous assertion that no open copy exists anywhere is stronger than the evidence warrants. [Publisher abstract](https://www.journals.uchicago.edu/doi/10.1086/597216)

**Patterning in a competitive three-species diffusion model is not new.** Kishimoto (1982) reports a competitive example with stable nonconstant equilibria. This rules out claiming the first such possibility. I did not retrieve its coefficient matrix. General instability criteria are available from Satnoianu et al. (2000) and Piskovsky (2025). [Kishimoto’s institutional article record](https://tus.elsevierpure.com/en/publications/the-diffusive-lotka-volterra-system-with-three-species-can-have-a/), [Satnoianu et al., full paper](https://people.maths.ox.ac.uk/maini/PKM%20publications/122.pdf), [Piskovsky](https://doi.org/10.1016/j.aml.2024.109269)

**Recent spatial engineering theory also treats shared resources.** Ardichvili et al. (2025), *Oikos* e11025, investigate when resource transport changes competition into facilitation between patches. That is an important alternative to putting diffusion only on organism densities. [Primary paper](https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2025-05/010093367.pdf)

## 4. Four empirical works supporting ecological plausibility

These papers provide **evidence for specified mechanisms**, not proof that your complete model describes nature. Distinguishing those levels is necessary to answer your second question honestly.

### E1. van der Heide et al. (2012): organisms measurably engineer their own habitat

**Citation:** Tjisse van der Heide et al. “Ecosystem Engineering by Seagrasses Interacts with Grazing to Shape an Intertidal Landscape.” *PLOS ONE* 7(8):e42060. [Full primary article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0042060), DOI `10.1371/journal.pone.0042060`.

**Evidence.** Field surveys, seagrass removal, and a spatial model link seagrass to sediment elevation and reduced grazing exposure. Removal plots lost $17.6\pm7.3$ mm of elevation, whereas controls showed a small nonsignificant increase. Their model uses

$$
\partial_t H=s\frac{Z}{Z+h_2}-eH+d_H\Delta H.
$$

**Supported part of yours.** A population can modify a physical habitat state in a density-dependent way and receive a persistence benefit. A saturating engineering flux is a published modeling choice.

**Unsupported part.** Habitat elevation affects grazing, rather than entering as your instantaneous logistic capacity. Waterfowl are consumers, not two competing species; neither the exact $K_u(u)$ law nor your Turing mechanism is tested.

**Parameter-table check.** The previously unread Table 1 gives $r=0.4\,\mathrm{day}^{-1}$, $h_2=1500$ shoots m$^{-2}$, $s=0.00035$ m day$^{-1}$, and $H_{\max}=0.2$ m, with $e=s/H_{\max}$. Thus the modeled habitat relaxation time is about **571 days**. This is a calculation from their parameters, not a measured decay time. Seasonal growth complicates comparison with $1/r$, but the table does not support assuming fast habitat equilibration. Its spatial coefficients were prescribed relative to rates, not measured physical diffusivities transferable to your PDE.

### E2. de Paoli et al. (2017): spatial organization can cause greater persistence

**Citation:** Hélène de Paoli, Tjisse van der Heide, Aniek van den Berg, Brian R. Silliman, Peter M. J. Herman and Johan van de Koppel. “Behavioral self-organization underlies the resilience of a coastal ecosystem.” *PNAS* 114(30):8035–8040. [Published paper in the authors’ institutional repository](https://pure.rug.nl/ws/portalfiles/portal/65312977/8035.full.pdf), DOI `10.1073/pnas.1619203114`.

**Evidence.** Field manipulations arranged blue mussels into clusters, bands, both, or an unpatterned control. Patterned beds persisted better, with small-scale aggregation protecting mussels against disturbance and larger-scale structure helping reaggregation. Behavioral experiments connected the persistence differences to movement and clustering.

**Supported part of yours.** Parts A–B ask an ecologically meaningful question: persistence can depend on how organisms are established and arranged, beyond a homogeneous equilibrium calculation. Spatial structure is therefore worth studying as an ecological mechanism.

**Unsupported part.** These experiments do not identify a Maxwell point or your three-species Turing instability. Active aggregation is also not equivalent to constant Fickian diffusion: the latter smooths density differences by itself. A model that uses this study as evidence must explain how its movement approximation represents the relevant scale.

**Practical use.** This is strong motivation for measuring patch survival and responses to rearrangement. It is not a source from which your six competition parameters can be copied. The published data and analysis are archived, but their response variables do not parameterize the full proposed triad.

### E3. Nadell et al. (2015): constructed benefits can remain local to their producers

**Citation:** Carey D. Nadell, Knut Drescher, Ned S. Wingreen and Bonnie L. Bassler. “Extracellular matrix structure governs invasion resistance in bacterial biofilms.” *The ISME Journal* 9:1700–1709. [Full primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC4511925/), DOI `10.1038/ismej.2014.246`.

**Evidence.** Experiments with *Vibrio cholerae* linked resistance to biofilm invasion to the matrix protein RbmA. Its protection localized to producing cell lineages. Removing RbmA allowed substantially more invasion; complementation restored resistance. Established biofilms could also exclude later-arriving cells genetically identical to residents. Another matrix component, RbmC, was shared more broadly.

**Supported part of yours.** This supports the possibility of localized construction benefits and a strong resident advantage. It gives biological motivation for studying the limiting case where competitors receive little direct engineering benefit.

**Unsupported part.** Producer lineages are not equivalent to one entire species benefiting uniformly. The study does not establish your hyperbolic capacity, a complete three-species competitive network, or its diffusion coefficients. Protection acts through adhesion and access, which may belong in establishment or loss terms.

**Practical use.** A constructed microbial community is a plausible future testing platform, but this particular experiment is not already your triad.

### E4. Schmitt et al. (2019): alternative attractors occur under the same environmental conditions

**Citation:** Russell J. Schmitt, Sally J. Holbrook, Samantha L. Davis, Andrew J. Brooks and Thomas C. Adam. “Experimental support for alternative attractors on coral reefs.” *PNAS* 116(10):4372–4381. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6410839/); [authors’ research-project record](https://mcr.lternet.edu/publications/experimental-support-alternative-attractors-coral-reefs), DOI `10.1073/pnas.1812412116`.

**Evidence.** Field experiments in Moorea varied herbivory and initial benthic states and tested recovery following disturbance. They found hysteresis and support for persistent alternative states in lagoon habitat under ambient conditions. The fore reef differed because ambient herbivory was outside that range.

**Supported part of yours.** The Part-A distinction between an attainable positive equilibrium and recovery from a depleted state has ecological relevance. Initial conditions and the size of disturbances can determine which long-term state occurs under the same external conditions.

**Unsupported part.** This is not evidence for your particular three-state equations or exact branch sequence. Herbivory is an essential external driver, and the experiments do not measure the frozen-engineer Jacobian submatrix required for Part D. “Alternative attractors on reefs” must not be translated into “the proposed $v,w$ competition pair has been verified.”

**Practical use.** Its strongest contribution is experimental design: compare multiple initial states under the same controlled driver, rather than treating a single recovery trajectory as proof of bistability.

### What these four works support together

| Component of your proposal | Evidence assessment | Best match above | Remaining gap |
|---|---|---|---|
| Organisms improve habitat conditions relevant to their own persistence | Strong mechanism-level support | E1 | Fit the pathway to population growth |
| Constructed benefit can be retained locally | Strong in a specific biological context | E3 | Establish its extent for the chosen populations |
| Initial state affects ecological outcome | Strong support | E3, E4 | Test the proposed invasion thresholds |
| Spatial arrangement affects persistence | Strong support | E2 | Identify the correct movement and feedback processes |
| Exact $K_1+\beta u/(1+\gamma u)$ demographic law | Plausible assumption; not established by these studies | No direct fit | Compare against mortality/growth alternatives |
| Three populations linked by all six proposed loss responses | Not established | None | Species-specific competition measurements |
| The report’s particular Hopf and Turing regimes | Mathematical examples, without a matched empirical system | None | Calibrated parameters and mechanism-specific tests |

## 5. Corrections that change the ecological interpretation

The calculations in this section follow directly from the proposed equations. They are analytical checks, not new simulations. Numerical examples explicitly attributed to the PDF or existing diagnostics were not rerun for this literature report.

### 5.1 Raising capacity does not produce positive self-density dependence here

For the engineer alone, define per-capita growth

$$
g(u)=r_1\left(1-\frac{u}{K_u(u)}\right).
$$

Then

$$
g'(u)=-r_1\frac{K_u(u)-uK_u'(u)}{K_u(u)^2},\qquad
K_u(u)-uK_u'(u)=K_1+\frac{\beta\gamma u^2}{(1+\gamma u)^2}>0.
$$

Therefore **$g'(u)<0$ for every $u\ge0$**. The same conclusion holds while holding competitor densities fixed, because their per-capita losses do not depend on $u$.

Increasing construction strength improves growth at a given positive density relative to the unengineered case, but increasing density within a fixed parameter setting still reduces per-capita growth. Those statements are compatible. The model represents **relief from self-crowding**, not a demographic Allee effect in an isolated engineer population.

Consequently, the Part-A establishment threshold arises through the coupled community feedback: sufficiently abundant engineers can suppress competitors, which in turn releases engineers from competition. An explanation that attributes the threshold solely to positive self-density dependence would be wrong.

This also prevents identifying your law with the mussel model of van de Koppel et al. (2005). Their per-capita mortality term is $d_Mk_M/(k_M+M)$; at fixed food concentration, it produces increasing per-capita growth with $M$. Your isolated $g'(u)$ has the opposite sign. The hyperbolas are related mathematical shapes embedded in different demographic equations. They are not interchangeable parameterizations. [van de Koppel et al., primary-paper PDF](https://files01.core.ac.uk/download/pdf/29282504.pdf)

### 5.2 The invasion result is correct, but “never protected coexistence” needs a scope

For a fixed resident equilibrium $(0,v^*,w^*)$,

$$
\lambda_u=r_1-\frac{\alpha_{12}v^*}{1+h_{12}v^*}
-\frac{\alpha_{13}w^*}{1+h_{13}w^*}.
$$

Because the $u=0$ subsystem does not depend on $\beta,\gamma$, construction cannot change this invasion exponent while all other parameters remain fixed. The intrinsic expansion makes the structural reason particularly clear:

$$
r_1u\left(1-\frac{u}{K_u(u)}\right)
=r_1u-\frac{r_1}{K_1}u^2+\frac{r_1\beta}{K_1^2}u^3+O(u^4).
$$

Construction strength first enters total growth at **cubic order**, or quadratic order in per-capita growth.

For the PDF’s Part-A coefficients, $\lambda_u\approx-0.0043086$, so the engineer-free equilibrium remains resistant to engineer invasion throughout that construction sweep. This excludes global recovery of all three species from arbitrary positive initial states in that family.

It does **not** establish that the whole model class can never support protected coexistence: other growth or competition coefficients can make the exponent positive. Nor does it mean the interior equilibrium is unstable or unable to recover from small perturbations. Local stability and recovery following near-extinction are different properties.

For three species, a full coexistence assessment must consider the relevant boundary equilibria or other resident invariant sets; a two-species mutual-invasibility slogan is not a complete general test. [Ranjan, Koffel and Klausmeier’s three-species analysis](https://doi.org/10.1111/ele.14426)

There is a further singularity in the comparison with P1. If $K_1=0$, then for $u>0$, $u/K_u(u)=(1+\gamma u)/\beta$, and the limiting intrinsic growth rate becomes $r_1(1-1/\beta)$. Thus the obligate limit can depend on construction at rarity. One cannot transfer your $K_1>0$ invasion conclusion to that limit by continuity without checking it.

### 5.3 A habitat-memory variable alone does not overturn the invasion result

A transparent extension that recovers the original model is

$$
\tau\partial_t S=u-S,\qquad
K_u(S)=K_1+\frac{\beta S}{1+\gamma S}.
$$

For sufficiently fast relaxation, $S\approx u$. Finite $\tau$ introduces memory and may alter finite-density trajectories and bifurcations. This is a **proposed extension**, not a result already demonstrated in your PDF.

However, if $S$ affects only this positive carrying capacity, the linearized engineer growth at rarity still equals $r_1$ minus resident competition. The effect through $S$ is multiplied by $u^2$. Therefore the previous refinement suggestion that a habitat lag will automatically create a delayed Allee effect is not justified for this model structure.

To model an engineer that cannot survive in unmodified habitat, environmental quality must affect a low-density demographic rate—for example survival or establishment—or one must analyze an obligate formulation explicitly. That is a biological modeling decision, not simply an extra differential equation.

### 5.4 The Turing interpretation needs a different mechanism

At a positive equilibrium, the engineer’s diagonal Jacobian entry is $J_{uu}=u^*g'(u^*)<0$. The other diagonal entries are also negative. Thus no single population supplies positive local self-activation at that equilibrium.

For a stationary diffusion-driven instability in this three-variable setting, inspect the two-dimensional principal submatrices. In particular,

$$
M_{vw}=J_{vv}J_{ww}-J_{vw}J_{wv}.
$$

A negative value means that, **with engineer density held fixed at $u^*$**, the linearized $v,w$ subsystem is unstable even though the full community can be stable. General theory explains why a sufficiently fast complementary component can enable spatial instability in this situation. [Satnoianu, Menzinger and Maini](https://people.maths.ox.ac.uk/maini/PKM%20publications/122.pdf)

The existing repository diagnostics report $M_{vw}\approx-0.00671$ for Part D at $\beta=1.5$, while the other two principal minors are positive. They classify the example as stationary Turing, not Turing–Hopf. This agrees with the PDF’s real growing eigenvalue, but **does not support its assertion that a near-Hopf complex pair is a necessary precursor**. The relevant explanation is the unstable subsystem and differential transport. [Existing diagnostic output](../verification/piskovsky_outputs.txt), [Piskovsky’s classification](https://doi.org/10.1016/j.aml.2024.109269)

The ecological interpretation should remain local: a competing engineer can stabilize the joint response of two competitors in a well-mixed setting, while rapid redistribution of the engineer permits those competitors to segregate spatially. A negative minor does **not** by itself prove that physically removing the engineer produces a globally bistable two-species community. Removal changes both the equilibrium and effective growth conditions.

The Part-D numerical example uses $D_u:D_v:D_w\approx14860:40:1$; this is its chosen ratio, not a proved requirement for the entire model class. These are organism-density diffusion coefficients. Fast water, nutrient, or signal transport cannot be substituted for $D_u$ without adding the transported environmental state to the model. None of E1–E4 verifies these three population diffusivities. This is presently the weakest ecological link in Part D.

### 5.5 A stationary front is not the same thing as a critical patch size

Part B brackets a zero-speed planar front at $\beta_M\in(4.00,4.15)$. Calling this a Maxwell point is acceptable if it is defined operationally by front speed; the system has not been shown to possess an energy function allowing an equal-area rule.

A critical patch radius is a different quantity. In a simple scalar isotropic approximation, the outward speed of a circular patch behaves as $c(R)\approx c_0-D/R$. When $c_0>0$, this yields a finite critical radius; at $c_0=0$, curvature still causes finite circles to contract. That formula is an illustration, not a derived law for your three-component PDE.

Michaels, Eppinga and Bever (2020) provide a useful ecological framework connecting local feedbacks with spatial establishment thresholds. But their critical-patch concept cannot be identified with your construction parameter $\beta_M$: the two even have different meanings and units. [“A nucleation framework for transition between alternate states: short-circuiting barriers to ecosystem recovery”](https://doi.org/10.1002/ecy.3099)

### 5.6 Several statements in the regime catalogue need qualification

**Monopoly is not global exclusion of alternative attractors.** The Part-A engineer-free state remains locally stable even in Regime IV. It therefore retains a local basin, including nearby positive initial conditions. The accurate statement is that the engineer-only equilibrium becomes invasion-resistant, while an alternative engineer-free state persists.

**Only some thresholds are closed form.** The report derives $\beta_2=247/48$, while the fold and symmetry-breaking thresholds are located numerically. Describe exactly which results are analytical.

**A symmetry-breaking eigenvector does not complete a pitchfork classification.** The claimed steep transition from a positive symmetric state to competitor extinction deserves local branch continuation and a nondegeneracy check. An ordinary local pitchfork initially creates nearby asymmetric states; an abrupt jump to a distant boundary state could involve a subcritical branch or additional structure. The PDF alone does not settle that issue.

**Numerical persistence at very low density is not demographic security.** Part C’s deep oscillatory troughs remain positive in deterministic equations, but a population-count interpretation would need a scale and stochastic extinction assessment. No ecological claim about viable minimum population size follows from those curves alone.

## 6. Which ecological interpretation should you use?

The most defensible present description is:

> A theoretical model of a facultative ecosystem engineer whose modification reduces self-crowding, while all three populations retain negative direct interspecific effects. Construction can change finite-density persistence without changing the engineer’s invasion exponent in a fixed resident community.

This describes the equations without claiming an empirical fit. It also makes the main prediction falsifiable: if engineered habitat measurably changes the engineer’s low-density survival or establishment at fixed competitor densities, the capacity-only formulation is missing a pathway.

For a **physical habitat** interpretation, seagrass or mussel systems provide good motivation, but habitat state, grazing or resource processes, and spatial scale must be represented honestly. For a **localized-benefit** interpretation, microbial matrix construction is a useful analogue, but the natural units may be lineages, patches, or functional groups rather than three named species. These are candidate directions, not established matches to the complete model.

The original examples should be revised. Mussel–cordgrass facilitation cannot stand as an example of an entirely competitive triad: Bertness’s field manipulations support a mutualistic association. The beaver–muskrat lodge study does not establish the proposed saturating food-competition response; failure to find that evidence also does not prove the animals never compete. [Bertness, primary paper](https://www.csun.edu/~msteele/classes/marine_ecology/readings/Bertness%201984.pdf), [Mott and colleagues’ author publication record](https://mottlab.weebly.com/publications.html)

Mathematically, the nonnegative state space is invariant and the positive capacity ceiling bounds engineer growth: $u$ is bounded above by the larger of its initial value and $K_1+\beta/\gamma$; corresponding logistic bounds hold for $v,w$. These are useful consistency properties. They do not establish that nature follows the equations.

## 7. What would make this a stronger research contribution?

The following are recommendations for subsequent work, not analyses claimed to be complete here.

1. **Make persistence versus invasion the central ecological distinction.** Prove the rarity result at its correct scope and quantify how the coexistence basin changes with construction. Include both engineer-poor and engineer-rich initial states. This gives Parts A–B a connected scientific purpose.

2. **Establish whether construction causes the claimed spatial effect.** Compare $\beta=0$, the engineered model, and a constant-capacity control matched to $K_u(u^*)$. The last control separates increased capacity from density-dependent feedback. A Turing example at positive $\beta$ alone does not establish that engineering is necessary for the instability.

3. **Test a mechanism-faithful habitat extension.** Use the relaxation formulation above first, because its fast limit is exactly the original law. If ecological evidence instead identifies effects on mortality or recruitment, put them there and acknowledge that the invasion prediction may change. Do not add a delay simply to obtain another bifurcation.

4. **Test modest benefits to the competitors.** For example, use $K_v=K_2+\varepsilon_v B(S)$, $K_w=K_3+\varepsilon_w B(S)$, with $B(S)=\beta S/(1+\gamma S)$. The current model is the $\varepsilon_v=\varepsilon_w=0$ case. Determine which conclusions survive small positive values. Resource transport can also change interaction signs, as emphasized by the recent Oikos work. [Ardichvili et al.](https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2025-05/010093367.pdf)

5. **Resolve the movement interpretation before emphasizing Turing patterns.** Identify what spreads, how fast, and on which spatial scale. If the engineer is sessile and the transported entity is sediment or nutrient, use the appropriate environmental variable. Quantify whether instability occupies a robust parameter region under plausible transport rates.

6. **Use density manipulations to distinguish functional forms.** Vary focal and competitor density independently. Your first-species loss saturates with the competitor’s density, whereas the 2020 interference response weakens with focal density. These alternatives can fit a narrow trajectory similarly while predicting different invasion and coexistence behavior. Compare separate versus shared denominators when two competitors act together.

7. **Validate branches and predictions, not just final trajectories.** Follow unstable equilibria where needed, resolve the symmetry-breaking transition, and distinguish finite-time persistence from asymptotic outcomes. For empirical testing, predict outcomes from initial conditions or interventions not used to fit parameters.

Construction cost is another relevant sensitivity if $\beta$ is interpreted as investment or an evolving trait. The present fixed-$r_1$ sweep is legitimate as a capacity-effect experiment, but cannot establish that stronger construction is affordable or favored by selection. An evolutionary claim would require an explicit cost and invasion-fitness framework.

A defensible novelty statement for the present stage is:

> We analyze a three-species competition model with a bounded, density-dependent carrying capacity for one facultative engineer. We distinguish construction-dependent persistence from construction-independent invasion growth and examine how interaction asymmetry and spatial transport affect alternative states and instabilities.

Avoid claiming the first niche-construction competition model, the first three-species engineering pattern model, or a new general near-Hopf route to stationary Turing instability. The value of the work will depend on the precision and robustness of its ecological distinction, beyond combining familiar ingredients.

## 8. Research scope, evidence limits, and corrections to retain

The search covered combinations of ecosystem engineering, niche construction, constructed or density-dependent carrying capacity, nonlinear competition, cyclic competition, bifurcations, fronts, and Turing instability. It included title/DOI checks, publisher records, primary papers in author or institutional repositories, and targeted 2025–2026 searches. This was a focused critical review, not a preregistered systematic review or a complete citation-database census.

| Source | Material checked for this report |
|---|---|
| Your combined report | Full 25-page PDF |
| Gonzalez et al. 2008 | Full primary PDF; habitat equations and invasion cases |
| Vera et al. 2024 | Indexed primary article text, Eq. (9), and the reduced-interference case |
| Han et al. 2016 | Publisher abstract and introduction; author bibliography; full paper unavailable |
| Moreno-Spiegelberg & Gomila | Author preprint equations and bifurcation tables; journal year verified separately |
| van der Heide et al. 2012 | Full article and PDF; Table 1 visually inspected |
| de Paoli et al. 2017 | Author manuscript and published PDF; findings taken from the published version |
| Nadell et al. 2015 | Full primary article, including experimental controls and localization results |
| Schmitt et al. 2019 | Indexed primary article text and discussion; authors’ project record; linked project PDF unavailable |
| Velasco-Hernández et al. 2017 | Primary PDF, including explicit niche and patch equations |
| Cuddington et al. 2009 | Publisher abstract only |
| Kishimoto 1982 | Publisher/institutional abstract; example coefficients unavailable |
| Satnoianu et al. 2000; Piskovsky 2025 | Primary instability theory and available article/preprint material |
| Ardichvili et al. 2025 | Primary paper’s resource-transport framework |

The most consequential corrections to the old notes are: the 2020 competition model is not equation-identical to yours; the 2024 model supplies the more direct special-case relationship; increasing $K_u(u)$ does not create a monoculture Allee effect; a habitat lag confined to $K$ does not automatically change the invasion exponent; the mussel mortality law is not equivalent to your capacity law; and a critical patch radius is not a Maxwell parameter. These corrections narrow the claims while making the proposal easier to defend.

**Decision:** proceed as a theoretical ecology project with a focused persistence-versus-invasion question. Treat ecological mechanisms as supported at the component level, exact demographic laws as hypotheses to test, and the complete three-species spatial system as unvalidated. The four comparisons above should anchor the novelty discussion; the four empirical papers should motivate specific assumptions with their limitations stated alongside them.
