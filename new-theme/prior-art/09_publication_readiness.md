# Is the niche-construction model ready for publication?

Assessment date: 7 September 2026.

Source model: [combined_report.pdf](../combined_report.pdf), all 25 pages. This assessment also inspected the existing verification outputs and the files within new-theme. It builds on the [earlier literature review](08_literature_and_ecological_plausibility.md).

This is a research and manuscript-readiness assessment, not a new simulation study or an independent reproduction of every result. No experiments were launched and the original model and report were not changed.

## 1. Direct answer

**No: I would not submit the present report unchanged. Yes: there is a defensible route toward a theoretical ecology paper, but acceptance cannot be predicted or guaranteed.**

The equations are not disqualified simply because they are simplified or have not been fitted to a real community. The immediate problems are narrower: some biological explanations do not match the equations, some conclusions are broader than the evidence, and the work needs a sharper contribution beyond a collection of familiar mathematical behaviours.

There is no single universal certificate showing that a model meets “all ecological modeling standards.” Expectations depend on the model's purpose and the journal. The TRACE framework explicitly distinguishes organized documentation from a guarantee of adequate model quality; it does not supply universal pass/fail thresholds. [Grimm et al. (2014), especially the discussion](https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/TRACE.ModFramework.GrimmEtAl2014.pdf)

Two different publication claims need to be separated:

| Intended claim | What must be supported | Current position |
|---|---|---|
| A theoretical mechanism: under these assumptions, engineering can produce specified community outcomes | Correct equations and reasoning, justified assumptions, useful new insight, and analysis appropriate to the breadth of the claim | Plausible route, but substantial corrections and targeted checks remain |
| A predictive model of a particular real community | The above plus matched species and scales, defensible parameter estimates, uncertainty analysis, and tests against observations not used to build or fit the model | Not established by the current report |

My recommendation is the first route. Do not force the equations onto beavers, mussels, or bacteria just to attach species names to an abstract model.

## 2. What the research says reviewers should care about

These are evidence-based expectations, not a claim that every journal mandates an identical checklist.

**Ecological understanding must be the outcome.** The current scope of Theoretical Ecology welcomes mathematical and computational work, but excludes mathematical analyses that do not advance ecological understanding. Producing another bifurcation diagram is therefore not sufficient by itself. [Official journal scope](https://link.springer.com/journal/12080/aims-and-scope)

**A theoretical paper does not always need a fitted dataset.** Servedio and colleagues explain how simplified models can test whether a proposed biological explanation is logically possible. However, the assumptions central to that explanation must actually represent it, and the effects of simplifying assumptions should be made clear. This is methodological guidance from evolutionary biology, relevant to the present proof-of-concept use of ecological equations; it is not an exemption from biological reasoning. [Servedio et al. (2014)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002017)

**Correct simulation and correct ecology are separate questions.** TRACE covers the research question, model description, inputs, assumptions, implementation, output checks, sensitivity, and independent corroboration. For this theoretical project, those categories are useful as an audit structure; empirical prediction tests apply if an empirical prediction claim is made. [Grimm et al. (2014)](https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/TRACE.ModFramework.GrimmEtAl2014.pdf)

**The choice of equation can matter as much as its parameter values.** Fussmann and Blasius showed that very similar-looking response curves can yield different ecological dynamics. That makes alternative functional forms important when the precise mechanism is unknown. This is a general warning, not proof that our own results will disappear under alternatives. [Fussmann and Blasius (2005)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1629049/)

## 3. Audit of the current report

The judgments below are my assessment of the supplied material, not editorial decisions.

| Question a reviewer can reasonably ask | Evidence already present | Assessment and remaining work |
|---|---|---|
| Is there an ecological question? | Engineering, coexistence, establishment, and spatial spread are identified | Present, but the narrative is spread across separately selected parameter families. Choose a central question. |
| Are the equations fully specified? | Population equations, diffusion, boundaries, and parameter sets are provided | Substantially present. Correct parameter meanings and document units or nondimensionalization. |
| Are populations mathematically sensible? | Growth is density limited and losses vanish with the affected population | Nonnegative, bounded solutions follow under the stated positive-parameter assumptions. Add the argument and check numerical positivity. |
| Do the equations represent the stated biological mechanism? | One population receives a density-dependent capacity increase | Only partly. It represents reduced self-crowding, not positive self-density dependence in monoculture. |
| Is the numerical implementation checked? | Step-size spot checks, equilibrium residuals, diffusion conservation, a homogenization check, and 1D/2D simulations are reported | Meaningful work exists. Document systematic convergence for headline thresholds, front speeds, and patterns. |
| Are the changes between regimes established? | Eigenvalues, exact special-case formulas, and Hopf coefficient calculations are reported | Several results have substantial numerical support. The fold-location method and pitchfork/exclusion account need stronger branch tracking. |
| Are the conclusions robust? | A 71-by-55 construction-parameter grid, multiple initial states, and different interaction structures are studied | Partial. This does not establish robustness to growth rates, competition strengths, symmetry breaking, or alternative response laws. |
| Does engineering cause the reported effects? | Outcomes change across construction sweeps | Separate increased capacity from feedback, and ask whether spatial instability also exists without construction. |
| Is the spatial interpretation biologically justified? | Population diffusion produces fronts and patterns | No matched movement scales. The Turing calculation is an existence example at a chosen large diffusion contrast. |
| Is the contribution clearly new? | Some earlier engineering and competition work is acknowledged | Incomplete. Important close comparisons were missing, and the claimed new Turing route is not justified. |
| Can another researcher reproduce the whole paper? | Verification scripts and text outputs exist in new-theme/verification | Useful foundation, but an end-to-end package reproducing all figures was not identified within new-theme. Full reproducibility remains unaudited. |
| Can it quantitatively predict real species? | Empirical analogues were discussed in the earlier review | Not currently. Those studies support ingredients, not the complete equations and parameters. |

The report deserves credit for its existing checks. It is not accurate to say that it has no sensitivity analysis, no numerical verification, or only a deliberately seeded 1D pattern. It has a two-parameter sweep and a noise-initialized 2D calculation. The question is whether those checks support each particular claim.

## 4. The most important issues to resolve

### 4.1 Correct the interpretation before adding complexity

The engineer's carrying capacity and isolated growth per individual are

$$
K_u(u)=K_1+\frac{\beta u}{1+\gamma u},
\qquad
g(u)=r_1\left(1-\frac{u}{K_u(u)}\right).
$$

Direct differentiation gives

$$
g'(u)=-r_1\frac{K_1+\beta\gamma u^2/(1+\gamma u)^2}{K_u(u)^2}<0.
$$

In plain language: construction makes crowding less harmful than it otherwise would be, but adding more engineers still reduces growth per individual. The isolated population does not acquire an Allee effect, meaning improved per-individual growth as a small population becomes larger.

The community can nevertheless have an establishment threshold. An established engineer can suppress competitors, and that suppression can help it survive. This is a feedback through the other populations, not a positive self-density effect in isolation.

The competition term should also be described as Holling-II-shaped, or as a restricted case of a broader interference response, not the full Beddington–DeAngelis response. Its half-saturation density is $1/h_{ij}$, not $h_{ij}$. The earlier review gives the equation-level comparison with [Vera et al. (2024)](https://www.mdpi.com/2227-7390/12/4/562).

These are corrections to what the equations mean. They do not require replacing the equations.

### 4.2 Separate a structural result from a fragile sign choice

At a fixed engineer-free resident equilibrium, the engineer's growth rate when rare is

$$
\lambda_u=r_1-\frac{\alpha_{12}v^*}{1+h_{12}v^*}
              -\frac{\alpha_{13}w^*}{1+h_{13}w^*}.
$$

Its independence from construction strength is a genuine structural property of the positive-baseline, capacity-only formulation, with other parameters fixed.

However, its negative sign is not a property of every parameter choice. Part A gives

$$
\lambda_u=-0.00430862889510264.
$$

The resident $v,w$ equilibrium does not depend on $r_1$. Therefore changing only $r_1$ from $1.000$ to $1.005$ adds $0.005$ to the invasion rate:

$$
\lambda_u^{\mathrm{new}}=+0.00069137110489736.
$$

This is a **0.5% change in the engineer's growth rate**, calculated directly from the formula and existing output, not from a newly run experiment.

It proves that “the engineer can never recover from rarity” needs qualification. It does not prove full three-species persistence after that change: the other boundary states and resulting dynamics would also need examination.

For publication, map where the sign changes and ask what happens to the establishment threshold and coexistence region on each side. A model is allowed to have sensitive transitions; the paper must identify and explain them rather than describe them as universal.

The independence of invasion from construction is also close to being built into the chosen equation. It is a useful starting result, but should not be the entire novelty claim. The richer contribution could be the relationship between invasion, establishment basins, and spatial spread.

### 4.3 Test the assumptions most responsible for the conclusions

The model assumes that construction benefits only the engineer directly, habitat adjustment has no separate timescale, construction strength can change without an explicit growth cost, competitive losses saturate separately, and some cases have exact species symmetry.

None automatically makes a theoretical model unacceptable. Nor must every paper add every omitted process. Their importance depends on the claim.

A focused comparison set would be:

1. No construction.
2. Construction as currently formulated.
3. A constant-capacity control matched to the engineered capacity at the equilibrium of interest.
4. Small departures from competitor symmetry and modest changes in growth and competition parameters.
5. One biologically motivated alternative to the most consequential assumption, such as partial benefit to competitors or a finite habitat response time.

The matched-capacity control asks whether an effect comes from a larger population ceiling or from its dependence on engineer density. It preserves that particular capacity value, not the full dynamics or all branches.

Adding a habitat state only inside a positive carrying capacity does not automatically change linear growth at rarity. A different prediction requires a pathway affecting low-density survival, establishment, or growth. A cost term becomes essential for a claim about the evolution or affordability of construction, but is not automatically mandatory for a fixed-trait ecological example.

Compare alternative response laws on a stated density range with a sensible matching rule. Arbitrarily changing both their shapes and strengths would make the comparison hard to interpret.

### 4.4 Strengthen the exact numerical claims being made

The additional work should target unresolved claims:

- Track stable and unstable equilibria through the reported fold. A long-time trajectory crossing a density cutoff can locate an establishment boundary for that initial condition; it does not by itself locate a saddle-node exactly.
- Track the asymmetric branches around the proposed pitchfork. An antisymmetric eigenvector identifies a symmetry-breaking direction, but does not establish the complete branch structure or immediate competitor extinction.
- Check how outcome labels change with final time and the “present” cutoff of $0.02$. A finite positive density threshold is a classification choice, not literal extinction.
- Present mesh and time-step convergence for front speed, the stationary-front estimate, dominant wavelength, and saturated pattern amplitude. Report errors in relevant outputs rather than only whether plots look similar.
- Check the actual spatial modes allowed by the domain and discrete operator. The report already checks an allowed 1D mode; retain and extend that practice.
- Use a few independent noise seeds and domain sizes to establish which pattern properties persist. Identical spot locations are not required; wavelength ranges or amplitude statistics may be more meaningful.
- Keep exact formulas, numerical estimates, and finite-time observations clearly separated.
- Do not describe Regime IV as global engineer monopoly. The engineer-free equilibrium remains locally stable for the Part-A parameters, so it retains a basin containing nearby positive initial conditions.

Handwritten NumPy code is not itself a problem. The issue is whether its implementation and accuracy can be checked. There is also no need to invent statistical significance tests for an exact deterministic calculation.

### 4.5 Do not make the current Turing example the strongest ecological claim

Part D uses population-diffusion coefficients in the ratio

$$
D_u:D_v:D_w\approx14860:40:1.
$$

This does not prove the example is impossible in nature, and it is not a demonstrated minimum ratio for the model. However, no matched ecological measurement supports it in the supplied material.

If $u$ is engineer population density, fast nutrient or water movement cannot be cited as a measurement of $D_u$. A transported environmental resource belongs in its own state variable if it is central to the mechanism.

The report also says that a nearly unstable oscillatory reaction state is a necessary precursor to its stationary spatial instability. That general explanation is unsupported. Existing diagnostics instead identify a negative two-species principal minor, with the engineer held fixed at the interior equilibrium. General theory distinguishes stationary from oscillatory spatial instabilities without the report's claimed universal near-Hopf requirement. [Satnoianu et al. (2000)](https://people.maths.ox.ac.uk/maini/PKM%20publications/122.pdf), [Piskovsky, author preprint of the 2025 paper](https://arxiv.org/pdf/2405.14682)

Retain Part D as a qualified existence example if it helps answer the main question. To make it central, explain the correct mechanism, determine whether construction is necessary, and map where it occurs. Do not present the chosen point as representative without evidence.

### 4.6 Establish what is actually new

Three close precedents matter immediately:

- [Gonzalez et al. (2008)](https://redpath-staff.mcgill.ca/ricciardi/Gonzalez_etal2008.pdf): construction, habitat-dependent population support, establishment, and replacement.
- [Han et al. (2016)](https://doi.org/10.1016/j.amc.2016.02.056): three cyclic competitors with niche construction and spatial dynamics.
- [Moreno-Spiegelberg and Gomila (2024)](https://doi.org/10.1051/mmnp/2023033): facilitation–competition models with multiple bifurcation regimes.

Different equations do not by themselves establish a valuable new contribution. Explain what readers learn that those papers did not already establish.

A possible focus is:

> When habitat modification only reduces crowding, how do growth from rarity, the population size needed for establishment, and the ability of an established patch to spread differ?

This is a proposed framing, not a claim that all three questions have already been answered generally. A minimum establishment population must be specified relative to competitor densities and, in space, patch geometry; it is not automatically one universal number.

Prior-art access gaps remain, particularly the full Han article and the full Cuddington et al. (2009) engineer-feedback paper. Resolve these before making strong priority claims.

## 5. What publication does not automatically require

We do not need to model an entire ecosystem, add every environmental process, or collect new field data simply to qualify as theoretical research.

We also do not need every result to survive every parameter change. A transition that disappears outside clearly stated conditions can still be informative.

Instead, match the claim to the evidence:

- “This behaviour can occur” needs a correct, reproducible example and a reason it is scientifically useful.
- “This mechanism usually produces this behaviour” needs evidence across a justified region of assumptions and parameters.
- “This explains this real community” needs a matched biological interpretation and empirical tests.
- “This predicts what a restoration intervention will do” needs stronger predictive checks and uncertainty assessment.

The seagrass, mussel, and biofilm studies reviewed previously justify selected mechanisms. Combining three analogies from different systems does not establish that all assumptions occur together in one community.

## 6. A practical package before submission

This is my proposed readiness target, not a journal-mandated experimental protocol.

### Main paper

1. One clear ecological question.
2. A corrected model explanation, including units or scaling and the assumptions defining its intended use.
3. Analytical results with their domains of validity.
4. Establishment and coexistence results across a justified parameter region.
5. Spatial-front results linked to the same question, if spatial analysis remains central.
6. Direct comparison with the nearest literature and an explicit statement of new insight.
7. Limitations and observations or interventions that could test the mechanism.

Consider moving the separate Hopf and Turing catalogues to supporting material unless they become necessary to the central argument. This is a narrative recommendation, not a request to delete completed work.

### Supporting material and reproducibility

- Complete figure-generation scripts, numerical settings, parameter files, and saved outputs.
- Software versions, random seeds where relevant, and instructions that work in a clean environment.
- Tests for basic limits, equilibrium residuals, numerical positivity, and headline convergence claims.
- Documentation of the parameter search that found Part D.
- A versioned release suitable for archiving, with an accurate data/code availability statement.

Organized projects, documented computational steps, and version control are established good practice. [Wilson et al. (2017)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)

Theoretical Ecology currently requires a data availability statement for original research and strongly encourages sharing supporting data. Its instructions also address transparency of custom code. An archived, runnable package is my proposed way to support reproducibility convincingly. [Current submission guidelines](https://link.springer.com/journal/12080/submission-guidelines)

Because AI assistance has gone beyond copy editing in this project, document its role accurately under the chosen journal's policy. The same guidelines require documentation of LLM use beyond the stated copy-editing exception. Human authors must verify the analysis, citations, and final claims. Do not list an AI system as an author.

## 7. Journal direction

**Theoretical Ecology** is a sensible scope benchmark if the paper centers on a new ecological explanation. Fit is conditional on that contribution, not an acceptance forecast. [Scope](https://link.springer.com/journal/12080/aims-and-scope)

**Journal of Mathematical Biology** is another benchmark if the work develops rigorous biological insight or genuinely new mathematical ideas relevant to biology. Adding routine stability calculations would not by itself satisfy that ambition. [Scope](https://link.springer.com/journal/285/aims-and-scope)

**Ecology** explicitly accepts theoretical and analytical approaches, emphasizing new ecological concepts and generalizable insights. This shows that lack of a new empirical dataset is not an automatic exclusion, not that this draft currently reaches its contribution threshold. [Scope](https://esajournals.onlinelibrary.wiley.com/hub/journal/19399170/aims-and-scope/read-full-aims-and-scope)

These are scope comparisons, not a ranking by acceptance probability. No acceptance odds or timelines are assigned. Current Elsevier author pages attempted during this search were inaccessible, so no requirements from those pages are asserted here.

## 8. Short explanation for your professor

> The model has a reasonable foundation for theoretical ecology, but I do not think the current report is ready to submit unchanged. We do not necessarily need to fit a real ecosystem to publish a theory paper. We do need to show a clear new ecological insight, correct some explanations, and test whether the main conclusions depend too strongly on the chosen numbers. For example, the model's negative growth-from-rarity result changes sign after only a 0.5% increase in the engineer's growth rate. The most useful next step is to focus on how environmental improvement affects establishment and spread, then check that explanation carefully. The empirical examples support parts of the idea, but they do not validate the full model.

## 9. Units and basic mathematical consistency

These checks follow directly from the equations rather than a new numerical experiment.

If population $x_i$ is measured in density units $N_i$, time in $T$, and distance in $L$, a consistent dimensional reading is:

| Quantity | Units |
|---|---|
| $x_i,K_i$ | $N_i$ |
| $r_i$ | $T^{-1}$ |
| $\alpha_{ij}$ | $(T N_j)^{-1}$ |
| $h_{ij}$ | $N_j^{-1}$ |
| $\gamma$ | $N_u^{-1}$ |
| $\beta$ | Dimensionless, since capacity and engineer density have the same units |
| $D_i$ | $L^2/T$ |

If the equations are nondimensional instead, show the density, time, and space scales used. Simulation time is not automatically days, and numerical domain length is not automatically metres.

With positive baseline capacities and nonnegative construction and competition parameters, each population derivative vanishes when its own density is zero. The continuous ODE therefore preserves nonnegative densities. Also,

$$
K_u(u)\le K_1+\beta/\gamma
$$

for $\gamma>0$, and competitive losses cannot increase growth. Logistic comparison bounds the engineer by the larger of its starting density and this ceiling; corresponding constant-capacity bounds apply to the other populations. Suitable comparison arguments extend these bounds to nonnegative, bounded initial fields under the stated no-flux diffusion model.

These are consistency properties, not empirical validation. They also do not guarantee a finite-step numerical scheme will preserve positivity. Pure diffusion should conserve population mass under no-flux boundaries; the full model need not conserve total population because birth and death are present.

**Final assessment:** develop the project further, but treat correction of the claims, a focused novelty argument, targeted robustness checks, and a reproducible package as prerequisites to a submission I would recommend. There is no basis to promise publication or certify the model as a validated real ecosystem.
