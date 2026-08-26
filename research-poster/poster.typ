// =====================================================================
//  Physics-Informed Neural Networks for a WNT-RA-HOX Model of
//  Colorectal Cancer Stemness
//  48in x 36in landscape research poster - Typst
//  v3: project walkthrough, built from ready assets.
// =====================================================================

#let W = 48in
#let H = 36in
#let MARGIN = 1.2in
#let LIVE = W - 2 * MARGIN
#let GUTTER = 0.65in
#let COL = (LIVE - 3 * GUTTER) / 4

// ---- palette --------------------------------------------------------
#let garnet = rgb("#9C2745")
#let paper = rgb("#F7F6F3")
#let ink = rgb("#152A3A")
#let ink2 = rgb("#5A6B78")
#let rule-c = rgb("#DFDCD5")
#let blue = rgb("#1f77b4")
#let orange = rgb("#ff7f0e")
#let grey = rgb("#7f7f7f")
#let red = rgb("#cc3333")

#let SERIF = "Source Serif 4"
#let SANS = "Inter"
#let MONO = "IBM Plex Mono"

#set page(width: W, height: H, margin: 0pt, fill: paper)
#set text(font: SANS, fill: ink, size: 26pt, lang: "en")
#set par(leading: 0.62em, spacing: 0.88em, justify: false)

// ---- building blocks -------------------------------------------------
#let colhead(n, kicker, claim) = block(width: 100%, below: 14pt)[
  #grid(columns: (56pt, 1fr), column-gutter: 12pt, align: (center, left),
    box(circle(radius: 25pt, fill: garnet)[
      #align(center + horizon)[#text(font: SERIF, size: 29pt, weight: 700,
        fill: white)[#n]]
    ]),
    [
      #text(font: MONO, size: 18pt, weight: 600, fill: garnet, tracking: 1.5pt)[
        #upper(kicker)]
      #v(3pt)
      #text(font: SERIF, size: 38pt, weight: 600, fill: ink)[#claim]
    ]
  )
  #v(7pt)
  #line(length: 100%, stroke: 2.2pt + garnet)
]

#let subhead(t) = block(above: 14pt, below: 6pt)[
  #text(font: SERIF, size: 29pt, weight: 600, fill: ink)[#t]
]

// the one-line implication that rides under every figure (SIAM convention)
#let implic(t) = block(above: 6pt, below: 4pt, width: 100%)[
  #text(size: 22pt, weight: 500, fill: ink)[#t]
]

#let fine(t) = text(size: 19pt, fill: ink2)[#t]
#let mono(t) = text(font: MONO, size: 23pt)[#t]

#let card(body, fill: white, stroke-c: none) = block(
  width: 100%, inset: 15pt, radius: 4pt, fill: fill,
  stroke: if stroke-c == none { none } else { 1pt + stroke-c },
)[#body]

#let tile(value, label, c: garnet) = block(width: 100%)[
  #text(font: SANS, size: 46pt, weight: 700, fill: c)[#value]
  #v(-6pt)
  #text(size: 20pt, fill: ink2)[#label]
]

#let fig(path, w: 100%) = block(width: 100%, above: 8pt, below: 2pt)[
  #align(center)[#image(path, width: w)]
]

// numbered step used in the nondimensionalization chain and the roadmap
#let step(n, title, body) = block(width: 100%, below: 8pt)[
  #grid(columns: (auto, 1fr), column-gutter: 10pt, align: (left + top, left + top),
    text(font: MONO, size: 20pt, weight: 700, fill: garnet)[#n],
    [
      #text(size: 24pt, weight: 600, fill: ink)[#title]
      #v(2pt)
      #text(size: 21pt, fill: ink2)[#body]
    ])
]

// =====================================================================
//  TITLE BAND  (no kicker line, no affiliation line - authors only)
// =====================================================================
#place(top + left, block(width: W, height: 4.58in, fill: garnet)[
  #pad(x: MARGIN, y: 0.40in)[
    #text(font: SERIF, size: 76pt, weight: 700, fill: white)[
      Physics-Informed Neural Networks and Neural-Mechanistic Hybrids for a
      WNT–RA–HOX Model of Colorectal Cancer Stemness
    ]
    #v(14pt)
    #grid(columns: (1fr, auto), column-gutter: 0.8in,
      align: (left + horizon, right + horizon),
      [
        #text(size: 27pt, fill: rgb("#F4DEE3"))[
          From a nondimensionalized mechanistic 7-ODE model to sparse-data
          forward solves, Bayesian identifiability, and learned terms.
        ]
      ],
      [
        #text(size: 30pt, weight: 600, fill: white)[
          Amirkhan Aidarkhan · Pascal Kataboh]
      ]
    )
  ]
])

// =====================================================================
//  ROADMAP STRIP - names the four columns
// =====================================================================
#place(top + left, dy: 4.58in, block(width: W, height: 2.42in, fill: white)[
  #pad(x: MARGIN, y: 0.24in)[
    #grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 0.45in,
      step("01", "The model",
        [Seven coupled ODEs for the WNT–RA–HOX axis, nondimensionalized to
         *36 parameters* across four disease regimes.]),
      step("02", "The PINN",
        [A Fourier-feature network solves the system forward from *40
         observations*, then inverts it for the 36 parameters.]),
      step("03", "Bayesian inference",
        [HMC turns the recovery ceiling into a diagnosis: which parameters
         the data can *never* pin down, and why.]),
      step("04", "Neural-mechanistic hybrid",
        [Replace one mechanistic term with a neural network, and the
         identifiability problem returns in a new form.]),
    )
  ]
])

// =====================================================================
//  FOUR COLUMNS
// =====================================================================
#let CTOP = 7.15in
#let CH = H - CTOP - 0.95in
#let colx(i) = MARGIN + i * (COL + GUTTER)

// ---------------- COLUMN 1 : THE MODEL --------------------------------
#place(top + left, dx: colx(0), dy: CTOP, block(width: COL, height: CH)[
  #colhead("1", "the model", "Seven equations, nondimensionalized to 36 parameters.")

  #text(size: 24pt)[
    In the intestinal crypt, WNT signalling drives proliferation, retinoic acid
    drives differentiation, and the HOX genes arbitrate between them. Loss of
    APC, the canonical adenoma-to-carcinoma driver, breaks that balance.
  ]

  #fig("assets/ready/schema.svg", w: 92%)
  #implic[Green activates, red inhibits, dashed mediates. Seven species, one
    constitutive WNT drive, and a vitamin-A input.]

  #subhead[Nondimensionalization]

  #step("01", "Scale each species by its characteristic value",
    [taken to be its initial condition, and scale time by the reference
     β-catenin degradation rate $d_B = 1$ hr⁻¹:])

  #align(center)[#text(size: 25pt)[
    $B = B_0 b, quad P = P_0 p, quad dots, quad tau = d_B t$
  ]]
  #v(2pt)
  #fine[β-catenin is WNT's direct target and drives most other species, so its
    turnover sets the network's reference timescale.]

  #v(6pt)
  #step("02", "Every equation collapses to one form",
    [each loss term normalizes to $-x$, and the prefactor is a *pure timescale
     ratio*:])

  #align(center)[#text(size: 27pt)[
    $epsilon_X (d x) / (d tau) = "production" - x$
  ]]

  #v(6pt)
  #step("03", "The parameter count falls to 36",
    [Nondimensionalization changed the units and the count, not the model.
     The system is *stiff*: $epsilon_R = 0.40$, $epsilon_M = 0.60$.])

  #v(6pt)
  #card(fill: white)[
    #text(size: 22pt, weight: 600)[The scientific readout: the stemness index]
    #v(4pt)
    #align(center)[#text(size: 27pt)[
      $S(tau) = (b (1 + alpha_13 h_13)) / ((1 + p)(1 + alpha_5 h_5))$
    ]]
    #v(3pt)
    #fine[Large when proliferative species dominate, small when differentiating
      species do. Not a state variable, but the aggregate the model exists to
      predict.]
  ]

  #v(6pt)
  #card(fill: rgb("#EFF4F8"))[
    #text(size: 22pt, weight: 600)[Four regimes, two parameters]
    #v(3pt)
    #text(size: 21pt, fill: ink2)[Normal → Early Adenoma → Advanced Adenoma →
      Severe APC Loss differ *only* in the WNT drive #mono[W] (0.8 → 2.0) and
      APC functionality $theta_P$ (1.0 → 0.25).]
  ]
])

// ---------------- COLUMN 2 : THE PINN ---------------------------------
#place(top + left, dx: colx(1), dy: CTOP, block(width: COL, height: CH)[
  #colhead("2", "the pinn", "One network solves it from 40 points, and inverts it.")

  #grid(columns: (1fr, 1fr), column-gutter: 14pt,
    [
      #text(size: 21pt, weight: 600, fill: garnet)[FORWARD · θ fixed]
      #image("assets/ready/forward_arch.svg", width: 100%)
    ],
    [
      #text(size: 21pt, weight: 600, fill: garnet)[INVERSE · θ trainable]
      #image("assets/ready/inverse_arch.svg", width: 100%)
    ])
  #implic[The network proposes a trajectory; automatic differentiation feeds it
    back through the ODEs. Three losses: data, initial condition, physics.
    Freeing θ turns the solver into a parameter estimator.]

  #text(size: 23pt)[
    A Fourier-feature time embedding (16 modes, σ = 4) into a 256-wide, 4-deep
    GELU network, 207,879 weights. Without it, spectral bias renders the
    circadian forcing as straight lines.
  ]

  #fig("assets/ready/forward_fit.png", w: 98%)
  #implic[Trained on *40 sparse observations* plus the physics residual, the
    PINN tracks the Radau reference through the ATRA window in every regime.]

  #v(2pt)
  #grid(columns: (1fr, 1fr), column-gutter: 12pt,
    tile("1.06%", "dense-label baseline, grand mean rel-L2", c: blue),
    tile("2.41%", "from 40 observations, the honest solve", c: orange),
  )

  #subhead[Running it backwards]

  #fig("assets/ready/inv_recovery_top.png")
  #implic[True (grey) against recovered (red), three of the best-converging
    parameters. Recovery is ill-posed at 36: the PINN-specific ceiling is ~8/36,
    from gradient starvation and *autodiff derivative bias*.]

  #card(fill: rgb("#EFF4F8"))[
    #text(size: 22pt, weight: 600)[A derivative-free integral residual breaks
      half that ceiling]
    #v(3pt)
    #text(size: 21pt, fill: ink2)[Enforcing the ODE in integrated form uses
      network *values* only, never the biased derivative. Recovery goes
      #text(fill: ink, weight: 600)[10/4/5/7 → 17/16/10/7], a same-conditions
      total of #text(fill: ink, weight: 600)[37 → 50] of 144. Severe APC Loss
      does not move; what remains is structural, not numerical.]
  ]
])

// ---------------- COLUMN 3 : BAYESIAN ---------------------------------
#place(top + left, dx: colx(2), dy: CTOP, block(width: COL, height: CH)[
  #colhead("3", "bayesian inference", "A posterior says what the data can never pin down.")

  #text(size: 24pt)[
    A point estimate cannot say *why* a parameter failed. We run HMC over the 36
    parameters with the state networks frozen: the Bayesian twin of the point
    refine stage, and derivative-free, because the likelihood reuses the same
    integral residual.
  ]

  #fig("assets/ready/bayes_marginals.png")
  #implic[#mono[W] stays tight in every regime. $theta_P$ reverts toward
    its prior as the WNT drive rises: the posterior-width statement of the
    high-WNT wall.]

  #fig("assets/fig_posterior.svg", w: 92%)
  #implic[Unprompted, the sampler rides a curved valley: the *product*
    $delta_(P 1)(1 - theta_P)$ is 4.7× better constrained than
    $delta_(P 1)$ alone. At the healthy truth $theta_P = 1$ the product
    vanishes, so $delta_(P 1)$ is structurally free.]

  #fig("assets/fig_fim.svg")
  #implic[The Fisher information agrees from a third direction: conditioning
    degrades with WNT drive, to 3.7 × 10¹⁹ in Severe APC Loss.]

  #card(fill: rgb("#F6EDF0"), stroke-c: garnet)[
    #text(font: SERIF, size: 28pt, weight: 600, fill: garnet)[
      The first run said 36/36. It was an artifact.]
    #v(4pt)
    #text(size: 22pt)[The physics likelihood summed ~210k residual terms against
      a tiny σ, producing a pinpoint, biased posterior. Below: truths far
      outside their own 95% intervals, coverage 13/36.]
  ]

  #fig("assets/ready/bayes_miscal_top.png")
  #implic[Honest recovery is *19/20/11/8*, not 36/36. The run still fails its
    own effective-sample-size and coverage gates, so we claim the *geometry* and
    the diagnosis, never the marginal widths.]
])

// ---------------- COLUMN 4 : HYBRID -----------------------------------
#place(top + left, dx: colx(3), dy: CTOP, block(width: COL, height: CH)[
  #colhead("4", "neural-mechanistic hybrid", "Replace a term with a network and identifiability returns.")

  #card(fill: white)[
    #align(center)[#text(size: 30pt)[
      $epsilon dot(x) = #text(fill: garnet)[$a$] + f_"NN" (u) - x$
    ]]
    #v(2pt)
    #align(center)[#fine[the anchor $f_"NN" (0) = 0$ is meant to stop the network
      absorbing a constant out of the basal-production parameter $a$]]
  ]

  #text(size: 23pt)[
    It does not work: across *five* parameterisations, including the
    monotone-and-bounded one the UDE literature recommends, the basal parameter
    comes out #text(fill: red, weight: 600)[14–203%] wrong.
  ]

  #fig("assets/fig_support.svg", w: 82%)
  #implic[Because across all 10 conditions and all 4 regimes, *no regulator ever
    approaches zero*. The anchor is a constraint asserted at a point the data
    never visit.]

  #subhead[The fix is an experiment, not an architecture]

  #text(size: 22pt)[
    A dose–response with the *condition count held fixed at eleven* at every
    dose, so "more data helps" is ruled out by construction.
  ]
  #card(fill: white)[
    #grid(columns: (1fr, 1fr), column-gutter: 12pt,
      [
        #text(font: MONO, size: 20pt, fill: ink2)[misses the anchor]
        #v(2pt)
        #text(size: 40pt, weight: 700, fill: orange)[2.2% / 1.5%]
      ],
      [
        #text(font: MONO, size: 20pt, fill: ink2)[reaches it · one more siRNA]
        #v(2pt)
        #text(size: 40pt, weight: 700, fill: blue)[0.0% / 0.0%]
      ])
    #v(5pt)
    #fine[Same edge, same equation, same parameter, same seeds, same budget.
      Normal / Severe APC Loss. The two arms differ by *one siRNA*.]
  ]

  #fig("assets/fig_dose_compact.svg", w: 96%)
  #implic[Against the *anchor ratio* the arms collapse onto one curve; against
    dose they do not. The anchor, not the intervention, is the governing
    variable.]

  #v(4pt)
  #block(width: 100%, inset: 15pt, radius: 4pt, fill: garnet)[
    #text(font: SERIF, size: 28pt, weight: 700, fill: white)[Takeaways]
    #v(6pt)
    #set text(size: 21pt, fill: rgb("#F7E9ED"))
    #grid(columns: (auto, 1fr), column-gutter: 9pt, row-gutter: 6pt,
      text(font: MONO, weight: 700, fill: white)[1],
      [Nondimensionalization plus Fourier features makes a stiff, multiscale
       7-ODE system solvable by a PINN from sparse data.],
      text(font: MONO, weight: 700, fill: white)[2],
      [Inverse recovery is limited by *information*, not architecture, and a
       posterior says exactly which directions are unidentifiable.],
      text(font: MONO, weight: 700, fill: white)[3],
      [A constraint on a learned term carries information only where the data
       have support; the remedy is experiment design, computable in advance.],
    )
  ]

  #v(5pt)
  #card(fill: white, stroke-c: rule-c)[
    #text(size: 21pt, weight: 600)[What these numbers are not]
    #v(3pt)
    #fine[Hybrid numbers come from an equation-local screen fed exact states,
      an upper bound though both arms are fitted identically so the differences
      are fair. Restart noise ±0.5–0.9 pp; single seed, so no error bars. Not
      shown: an information-matched control (no gain in 8 of 8 cells) and a
      prospective test (95.6% → 0.0%). 4 of 6 pre-registered predictions failed
      and are reported as failures.]
  ]
])

// =====================================================================
//  FOOTER  (references, compact; QR space reserved at right)
// =====================================================================
#place(bottom + left, block(width: W, height: 0.95in, fill: white)[
  #pad(x: MARGIN, y: 0.13in)[
    #grid(columns: (1fr, 2.6in), column-gutter: 0.5in,
      align: (left + horizon, right + horizon),
      [
        #set text(size: 16.5pt, fill: ink2)
        #set par(leading: 0.52em)
        #text(weight: 600, fill: ink)[References] ·
        Raissi, Perdikaris & Karniadakis, _J. Comput. Phys._ 378:686 (2019);
        Karniadakis et al., _Nat. Rev. Phys._ 3:422 (2021);
        Tancik et al., NeurIPS (2020);
        Yang, Meng & Karniadakis, arXiv:2003.06097 (2020);
        Jung & Choi, arXiv:2210.11737 (2022);
        Raue et al., _Bioinformatics_ 25:1923 (2009);
        Engl et al., _Inverse Problems_ 25:123014 (2009);
        Rackauckas et al., arXiv:2001.04385 (2020);
        Philipps, Schmid & Hasenauer, _npj Syst. Biol. Appl._ 11:101 (2025);
        Loman & Baker, arXiv:2510.14140 (2025);
        Wang & Hill, _IEEE Trans. Neural Netw._ 17(1):130 (2006);
        Plate, Martensen & Sager, arXiv:2408.07143 (2024);
        Velioglu et al., _Comput. Chem. Eng._ (2025);
        Faure et al., _Nat. Commun._ 14:4213 (2023).
      ],
      [#text(size: 16pt, fill: rgb("#C9C6C0"))[]]
    )
  ]
])
