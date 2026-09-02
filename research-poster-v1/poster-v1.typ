// =====================================================================
//  Physics-Informed Neural Networks and Neural-Mechanistic Hybrids for a
//  WNT-RA-HOX Model of Colorectal Cancer Stemness
//  48in x 36in landscape research poster - Typst
//  v1 (poster-v1): abstract + equations, less prose, vector figures.
// =====================================================================

#let W = 48in
#let H = 36in
#let MARGIN = 1.2in
#let LIVE = W - 2 * MARGIN
#let GUTTER = 0.7in
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
#set text(font: SANS, fill: ink, size: 27pt, lang: "en")
#set par(leading: 0.68em, spacing: 1.02em, justify: false)

// ---- building blocks -------------------------------------------------
#let colhead(n, claim) = block(width: 100%, below: 16pt)[
  #grid(columns: (58pt, 1fr), column-gutter: 14pt, align: (center, horizon),
    box(circle(radius: 26pt, fill: garnet)[
      #align(center + horizon)[#text(font: SERIF, size: 30pt, weight: 700,
        fill: white)[#n]]
    ]),
    text(font: SERIF, size: 39pt, weight: 600, fill: ink)[#claim],
  )
  #v(9pt)
  #line(length: 100%, stroke: 2.4pt + garnet)
]

#let subhead(t) = block(above: 30pt, below: 16pt)[
  #text(font: SERIF, size: 30pt, weight: 600, fill: garnet)[#t]
]

// the one-line implication that rides under every figure
#let implic(t) = block(above: 9pt, below: 5pt, width: 100%)[
  #text(size: 25pt, fill: ink)[#t]
]

#let fine(t) = text(size: 23pt, fill: ink2)[#t]
#let mono(t) = text(font: MONO, size: 27pt)[#t]

#let card(body, fill: white, stroke-c: none) = block(
  width: 100%, inset: 16pt, radius: 4pt, fill: fill,
  stroke: if stroke-c == none { none } else { 1pt + stroke-c },
)[#body]

#let eq(body, size: 28pt) = block(width: 100%, above: 18pt, below: 18pt)[
  #set text(size: size, top-edge: "bounds", bottom-edge: "bounds")
  #align(center)[#body]
]

#let tile(value, label, c: garnet) = block(width: 100%)[
  #text(font: SANS, size: 48pt, weight: 700, fill: c)[#value]
  #v(-6pt)
  #text(size: 23pt, fill: ink2)[#label]
]

#let fig(path, w: 100%) = block(width: 100%, above: 10pt, below: 2pt)[
  #align(center)[#image(path, width: w)]
]

// =====================================================================
//  TITLE BAND
// =====================================================================
#place(top + left, block(width: W, height: 3.75in, fill: garnet)[
  #pad(x: MARGIN, y: 0.40in)[
    #text(font: SERIF, size: 74pt, weight: 700, fill: white)[
      Physics-Informed Neural Networks and Neural-Mechanistic Hybrids for a
      WNT–RA–HOX Model of Colorectal Cancer Stemness
    ]
    #v(16pt)
    #grid(columns: (1fr, auto), column-gutter: 0.8in,
      align: (left + horizon, right + horizon),
      text(size: 28pt, fill: rgb("#F4DEE3"))[
        A nondimensionalized 7-ODE model, sparse-data forward solves, Bayesian
        identifiability, and learned terms.],
      text(size: 31pt, weight: 600, fill: white)[
        Amirkhan Aidarkhan · Pascal Kataboh],
    )
  ]
])

// =====================================================================
//  FOUR COLUMNS
// =====================================================================
#let CTOP = 4.35in
#let CH = H - CTOP - 1.55in
#let colx(i) = MARGIN + i * (COL + GUTTER)

// ---------------- COLUMN 1 --------------------------------------------
#place(top + left, dx: colx(0), dy: CTOP, block(width: COL, height: CH)[
  #colhead("1", "Seven equations, 36 parameters.")

  #card(fill: white, stroke-c: rule-c)[
    #text(size: 27pt)[
      WNT signalling drives proliferation in the intestinal crypt, retinoic acid
      drives differentiation, and the HOX genes arbitrate. Loss of APC breaks
      that balance. We nondimensionalize a 7-ODE model of this network, solve it
      with a Fourier-feature PINN from 40 observations, invert it for the 36
      parameters, and read the non-identifiability that remains off a Bayesian
      posterior. Replacing one mechanistic term with a neural network returns
      the problem in a sharper form: the constraint meant to protect the host
      equation is imposed where the data never reach. One knockdown condition,
      at a fixed budget, removes the error.
    ]
  ]

  #fig("assets/ready/schema.pdf", w: 100%)
  #implic[Green activates, red inhibits, dashed mediates. Seven species, one
    constitutive WNT drive, one vitamin-A input.]

  #subhead[Nondimensionalization]

  #text(size: 27pt)[Scale each species by its initial value and time by the
    β-catenin turnover rate #mono[d#sub[B]] = 1 hr#super[−1]:]

  #eq(size: 29pt)[$B = B_0 b, quad P = P_0 p, quad dots, quad tau = d_B t$]

  #text(size: 27pt)[Every loss term normalizes to $-x$, so each equation
    collapses to one form with a pure timescale ratio in front:]

  #eq(size: 32pt)[$epsilon_X (d x) / (d tau) = "production" - x$]

  #text(size: 27pt)[For β-catenin, in full:]

  #eq(size: 29pt)[$(d b)/(d tau) = W + eta_13 (h_13^n)/(kappa_13^n + h_13^n)
    - b - lambda_P p b - lambda_5 (h_5 b)/(kappa_5 + b)$]

  #text(size: 23pt, fill: ink2)[The units and the count change, not the model.
    The system is stiff: $epsilon_R = 0.40$, $epsilon_M = 0.60$.]

  #card(fill: rgb("#EFF4F8"))[
    #text(size: 27pt, weight: 600)[The readout: the stemness index]
    #eq(size: 31pt)[$S(tau) = (b (1 + alpha_13 h_13)) / ((1 + p)(1 + alpha_5 h_5))$]
    #fine[Four regimes, Normal to Severe APC Loss, differ only in the WNT drive
      #mono[W] (0.8 → 2.0) and APC functionality $theta_P$ (1.0 → 0.25).]
  ]
])

// ---------------- COLUMN 2 --------------------------------------------
#place(top + left, dx: colx(1), dy: CTOP, block(width: COL, height: CH)[
  #colhead("2", "One network solves it, then inverts it.")

  #fig("assets/ready/forward_arch.pdf", w: 66%)
  #implic[The network proposes a trajectory, and automatic differentiation feeds
    it back through the ODEs. The inverse network is the same graph with θ
    trainable.]

  #eq(size: 30pt)[$cal(L) = cal(L)_"data" + cal(L)_"phys" + 20 cal(L)_"ic"$]

  #text(size: 27pt)[Time enters through a fixed random Fourier embedding, 16
    modes at $sigma = 4$, then a 256-wide, 4-deep GELU network:]

  #eq(size: 28pt)[$gamma(tau) = [tau/T, space sin(2 pi tau/T B_k), space
    cos(2 pi tau/T B_k)]_(k=1)^(16)$]

  #fig("assets/fig_forward_fit.pdf", w: 100%)
  #implic[Trained on 40 sparse observations plus the physics residual, the PINN
    tracks the Radau reference in every regime. Without the embedding the same
    network returns straight lines.]

  #card(fill: white, stroke-c: rule-c)[
    #grid(columns: (1fr, 1fr), column-gutter: 14pt,
      tile("1.06%", "dense labels, grand mean rel-L2", c: blue),
      tile("2.41%", "from 40 observations", c: orange),
    )
  ]

  #subhead[Running it backwards]

  #text(size: 27pt)[Freeing θ costs accuracy through gradient starvation and
    autodiff derivative bias. A derivative-free integral residual removes the
    second cause:]

  #eq(size: 28pt)[$r_k = (z_(k+1) - z_k) - (Delta tau_k)/2 (f_k + f_(k+1))$]

  #fig("assets/fig_inv_recovery.pdf", w: 100%)
  #implic[True against recovered, for three of the best-converging parameters.]

  #card(fill: rgb("#EFF4F8"))[
    #text(size: 25pt)[Parameters recovered under 10% error, four regimes:
      #text(fill: ink, weight: 600, size: 25pt)[10/4/5/7 → 17/16/10/7], a total
      of #text(fill: ink, weight: 600, size: 25pt)[37 → 50] of 144. Severe APC
      Loss does not move. What is left there is structural.]
  ]
])

// ---------------- COLUMN 3 --------------------------------------------
#place(top + left, dx: colx(2), dy: CTOP, block(width: COL, height: CH)[
  #colhead("3", "A posterior says what data cannot pin down.")

  #text(size: 27pt)[
    HMC over the 36 parameters with the state networks frozen, derivative-free
    because the likelihood reuses the integral residual.
  ]

  #fig("assets/fig_bayes_marginals.pdf", w: 100%)
  #implic[#mono[W] stays tight in every regime. $theta_P$ reverts toward its
    prior as the WNT drive rises. That is the high-WNT wall, stated as a
    posterior width.]

  #fig("assets/fig_posterior.pdf", w: 100%)
  #implic[The sampler rides a curved valley unprompted. The product
    $delta_(P 1)(1 - theta_P)$ is 4.7× better constrained than $delta_(P 1)$
    alone, and at the healthy truth $theta_P = 1$ it vanishes, so
    $delta_(P 1)$ is free.]

  #fig("assets/fig_fim.pdf")
  #implic[The Fisher information agrees from a third direction, degrading with
    WNT drive.]

  #card(fill: rgb("#F6EDF0"), stroke-c: garnet)[
    #text(font: SERIF, size: 27pt, weight: 600, fill: garnet)[
      The first run said 36 of 36. It was an artifact.]
    #v(4pt)
    #text(size: 25pt)[The physics likelihood summed 210k residual terms against
      a tiny σ, giving a pinpoint, biased posterior. Honest recovery is
      19/20/11/8, and the run still fails its own sample-size and coverage
      gates. We claim the geometry and the diagnosis, never the widths.]
  ]

  #fig("assets/fig_bayes_miscal.pdf", w: 100%)
  #implic[Severe APC Loss, the four worst parameters. Each truth sits outside
    its own 95% interval, and coverage is 13 of 36.]
])

// ---------------- COLUMN 4 --------------------------------------------
#place(top + left, dx: colx(3), dy: CTOP, block(width: COL, height: CH)[
  #colhead("4", "A constraint is only as good as the data that visit it.")

  #card(fill: white, stroke-c: rule-c)[
    #eq(size: 33pt)[$epsilon dot(x) = #text(fill: garnet)[$a$] + f_"NN" (u) - x$,
      #h(14pt) $f_"NN" (0) = 0$]
    #fine[The anchor is meant to stop the network absorbing a constant out of
      the basal-production parameter $a$.]
  ]

  #text(size: 27pt)[
    It does not. Across five parameterisations, including the monotone and
    bounded one the literature recommends, the basal parameter comes out
    #text(fill: red, weight: 600)[14 to 203%] wrong.
  ]

  #fig("assets/fig_support.pdf", w: 100%)
  #implic[No regulator ever approaches zero, in any of 10 conditions or 4
    regimes. The anchor is asserted at a point the data never visit.]

  #subhead[The fix is an experiment]

  #card(fill: white, stroke-c: rule-c)[
    #grid(columns: (1fr, 1fr), column-gutter: 14pt,
      [
        #text(font: MONO, size: 23pt, fill: ink2)[misses the anchor]
        #v(2pt)
        #text(size: 42pt, weight: 700, fill: orange)[2.2% / 1.5%]
      ],
      [
        #text(font: MONO, size: 23pt, fill: ink2)[reaches it, one more siRNA]
        #v(2pt)
        #text(size: 42pt, weight: 700, fill: blue)[0.0% / 0.0%]
      ])
    #v(6pt)
    #fine[Normal / Severe APC Loss. Same edge, same parameter, same seeds,
      eleven conditions in both arms.]
  ]

  #fig("assets/fig_dose_compact.pdf", w: 100%)
  #implic[Against the anchor ratio the arms collapse onto one curve. Against
    dose they do not. The anchor is the governing variable.]

  #card(fill: white, stroke-c: rule-c)[
    #fine[Not shown: an information-matched control that gains nothing in 8 of 8
      cells, and a prospective test at 95.6% → 0.0%. Restart noise is ±0.5 to
      0.9 pp, on a single seed, so there are no error bars.]
  ]

  #v(6pt)
  #block(width: 100%, inset: 16pt, radius: 4pt, fill: garnet)[
    #text(font: SERIF, size: 29pt, weight: 700, fill: white)[Takeaways]
    #v(7pt)
    #set text(size: 24pt, fill: rgb("#F7E9ED"))
    #grid(columns: (auto, 1fr), column-gutter: 10pt, row-gutter: 7pt,
      text(font: MONO, weight: 700, fill: white)[1],
      [Nondimensionalization and Fourier features make a stiff 7-ODE system
       solvable by a PINN from sparse data.],
      text(font: MONO, weight: 700, fill: white)[2],
      [Inverse recovery is limited by information, not architecture.],
      text(font: MONO, weight: 700, fill: white)[3],
      [A constraint on a learned term carries information only where the data
       have support. The remedy is experiment design, computable in advance.],
    )
  ]
])

// =====================================================================
//  FOOTER: two key references
// =====================================================================
#place(bottom + left, block(width: W, height: 1.35in, fill: white)[
  #pad(x: MARGIN, y: 0.22in)[
    #grid(columns: (auto, 1fr), column-gutter: 0.55in, align: (left + horizon, left + horizon),
      text(font: SERIF, size: 26pt, weight: 600, fill: garnet)[Key references],
      [
        #set text(size: 27pt, fill: ink)
        #set par(leading: 0.55em)
        M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural
        networks. _J. Comput. Phys._ *378*, 686 (2019).
        #linebreak()
        C. Rackauckas et al. Universal differential equations for scientific
        machine learning. arXiv:2001.04385 (2020).
      ],
    )
  ]
])
