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
#set text(font: SANS, fill: ink, size: 28pt, lang: "en")
#set par(leading: 0.72em, spacing: 1.08em, justify: false)

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

#let subhead(t) = block(above: 32pt, below: 17pt)[
  #text(font: SERIF, size: 31pt, weight: 600, fill: garnet)[#t]
]

// the one-line implication that rides under every figure
#let implic(t) = block(above: 10pt, below: 6pt, width: 100%)[
  #text(size: 24pt, fill: ink)[#t]
]

#let fine(t) = text(size: 22pt, fill: ink2)[#t]
#let mono(t) = text(font: MONO, size: 26pt)[#t]

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
        Sparse data PINNs · identifiability · experiment design],
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
  #colhead("1", "WNT–RA–HOX in seven states.")

  #card(fill: white, stroke-c: rule-c)[
    #text(font: SERIF, size: 27pt, weight: 600, fill: garnet)[Abstract]
    #v(6pt)
    #text(size: 27pt)[
      APC loss deregulates WNT/β-catenin signalling; retinoic acid counters it
      through HOX. We study a nondimensional seven-state ODE system across four
      disease regimes. A Fourier-feature PINN solves it from 40 observations;
      an integral residual improves inverse recovery. Fisher and posterior
      analyses expose the remaining high-WNT ambiguity. In a neural-mechanistic
      hybrid, one targeted siRNA restores identifiability by making the data
      visit the learned term's zero anchor.
    ]
  ]

  #fig("assets/ready/schema.pdf", w: 100%)
  #implic[Green activates, red inhibits, blue mediates.]

  #subhead[Dimensionless system]

  #eq(size: 28pt)[$bold(y) = [b,p,h_5,h_13,m,r,c]^T, quad
    X = X_0 x, quad tau = d_B t$]

  #eq(size: 31pt)[$epsilon_x dot(x) = g_x(bold(y); theta) - x,
    quad theta in RR^36$]

  #eq(size: 26pt)[$dot(b) = W + eta_13 (h_13^n)/(kappa_13^n+h_13^n) - b
    - lambda_P p b - lambda_5 (h_5 b)/(kappa_5+b)$]

  #eq(size: 26pt)[$epsilon_5 dot(h)_5 = a_5 + eta_R r/(kappa_R+r) - h_5
    - eta_M (m h_5)/(kappa_M+m)$]

  #card(fill: rgb("#EFF4F8"))[
    #eq(size: 30pt)[$S(tau) = (b (1 + alpha_13 h_13)) /
      ((1 + p)(1 + alpha_5 h_5))$]
    #fine[Normal → Severe APC Loss: $W = 0.8 → 2.0$;
      $theta_P = 1.0 → 0.25$.]
  ]
])

// ---------------- COLUMN 2 --------------------------------------------
#place(top + left, dx: colx(1), dy: CTOP, block(width: COL, height: CH)[
  #colhead("2", "Sparse trajectories, then inversion.")

  #fig("assets/ready/forward_arch.pdf", w: 66%)
  #implic[Fourier features resolve fast forcing; θ is fixed forward and
    trainable inverse.]

  #eq(size: 30pt)[$cal(L) = cal(L)_"data" + cal(L)_"phys" + 20 cal(L)_"ic"$]

  #eq(size: 28pt)[$gamma(tau) = [tau/T, space sin(2 pi tau/T B_k), space
    cos(2 pi tau/T B_k)]_(k=1)^(16)$]

  #fig("assets/fig_forward_fit.pdf", w: 100%)
  #implic[40 samples plus the ODE residual; PINN predictions are dashed.]

  #card(fill: white, stroke-c: rule-c)[
    #grid(columns: (1fr, 1fr), column-gutter: 14pt,
      tile("1.06%", "dense labels", c: blue),
      tile("2.41%", "40 observations", c: orange),
    )
  ]

  #subhead[Integral inverse residual]

  #eq(size: 28pt)[$r_k = (z_(k+1) - z_k) - (Delta tau_k)/2 (f_k + f_(k+1))$]

  #fig("assets/fig_inv_recovery.pdf", w: 100%)
  #implic[True (grey) and recovered (blue).]

  #card(fill: rgb("#EFF4F8"))[
    #text(size: 26pt, weight: 600)[Under 10% error]
    #v(4pt)
    #text(size: 28pt)[10/4/5/7 → 17/16/10/7]
    #fine[Across regimes: 37 → 50 of 144 parameters.]
  ]
])

// ---------------- COLUMN 3 --------------------------------------------
#place(top + left, dx: colx(2), dy: CTOP, block(width: COL, height: CH)[
  #colhead("3", "High WNT hides APC functionality.")

  #eq(size: 30pt)[$p(theta | cal(D)) prop exp(-cal(L)_"int"(theta)/(2 sigma^2))
    p(theta)$]

  #fig("assets/fig_bayes_marginals.pdf", w: 100%)
  #implic[$W$ stays tight; $theta_P$ broadens as WNT rises.]

  #fig("assets/fig_posterior.pdf", w: 100%)
  #eq(size: 29pt)[$delta_P(theta_P) = 1 + delta_(P 1)(1-theta_P)$]
  #implic[Only the product $delta_(P 1)(1-theta_P)$ stays constrained.]

  #fig("assets/fig_fim.pdf")
  #implic[Independent Fisher analysis gives the same verdict.]

  #card(fill: rgb("#F6EDF0"), stroke-c: garnet)[
    #text(font: SERIF, size: 28pt, weight: 600, fill: garnet)[Calibration check]
    #v(5pt)
    #text(size: 27pt)[Recovery: *19/20/11/8*. Coverage: *13/36*.]
    #fine[Interpret posterior geometry, not interval widths.]
  ]
])

// ---------------- COLUMN 4 --------------------------------------------
#place(top + left, dx: colx(3), dy: CTOP, block(width: COL, height: CH)[
  #colhead("4", "Visit the constraint.")

  #card(fill: white, stroke-c: rule-c)[
    #eq(size: 33pt)[$epsilon dot(x) = #text(fill: garnet)[$a$] + f_"NN" (u) - x$,
      #h(14pt) $f_"NN" (0) = 0$]
    #fine[The zero anchor protects the basal parameter $a$.]
  ]

  #implic[Five parameterisations still leave $a$ #text(fill: red,
    weight: 600)[14 to 203%] wrong.]

  #fig("assets/fig_support.pdf", w: 100%)
  #implic[Ten conditions never approach the anchor.]

  #subhead[One targeted condition]

  #card(fill: white, stroke-c: rule-c)[
    #grid(columns: (1fr, 1fr), column-gutter: 14pt,
      [
        #text(font: MONO, size: 23pt, fill: ink2)[misses anchor]
        #v(2pt)
        #text(size: 42pt, weight: 700, fill: orange)[2.2% / 1.5%]
      ],
      [
        #text(font: MONO, size: 23pt, fill: ink2)[+ one siRNA]
        #v(2pt)
        #text(size: 42pt, weight: 700, fill: blue)[0.0% / 0.0%]
      ])
    #v(6pt)
    #fine[Normal / Severe APC Loss; eleven conditions per arm.]
  ]

  #fig("assets/fig_dose_compact.pdf", w: 100%)
  #implic[Error collapses by anchor ratio, not by dose.]

  #v(12pt)
  #block(width: 100%, inset: 16pt, radius: 4pt, fill: garnet)[
    #align(center)[#text(font: SERIF, size: 30pt, weight: 700, fill: white)[
      Experiment design restores identifiability.]]
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
