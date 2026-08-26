# Poster B — full design plan (FOR REVIEW, nothing built yet)

Synthesis of: the MATS poster corpus (20 photos, see `DESIGN-RESEARCH.md`),
SIAM poster conventions (Hammarling & Higham guidance, board specs, a real
48×36 DSWeb specimen, Red Sock judging), and modern designed-poster research
(betterposter v1/v2 + what adopters keep, grid/typography numbers, Typst
ecosystem). Sources and details live with the two research reports; this file
is the decisions.

---

## 0. The design thesis

The poster is **an argument, not a summary**. Every section header is a
finding; a judge reading only the headers gets the whole chain:

> Every method hits the same identifiability wall → the wall for *learned*
> terms is a constraint asserted where no data live → one extra knockdown
> (same data budget) closes it, causally → a training-free table predicts
> which experiment rescues which term.

Three reading layers (MIT #evenbetterposter + Red Sock anonymous judging both
demand self-sufficiency):
1. **3-second layer**: claim title + hero figure + four garnet numerals.
2. **2-minute layer**: the fast-path strip + section heads + stat tiles +
   figure implication lines.
3. **Question-time layer**: the evidence panels, the honest-failure card, the
   fine print. The poster works with nobody standing at it.

## 1. Canvas, grid, chrome

- **48in × 36in landscape** (typst: `width: 48in, height: 36in`). Fits SIAM's
  4×8 boards with air; landscape columns read top-to-bottom (SIAM's stated
  preference for vertical ordering).
- Outer margins **1.2in**; one spacing unit **u = 0.4in**; all gaps are 1u/2u/3u.
- **Title band** across the top, **~4.8in** tall (13% of height — well under
  the 15–18% cap), garnet ground.
- **Fast-path strip** under it, **~4.0in**, paper ground with hairline garnet
  rule: the "Only got 2 minutes?" layer (MATS 2355/2396 move).
- Below: **4 columns**, each **~10.85in** wide with **0.65in** gutters,
  ~23.6in tall. Strict top-to-bottom reading inside a column. Regions are
  separated by space + background tint (white cards on `#F7F6F3` paper),
  hairline rules only under section heads — no heavy boxes (Faulkes).
- **Numbered story path**: garnet circle badges ①–④ at each column head
  (the cheap fix for reading-order ambiguity; MATS 2380 move).

## 2. Type system (fonts already downloaded to `research-poster/fonts/`)

| role | face | size @ full scale |
|---|---|---|
| eyebrow over title (venue line) | IBM Plex Mono, caps, tracked | 22pt |
| **title (the claim)** | **Source Serif 4** (to download) or Space Grotesk 700 — *recommend serif; it is what the best MATS boards use* | 92–100pt |
| thesis subtitle (1 sentence, 2 highlighted spans) | Inter 500 | 34pt |
| authors + affiliation | Inter 400/600 | 26–30pt |
| section eyebrows (`01 · IDENTIFIABILITY`) | IBM Plex Mono 600, caps | 20pt |
| **section heads (findings)** | Source Serif 4 semibold, garnet | 40–44pt |
| body | Inter | 26–28pt, line length ≤65 chars |
| stat-tile values | Inter 700 (proportional figures) | 54–64pt |
| figure implication lines (above/below each figure — SIAM's own prescription) | Inter 600 | 24pt |
| captions / fine print | Inter | 20–22pt |
| params, protocols, numerals in text | IBM Plex Mono | ~2pt below body |
| references | Inter | 17pt |

Word budget: **≤280 words** in the walk-by layer (heads, tiles, claim lines);
**≤900 total** including captions and fine print. No abstract (SIAM guidance
sides with dropping it). Sentence fragments allowed.

## 3. Color

| role | color |
|---|---|
| chrome (band, badges, heads, rules) | Garnet `#9C2745` (≈ MATS maroon — the family look) |
| ground | Paper `#F7F6F3`; cards white |
| text | Ink `#152A3A`, secondary `#5A6B78` |
| data series | professor's SA palette: `#1f77b4` / `#ff7f0e` / `#7f7f7f` / `#cc3333` |
| semantic mapping (stated ONCE in a small legend, col 1) | IDENT = blue · WEAK = orange · NON-IDENT = red; regimes Normal→Severe = blue→orange→grey→red |

Chrome never uses the data colors; data figures never use garnet. Red is
reserved for failure/NON-IDENT semantics (never a neutral 3rd series).

## 4. Title options (pick one — my recommendation first)

1. **"A constraint is only as good as the data that visit it"** —
   subtitle: *Identifiability and experiment design for a neural–mechanistic
   model of colorectal-cancer stemness: an `f(0)=0` anchor is inert until one
   knockdown drives the regulator to zero — **2.2% → 0.0%** basal-parameter
   error at a **fixed** experiment budget.*
2. "When the data can't see the parameter" (earlier working title — good, less
   specific).
3. Question form (MATS style): "Why does a constrained neural term still steal
   your parameter?"

Author block: **Amirkhan Aidarkhan** + affiliation/mentor line — NEED FROM
USER (department wording; whether professor's name appears).

## 5. Fast-path strip ("Only got 2 minutes?")

Left → right, four steps with arrows (this strip doubles as the blitz slide):
1. **The system** — mini WNT–RA–HOX network schematic (recolored TikZ; the
   non-math entry point every math-poster critique demands).
2. **The problem** — "replace one term with a small net → the equation's basal
   parameter comes out 14–203% wrong, under *every* constraint tried."
3. **The cause** — mini support diagram: observed data range vs the `f(0)=0`
   anchor it never visits.
4. **The fix + rule** — "one extra knockdown reaches the anchor: 2.2→0.0%
   (Normal), 1.5→0.0% (Severe); a training-free table predicts which
   experiment rescues which term — prospectively."
Under the steps, one line each: *We study / We find / We claim* (MATS 2355).

## 6. The four columns (content locked to sources)

### ① The model, and a wall every method hits
*Eyebrow* `01 · THE MODEL & THE WALL` · *Head*: **"Every method finds the
same identifiability wall."**
- 2-line model intro: 7-state WNT–RA–HOX ODE system, 36 parameters, 4 disease
  regimes (only `W`, `thetaP` differ), stiff circadian + ATRA forcing.
  **1 display equation** (compact UDE-ready form of one balance law).
- FIG `fig_regimes` — β-catenin + APC trajectories across regimes (regime
  colors), implication line: "Rising WNT drive saturates β-catenin — the high
  regimes are where recovery dies."
- Stat tile row: **36 params · ceiling 13–18/36 by any method · cond(FIM) up
  to 3.7×10¹⁹**.
- FIG `fig_fim` — stacked IDENT/WEAK/NON-IDENT composition per regime
  (blue/orange/red), + correlated-pairs counts; implication: "The wall is
  practical, not structural — and it is the same wall in every view."
- FIG `fig_posterior` (small) — `deltaP1`–`thetaP` posterior valley (HMC
  samples, Normal); implication: "The sampler finds the degeneracy on its own:
  corr 0.998; only the product `deltaP1·(1−thetaP)` is constrained."
- One honest line (fine print): Bayesian marginal *widths* not quoted — the
  run fails its own calibration gates (truth 34–1420σ out); geometry only.
  Profile likelihood: one regime (Advanced Adenoma), agrees.

### ② The diagnosis: the anchor is asserted where no data live
*Eyebrow* `02 · LEARNED TERMS` · *Head*: **"The constraint is applied where
nothing observes it."**
- **1 display equation**: `ε·dx/dt = a + f_NN(u) − loss terms`, with
  `f_NN(0) = 0` — "the anchor exists to stop the net stealing the constant
  `a`."
- FIG `fig_param_fail` — five constraint parameterisations (incl. the
  monotone+bounded one the literature recommends): basal error 13.8–202.8%
  (bar chart, one start, Normal, `bm_myc`); implication: "Not fixable by
  architecture. Multi-start matters more than any of them."
- FIG `fig_support` (column hero) — per regulator, observed support
  `[x_lo/x_hi, 1]` vs the anchor at 0; floors: `b` 0.036, `r` 0.136 — **no
  regulator ever approaches zero across all 10 conditions × 4 regimes**;
  protocol arrows to 0 from the design table; APC's structural floor in red.
- Modulator aside (2 lines + inline 3-number strip): edges that *multiply* a
  state have no constant to steal — 0.0–4.5% error, all equation params
  recovered, no depletion needed. The degeneracy is specific to production
  edges.

### ③ The evidence: reach the anchor and the error vanishes
*Eyebrow* `03 · CAUSAL TEST` · *Head*: **"One siRNA: 2.2% → 0.0%. Same
equation, same budget, eleven conditions in every arm."**
- FIG `fig_dose` (POSTER HERO — biggest box on the board) — dose–response,
  3 arms × 6 doses × 2 regimes, condition count fixed at 11:
  left panel basal error vs **dose** (arms disagree), right panel vs **anchor
  ratio** (arms collapse onto one curve — including two different edges in
  different equations). Bad-restart cell (Severe `ra`, k=0.1) annotated
  honestly. Implication line: "The anchor ratio, not the intervention, is the
  governing variable — this is what makes it a design rule."
- Stat tile pair (the causal claim): `wnt` (misses anchor, 0.013) **2.2 / 1.5%**
  vs `bcat` (+1 siRNA, reaches 0) **0.0 / 0.0%** — Normal / Severe.
- FIG `fig_attribution` — 8-cell dot plot: 10 cond → +2 info-matched (no
  anchor moved) → +2 depletion. Implication: "Two extra experiments that miss
  the anchor change nothing (0 gains in 8 cells, deltas ≤0.7pp); the same-sized
  pair that reaches it: 3/8→8/8 and 6/8→8/8."

### ④ The design rule — and where it fails
*Eyebrow* `04 · PROSPECTIVE DESIGN` · *Head*: **"A table computed without
training prescribes the rescue — and predicted it in advance."**
- Mini design table (typeset, mono): regulator → cheapest protocol reaching
  its anchor → measured outcome (from `anchor_reach.txt`), APC row flagged
  structurally unreachable (production ≥ 1).
- FIG `fig_prospective` — two small multiples:
  `m_h13`: baseline 12.8/95.6% → near-miss (a LARGER double knockout that
  misses the anchor) 51.5/59.8% → prescribed `mycKO` **0.1/0.0%**;
  `h13_b`: flat (16.2% everywhere) — the informative failure. Implication:
  "Bigger perturbation ≠ better: the near-miss is worse than doing nothing.
  And `h13_b` sharpens the rule…"
- **The two-clause rule** (garnet pull-quote): *reach the anchor — with a
  condition that still contains the basal term.* (`hox13KO` reaches `h13`'s
  anchor by deleting `W`, which IS the parameter; encoded in `anchor_reach.py`,
  which now rejects that cell from theory.)
- **Takeaways card** (dark garnet, white text, 3 numbered):
  1. Anchor/shape constraints on learned terms carry information only where
     data have support — partial persistence of excitation, applied to the
     constraint rather than the estimate.
  2. The remedy is experiment design, not architecture: one graded knockdown
     at a fixed budget, predicted by a training-free reachability table.
  3. Functional and parametric identifiability move independently — the
     prescribed protocol fixed the parameter while functional error rose.
- **Honest limits card** (paper tint): equation-local screen = upper bound
  (exact states); ±0.5–0.9pp restart noise floor; 2 restarts thin in Severe;
  single seed → no error bars; cross-sectional corr is −0.72 (the controlled
  comparisons carry the claim); 4 of 6 pre-registered dose predictions failed
  (2 miscalibrated thresholds, 1 bad restart, 1 design flaw) — stated, not
  hidden.
- References (≤5): Loman & Baker arXiv:2510.14140 · Philipps/Schmid/Hasenauer
  npj Syst Biol Appl 11:101 (2025) · Wang & Hill IEEE TNN 2006 · Plate et al.
  arXiv:2408.07143 · Yang et al. arXiv:2003.06097. (Only verified citations;
  the two flagged-unverified ones from the prior-art note stay off the poster.)
- Contact line (email). QR codes: only if the user wants repo/paper links
  public — ASK.

## 7. Figures to build (9 total; all matplotlib→SVG at final print size, all
in the SA palette, direct labels, implication line lives in typst not the SVG)

| id | source data | form |
|---|---|---|
| strip schematic | `research-paper/schematic.tex` recolored, compiled standalone | TikZ→PDF/SVG |
| strip mini-support | derived from `fig_support` | inline SVG |
| `fig_regimes` | Radau solve via `PINN-hybrid-ude` config | 2-panel line |
| `fig_fim` | `PINN-fisher-matrix/runs/20260711_203325_fisher/*_fim_summary.json` + `*_correlated_pairs.csv` | stacked verdict bars + pair counts |
| `fig_posterior` | `PINN-bayesian/runs/20260713_204442_bayes/Normal_posterior_samples.npz` | 2-D sample cloud + product line |
| `fig_param_fail` | round-2 note screen table (gated/sc/sc_bounded/lin/lin_mono) | thin bar |
| `fig_support` | `runs/20260802_anchor_reach/anchor_reach.txt` + floors | horizontal support bars + protocol arrows |
| `fig_dose` | `runs/20260802_dose_response/dose_*.json` | 2-panel (vs dose / vs ratio) × 2 regimes |
| `fig_attribution` | `runs/20260801_infoctl/comparison.txt` | 8-row paired dot plot |
| `fig_prospective` | `runs/20260802_protocol_test/protocol_*.json` | grouped bars, 2 small multiples |

Figure rules (dataviz skill): thin marks, hairline solid grids, direct labels
selective, legend for ≥2 series, text in ink not series color, no dual axes,
per-figure secondary encoding so color is never the only channel.

## 8. Build & verification pipeline (all tools verified installed)

1. `research-poster/figures/*.py` → SVG+PNG (PNG for my inspection).
2. Schematic: recolor tex → `pdflatex` → convert (check `pdftocairo -svg`,
   else 600dpi PNG).
3. `poster.typ` — hand-rolled layout (no theme package): absolute grid via
   `place()`/`grid`, fonts via `--font-path fonts/`.
4. `build.sh`: figures → `typst compile` → assert page size → `typst compile
   --format png --ppi 40` for a squint-test render I inspect visually each
   iteration (3-second layer must show: claim, hero, four numerals — nothing
   else).
5. Word-count gate (script counts text in poster.typ layers).
6. Iterate on rendered previews until balanced; then hand to user with the
   judgment-call list. Git commits only when user asks.

## 9. Open items for the user (not blockers)

1. Title choice (§4 — recommend option 1).
2. Author/affiliation line wording; include professor/mentor name?
3. QR codes: link the GitHub repo / paper PDF, or omit?
4. Serif display face OK (Source Serif 4, one more font download), or keep
   all-sans with Space Grotesk?
