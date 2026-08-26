# Poster B (advanced) — build plan

Status: PLAN ONLY, awaiting user review. Poster A (basic: forward + inverse
PINN) is deferred; its split and outline live in `../README.md`.

## Deliverable

One print-ready PDF, **landscape 48" × 36"** (121.92 × 91.44 cm), 4-column
layout, built start-to-finish by Claude, then reviewed by the user.

Working title: *"When the Data Can't See the Parameter: Identifiability and
Experiment Design for a Hybrid Mechanistic–Neural Cancer Model"*.
Banner claim: *an `f(0)=0` constraint is inert where the data never visit — one
extra siRNA takes basal-parameter error from 2.2% to 0.0%, and a training-free
design table predicts which experiment rescues each learned term.*

## Toolchain (verified present on this machine)

| purpose | tool |
|---|---|
| poster document | **LaTeX `beamerposter`** (`/usr/share/texlive/.../beamerposter.sty`), `size=custom,width=121.92,height=91.44,orientation=landscape`, compiled with `latexmk -pdflua` (fontspec → Avenir Next w/ TeX Gyre Heros fallback, same design system as `presentation-codex`). Fallback if fonts fight: pdflatex + Heros directly. `tikzposter` exists too but beamerposter matches the deck preamble — use beamerposter. |
| new data figures | **python3 + matplotlib 3.11 / pandas** — scripts in `research-poster/figures/`, output **vector PDF**, palette locked to `#1f77b4`/`#ff7f0e`/grey/`#cc3333`, fontsize ≥ 16 at native size |
| schematic / mechanism diagrams | **TikZ** — reuse `research-paper/schematic.tex` (model network); one new small diagram: "anchor outside the data support" (observed range vs the f(0)=0 point) |
| existing figures | snapshot-copy PNGs/PDFs into `research-poster/assets/` with provenance recorded |
| build | `research-poster/build.sh`: figures → latexmk → `pdfinfo` page-size assert → `pdftoppm` render for visual check |
| verification | Read the rendered PNG(s) to eyeball layout/overflow; grep the .log for overfull boxes; word-count gate ≤ ~800 words |
| version control | atomic git commits per step (figures / tex / pdf), **no Claude co-author trailer** (user convention) |

No tmux needed: nothing long-running (figure scripts are seconds, latexmk
~minutes). If any result has to be re-run (it shouldn't — all numbers exist on
disk), that run goes in tmux per repo convention.

## Directory to create

```text
research-poster/                  # sibling of research-paper/
├── poster.tex
├── build.sh
├── figures/                      # matplotlib + TikZ sources
│   ├── fig_dose_response.py
│   ├── fig_attribution.py
│   ├── fig_prospective.py
│   └── anchor_diagram.tex
└── assets/                       # generated PDFs + snapshot copies (provenance in header comment / manifest)
```

## Figure plan (slot → source → action)

| poster slot | source of truth | action |
|---|---|---|
| model schematic (col 1) | `research-paper/schematic.tex` | recompile standalone → PDF |
| reference dynamics, small (col 1) | `research-paper/scipy_solutions.png` (or regenerate 2–3 states via `reference.py`) | snapshot; regenerate only if palette/fonts too small |
| FIM correlated-pairs / spectra (col 2) | `research-paper/fim_cross_regime_spectra.png`, `fim_severe_top8.png` (run `PINN-fisher-matrix/runs/20260711_203325_fisher/`) | snapshot; check IDENT=blue/WEAK=orange/NON-IDENT=red — regenerate from the run if not compliant |
| profile likelihood (col 2, small) | `research-paper/profile_likelihood_advanced.png` | snapshot |
| Bayesian marginals / degeneracy (col 2) | `research-paper/bayes_worst8_severe.png`; `PINN-bayesian/runs/20260713_204442_bayes/` (latest) for W/thetaP overlay | snapshot the paper figure; check the run dir for `bayes_W_thetaP.png` |
| **anchor-support diagram** (col 3) | numbers from round-2 note (regulator floors: b 0.036–0.20, r 0.136–0.142, …) | NEW TikZ or matplotlib — the conceptual centerpiece |
| **dose-response** (col 3, headline) | `PINN-hybrid-ude/runs/20260802_dose_response/dose_*_*.json` (+ existing `dose_response.png` — inspect first) | NEW matplotlib: basal error vs **anchor ratio** (log-x), 3 arms × 2 regimes, showing the cross-arm collapse; inset or twin panel vs **dose** showing no collapse |
| **attribution test** (col 4) | `runs/20260801_infoctl/comparison.txt` | NEW matplotlib: 8-cell paired dot/slope chart — baseline vs info-matched (no change) vs depletion (3/8→8/8, 6/8→8/8) |
| **prospective design-table test** (col 4) | `runs/20260802_protocol_test/`, `runs/20260802_anchor_reach/anchor_reach.txt` | NEW matplotlib grouped bars: baseline / near-miss / prescribed for `m_h13` (12.8→51.5→0.1% Normal; 95.6→59.8→0.0% Severe) + typeset mini design-table |

## Content sources (text)

Panel copy drawn from, in priority order: `notes/2026-08-02-full-writeup.md`
(hybrid numbers + interpretation + limits), `notes/2026-08-01-hybrid-edge-atlas-and-anchor-visiting.md`
(five parameterisations 14–203%, regulator floors, modulator-class result),
`notes/2026-07-07-bayesian-pinn.md` + `research-paper/paper.tex` (FIM 0→1→6→10,
cond 1.9e7–3.7e19; profile likelihood Advanced Adenoma only; honest Bayesian
19/20/11/8 + deltaP1–thetaP corr 0.998), `notes/2026-08-02-anchor-visitation-prior-art.md`
(Loman & Baker positioning; 2 citations flagged unverified — poster cites only
verified ones).

Forbidden claims (from the notes' own honesty gates): "36/36 IDENT", global
"thetaP dominates" (SA artifact), any absolute magnitude comparison across SA
boxes, crediting Stage 3, error bars that don't exist (single seed; only the
±0.5–0.9 pp screen noise floor is quotable). Screen numbers labeled as
equation-local upper bounds.

## Execution order (once approved)

1. **Content files** in this folder (`01`–`04`): final panel copy, every number
   with its source path — the poster's text is written HERE first, reviewed
   against the notes, then flowed into LaTeX.
2. **Figure scripts** — generate the 3–4 new figures; Read each rendered output
   to verify palette, fonts, and that the story is legible at ~1/4 scale.
3. **`poster.tex`** — beamerposter skeleton with the deck's colors/fonts; flow
   in copy + figures; compile.
4. **Layout iteration** — render to PNG, inspect, fix overflow/balance; assert
   page size 48×36 in via `pdfinfo`.
5. **Commit** (atomic: figures / tex+assets / final pdf) and hand over for
   review with a list of the judgment calls made.

Open items for the user (not blockers — defaults in parentheses):
author/affiliation block text (user's name, department; no logos), venue-specific
requirements if any (none assumed), title wording (working title above).
