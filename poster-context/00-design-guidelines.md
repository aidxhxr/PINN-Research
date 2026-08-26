# Design guidelines (both posters)

## Physical constraints

- **Hard maximum: 48" × 36", either orientation.**
- Recommended: **landscape 48w × 36h** for both posters — both stories read
  left-to-right in 3 acts, which maps onto a 3- or 4-column landscape grid.
  (Use portrait 36×48 only if the venue's easels demand it.)
- Print bleed: keep all content ≥ 1" from every edge; nothing critical within
  2" of the edge.

## Layout skeleton (landscape 48×36)

- **Title band** across the full width, ~5–6" tall: title, authors,
  affiliation, logos. One-line take-home message directly under the title
  ("banner claim") — many viewers read only this.
- **4 columns** of ~11" each with ~0.75" gutters, or 3 columns of ~15".
  Reading order: col 1 = intro/model, middle = methods + results, last col =
  the headline result + limits + references.
- **The headline figure gets the biggest box** (≥ 12" wide). One figure per
  panel; never shrink a 7-subplot grid onto a poster — pick 2–3 states or
  re-render.

## Type sizes (at full 48×36 scale)

| element | size |
|---|---|
| title | 80–96 pt |
| banner claim | 48–54 pt |
| section headers | 44–54 pt |
| body text | 28–32 pt (minimum 24) |
| captions / axis labels | 24 pt (re-render matplotlib figures with `fontsize>=16` at the figure's native size so they survive scaling) |
| references | 18–20 pt, it's fine to be small |

Body text budget: **≤ 800 words total** per poster. Bullets over paragraphs;
every panel should be parseable in 10 seconds.

## Palette — REQUIRED (professor's standing request)

Use the SA-big colors on **all** figures and poster accents:

| role | hex |
|---|---|
| primary / IDENTIFIABLE | `#1f77b4` (blue) |
| secondary / WEAK | `#ff7f0e` (orange) |
| neutral / reference / de-emphasis | grey (`#7f7f7f`) |
| alert / NON-IDENTIFIABLE / failure | `#cc3333` (red) |

Identifiability verdicts are always: **IDENT = blue, WEAK = orange,
NON-IDENT = red.** Regenerate any legacy figure that uses a different mapping
before it goes on a poster.

Background: white or the presentation's Paper `#F7F6F3`; ink `#152A3A`. If you
want a heading accent beyond the four SA colors, the deck's Garnet `#9C2745`
exists, but don't introduce it into data figures.

## Fonts

Match the deck (`presentation-codex`): Avenir Next (fallback TeX Gyre Heros /
Helvetica) for text, Menlo / TeX Gyre Cursor for code and parameter names.

## Figure rules

- Every figure is a **copy** placed into the poster project's `assets/`, with
  its source run-dir recorded in the figure manifest (`04-figures.md`) —
  snapshots, not symlinks, same convention as `presentation-codex`.
- Prefer vector (PDF/SVG) where a source exists (`schematic.tex`, TikZ network);
  PNGs from runs should be ≥ 200 dpi at printed size.
- Tables beat bar-chart forests for recovery counts; one bar chart maximum per
  poster.

## Build route

Either LaTeX `beamerposter` (`\usepackage[orientation=landscape,size=custom,width=121.92,height=91.44,scale=1.4]{beamerposter}` — cm for 48×36 in) reusing the
`presentation-codex` preamble colors, or a 48×36 in PowerPoint/Illustrator
board. LaTeX recommended: the math is heavy and the deck preamble already
defines the design system.
