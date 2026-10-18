# Archived poster: Typst rebuild, v1 (September 2026)

**Physics-Informed Neural Networks and Neural-Mechanistic Hybrids for a
WNT–RA–HOX Model of Colorectal Cancer Stemness**
Amirkhan Aidarkhan and Pascal Kataboh.

[PDF](poster-v1.pdf) · [source](poster-v1.typ) · [preview](preview.png)

A shorter rebuild of the [August 2026 poster](../2026-08-typst/), same title
and four-column structure (model, PINN, Bayesian inference, learned terms).
Column one opens with a short abstract followed by the key display
equations, prose is cut throughout, and every placed figure is a vector
PDF. It was superseded by the LaTeX posters in [`../../pinn-recovery/`](../../pinn-recovery/) and
[`../../hybrid-models/`](../../hybrid-models/).

## Contents

| Path | Contents |
|---|---|
| `poster-v1.typ`, `poster-v1.pdf` | Typst source and compiled poster |
| `assets/` | Figures as PDF, SVG and PNG; `assets/ready/` holds figures copied from earlier runs, the architecture diagrams and the network schematic |
| `figures/` | Matplotlib figure scripts, shared `style.py`, TikZ network schematic |
| `fonts/` | Bundled Inter, Space Grotesk, IBM Plex Mono and Source Serif 4 |
| `build.sh` | Figures, typesetting, page-size check, previews at 34 and 60 ppi |

## Build

The committed assets are enough to typeset it:

```bash
typst compile --font-path fonts poster-v1.typ poster-v1.pdf
```

Tested with Typst 0.15.1. `build.sh` also reruns the figure scripts, which
read run outputs through absolute paths of the original checkout; they are
kept as a record and will not run unchanged after the layout change.

Archived: the figures and numbers have not been re-audited, and the Bayesian
panels come from HMC runs that fail their calibration checks. The folder was
`research-poster-v1/` before the layout change.
