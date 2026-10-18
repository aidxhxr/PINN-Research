# Archived poster: Typst project walkthrough (August 2026)

**Physics-Informed Neural Networks and Neural-Mechanistic Hybrids for a
WNT–RA–HOX Model of Colorectal Cancer Stemness**
Amirkhan Aidarkhan and Pascal Kataboh.

[PDF](poster.pdf) · [source](poster.typ) · [preview](preview.png)

A 48 × 36 inch landscape poster in Typst that walks through the whole
project in four columns: the nondimensionalized seven-ODE model, the forward
and inverse PINN, Bayesian inference, and the neural–mechanistic hybrid. It
was superseded by the shorter [2026-09 rebuild](../2026-09-typst-v1/) and then
by the LaTeX posters in [`../../pinn-recovery/`](../../pinn-recovery/) and
[`../../hybrid-models/`](../../hybrid-models/).

## Contents

| Path | Contents |
|---|---|
| `poster.typ`, `poster.pdf` | Typst source and compiled poster |
| `assets/` | Figures placed by the poster; `assets/ready/` holds figures copied from earlier runs |
| `figures/` | Matplotlib scripts for the poster's own figures, shared `style.py`, recoloured TikZ network schematic |
| `fonts/` | Bundled Inter, Space Grotesk, IBM Plex Mono and Source Serif 4 |
| `build.sh` | Figures, typesetting, page-size check, low-resolution preview |

## Build

The committed assets are enough to typeset it:

```bash
typst compile --font-path fonts poster.typ poster.pdf
```

Tested with Typst 0.15.1. `build.sh` also reruns the figure scripts, which
read run outputs through absolute paths of the original checkout; they are
kept as a record and will not run unchanged after the layout change.

Archived: the figures and numbers have not been re-audited, and the Bayesian
panels come from HMC runs that fail their calibration checks. The folder was
`research-poster/` before the layout change.
