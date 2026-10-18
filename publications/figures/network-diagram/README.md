# Regulatory network diagram

TikZ drawings of the reduced WNT–RA–HOX regulatory network: β-catenin, APC,
HOXA5, HOXA13, MYC, retinoic acid and CYP26A1, with WNT and vitamin A as
inputs. Green arrows are activation, red bars inhibition and blue dashed
arrows mediation (external input in the earlier `network_schematic`).

[Schematic (PDF)](schematic-better.pdf) · [source](schematic-better.tex) ·
[preview](preview.png)

## Contents

| File | Contents | Used by |
|---|---|---|
| `schematic-better.tex`, `.pdf` | Reviewed schematic | Both current posters (as `assets/model-original.pdf`); slide 3 of both slide decks (as `schematic.pdf`) |
| `network_schematic.tex`, `.pdf` | Earlier planar layout with a title, APC at the centre and no edge crossings | Copied into the department-talk assets; not placed by either deck |
| `preview.png` | Raster preview | Repository README figure (`docs/figures/network.png`) |

The paper draws its own network panel (`paper/figures/fig01a_network.tex`).

## Build

```bash
pdflatex schematic-better.tex
pdflatex network_schematic.tex
```

Both files are standalone and need only TikZ. The catalog adapter `network`
in [`../../catalog.json`](../../catalog.json) builds `schematic-better.tex` in
an isolated copy.

A rebuild produces an equivalent drawing, not the same bytes. The posters keep
the original PDF, whose SHA-256 is recorded in their `SOURCES.md` and checked
by their `verify.py`
(`6836f765e455581fdfdf8bff7f8b65423f9fa1c051da3ed7afed3d80576ab086`).

Before the layout change this folder was `network-diagram/`.
