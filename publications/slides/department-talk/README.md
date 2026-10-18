# Department talk (July 2026)

**Physics-Informed Neural Network for a Reduced WNT-HOX-Retinoic Acid Model of
Colorectal Cancer**
Amirkhan Aidarkhan and Nathaniel Kim; advisor Pascal K. Kataboh. Department
of Mathematics & Statistics, Swarthmore College, 21 July 2026.

[Slides (PDF)](output/latex/pinn_presentation_native.pdf) ·
[source](output/latex/pinn_presentation_native.tex)

A 20-slide, 16:9 Beamer reconstruction of the talk, slide for slide: model and
regimes, Radau reference, forward PINN, inverse PINN, Bayesian PINN,
parameter-recovery comparison and next steps. Plots are repository outputs at
native resolution, on a light academic theme instead of the original's dark
one. The original slide export is not distributed.

## Contents

| Path | Contents |
|---|---|
| `output/latex/pinn_presentation_native.{tex,pdf}` | The deck: repository-native vector diagrams, TeX Gyre font fallback |
| `output/latex/pinn_presentation.tex` | Closer reconstruction of the original; needs the Avenir Next font |
| `output/latex/assets/` | Figure copies, TikZ neural-network diagram, slide images from the original deck (`source_*.jpg`) |
| `output/latex/README.txt` | Design notes and the source of each figure |

## Build

```bash
cd output/latex
lualatex -interaction=nonstopmode -halt-on-error pinn_presentation_native.tex
lualatex -interaction=nonstopmode -halt-on-error pinn_presentation_native.tex
```

The native deck uses Avenir Next and Menlo when installed, otherwise TeX Gyre
Heros and Cursor, and also builds with `pdflatex`. The figures in `assets/`
are copies, not links: re-copy one after rerunning its experiment.
`README.txt` gives each figure's source path in the current layout (the
figures were matched at commit `e29ecd9`, whose old-layout paths
[`docs/layout.md`](../../../docs/layout.md) translates).
