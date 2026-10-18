# Archived slides: dark-theme Beamer rebuild (July 2026)

**Physics-Informed Neural Network for a Reduced WNT-HOX-Retinoic Acid Model of
Colorectal Cancer**
Amirkhan Aidarkhan and Nathaniel Kim; advisor Pascal K. Kataboh. Department
of Mathematics & Statistics, Swarthmore College, 21 July 2026.

[Slides (PDF)](pinn_presentation.pdf) · [source](pinn_presentation.tex)

An earlier slide-for-slide Beamer rebuild of the 20-slide department talk, in
a dark Swarthmore-style theme. Superseded by the light-theme reconstruction in
[`../../department-talk/`](../../department-talk/).

| Path | Contents |
|---|---|
| `pinn_presentation.{tex,pdf}` | Deck source and compiled PDF |
| `beamerthemeswatdark.sty` | Dark Beamer theme |
| `assets/figures/` | Figure copies used by the slides |
| `sync_figures.sh` | Records the run each figure came from, and re-copies them |
| `legacy/` | Stand-alone references slide |
| `build.sh` | Two `pdflatex` passes into `build/`, then copies the PDF here |

Build with `bash build.sh`. `bash sync_figures.sh` re-copies the eleven slide
figures from their source runs (paths relative to the repository root, in the
current layout; five are archived run outputs, so restore those first with
`wnt-pinn artifacts restore github-history --select <path>`).

## Status

Superseded. Kept because the catalog adapter `presentation` in
[`../../../catalog.json`](../../../catalog.json) compiles it, and because the
first poster's Bayesian marginals use the same posterior draws. Its marginal
plots carry a manual Normal coverage override of 18/36; the saved posterior
summary gives 19/36, which the first poster and the result registry use.
The folder was `presentation/` before the layout change.
