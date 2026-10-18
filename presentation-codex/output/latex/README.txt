PINN presentation - faithful LaTeX/Beamer conversion
=========================================

Primary source
--------------
The original 20-slide (16:9) department talk deck. It is not distributed
here; the sources below reconstruct it.

Output
------
pinn_presentation_native.tex  Beamer source with TeX Gyre font fallback
pinn_presentation_native.pdf  Compiled presentation
pinn_presentation.tex         Faithful variant; needs the Avenir Next font
assets/                       Repository-native plots and diagrams

Compile
-------
Run from this directory (twice, for cross-references):

    lualatex -interaction=nonstopmode -halt-on-error pinn_presentation_native.tex

Design
------
The deck is a 20-slide, one-for-one reconstruction of the source PDF. Slide
order, titles, wording, tables, figures, and the original closing slide are
preserved, except that the closing slide keeps only the questions prompt
and the funding acknowledgement. Only the original dark geometric background was replaced with a
consistent light academic background. The refined visual system uses Avenir
Next, warm paper, deep navy typography, Swarthmore-inspired garnet accents,
muted teal details, restrained image borders, and a lightly banded comparison
table. Slide composition and object placement remain unchanged.

Figure provenance
-----------------
The repository was read at main tree:
bc2aa2a74bf58cf7f1ca5ba55c8e9efe1df42eb3

Repository: https://github.com/aidxhxr/PINN-Research

The following assets were matched to the PDF's embedded figures and used at
native repository resolution:

  network-diagram/network_schematic.pdf
  experiments/04-forward-pinn/forward-pinn-train-hybrid/architecture.pdf
  experiments/04-forward-pinn/forward_pinn_train/scipy_core_dynamics.png
  experiments/04-forward-pinn/forward_pinn_train/runs/20260711_203325/pinn7_losses.png
  experiments/04-forward-pinn/forward_pinn_train/pinn_forward_core_dynamics.png
  experiments/06-inverse-three-condition/architecture.pdf
  PINN-inverse-pinn-boost/runs/20260711_170848_replot_legendfix/inv_losses.png
  PINN-inverse-pinn-boost/pinn_inverse_core_dynamics.png
  PINN-inverse-pinn-boost/runs/20260711_170848_replot_legendfix/inv_recovery_bars_best8.png
  PINN-bayesian/runs/20260712_204532_bayes/Normal_top8_marginals.png
  PINN-bayesian/runs/20260712_204532_bayes/Severe_APC_Loss_worst8_marginals.png

The severe-regime Bayesian panel is deliberately the worst8 plot; top8 is a
different figure and does not match the source deck.

Slide 6 preserves the source deck's complete four-regime text and adds the
user-supplied colorectal progression illustration on the right, stored as
assets/colorectal_progression.png.
