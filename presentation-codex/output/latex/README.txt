PINN presentation - faithful LaTeX/Beamer conversion
=========================================

Primary source
--------------
Department PINN Presentation.pdf (20 slides, 16:9)

Output
------
pinn_presentation.tex   Editable Beamer source
pinn_presentation.pdf   Compiled presentation
assets/                 Repository-native plots and diagrams

Compile
-------
Run from this directory:

    latexmk -lualatex -interaction=nonstopmode -halt-on-error pinn_presentation.tex

Design
------
The deck is a 20-slide, one-for-one reconstruction of the source PDF. Slide
order, titles, wording, tables, figures, and the original closing slide are
preserved. Only the original dark geometric background was replaced with a
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
  PINN-smaller/forward-pinn-train-hybrid/architecture.pdf
  PINN-smaller/forward_pinn_train/scipy_core_dynamics.png
  PINN-smaller/forward_pinn_train/runs/20260711_203325/pinn7_losses.png
  PINN-smaller/forward_pinn_train/pinn_forward_core_dynamics.png
  PINN-inverse-better/architecture.pdf
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
