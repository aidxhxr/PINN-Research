# Poster B — neural–mechanistic hybrid revision

This is the **second poster**, focused on neural–mechanistic hybrids. The first
poster in `../research-poster-latex/` focuses on dynamics, PINN reconstruction
and uncertainty. The approved original PDF, source ZIP and previews are
preserved locally in `../poster-versions/20260907_155254_approved/`.

The revised reading order is:

1. **Model and formulation:** original schematic, disease regimes,
   nondimensionalization, all seven ODEs, APC functioning and retinoid forcing.
   The additive ATRA treatment-pulse plot replaces the derived stemness index.
2. **PINNs and identifiability:** Fourier features, network architecture,
   forward loss and accuracy, inverse integral residual, recovery benchmark,
   Bayesian geometry and local Fisher classification.
3. **Neural–mechanistic hybrid:** learned MYC activation within the ODE model,
   actual function-recovery curves, direct comparison against the mechanistic
   control, basal/neural compensation, and intervention design.

The title foregrounds neural–mechanistic hybrids. Typography, garnet header,
three-column grid, print dimensions and original model artwork are retained.
Column 2 includes beta-catenin/HOXA5 forward trajectories and a compact
accuracy summary, alongside the PINN diagram and identifiability results. The new hybrid plot uses saved learned
functions rather than an illustrative curve.

## Files

- `poster.pdf`: compiled, one-page **48 × 36 inch** landscape poster.
- `poster.tex`: editable **LuaLaTeX** source.
- `preview.png` / `preview-detail.png`: whole-page previews.
- `poster-latex-source.zip`: **compiled poster.pdf at the ZIP root**, plus
  source, required figures, numerical data and documentation.
- `assets/pinn-architecture.tex`: editable vector PINN diagram: Fourier features,
  four hidden layers, seven state outputs and the three losses with backpropagation.
- `assets/model-original.pdf`: original regulatory schematic, unchanged.
- `assets/atra-treatment.pdf`: smooth additive ATRA pulse from the saved
  trajectory parameters, with the treatment window at times 40–88.
- `assets/posterior.pdf`: compact Normal-regime HMC geometry.
- `assets/learned-myc.pdf`: actual learned and true MYC-activation curves.
- `assets/tables.tex`: native LaTeX tables generated from numerical data.
- `assets/forward.pdf`: plain beta-catenin/HOXA5 curves in column 2;
  `assets/forward-window.tex` adds native LaTeX ATRA window arrows.
- `data/hybrid_comparison.json`: original learned curves and recovered/true
  parameter dictionaries for the matched-parameter comparisons.
- `SOURCES.md`: provenance, numerical interpretation and corrections.
- `PRESERVED_VERSION.md`: location of the saved approved version.

All equations, tables, headings and highlighted text are editable LaTeX.
Fonts are standard TeX Gyre Pagella and Heros. Body text is 28 pt, captions
24 pt and tables 24–26 pt. Authors: Amirkhan Aidarkhan and Pascal K. Kataboh.
The current title is retained for discussion. The initial-condition multiplier
20 is explained as a heuristic weight that emphasizes the known initial state;
no optimality claim or weight ablation is made.

## Compile

Upload the ZIP to Overleaf, select `poster.tex` as the main document and
**LuaLaTeX** as the compiler. Included figures require no Python step.
No private fonts, Gemini theme or files outside the ZIP are needed.

Locally:

```bash
lualatex -interaction=nonstopmode -halt-on-error poster.tex
```

For figure/table regeneration, two-pass compilation, layout checks, previews
and packaging:

```bash
bash build.sh
```

This uses Python with NumPy, Matplotlib and PyMuPDF. It reads saved results;
it performs no training, ODE solving or resampling. In this repository, builds
run in the `poster_b_hybrid:build` tmux session and write new timestamped
directories under `builds/`.


## ATRA trajectory plots

The two beta-catenin and HOXA5 panels have white backgrounds and no grid. Native TikZ
annotations show a double-headed arrow from time 40 to 88, with ATRA centered
above it. `figures/forward_plot.py` regenerates the vector curves and the LaTeX
annotation wrapper from `data/forward_trajectories.npz`. Checkpoint hashes,
solver tolerances, and runtime parameters are in the adjacent provenance JSON.
These use the existing 40-observation run; no new training was performed.
