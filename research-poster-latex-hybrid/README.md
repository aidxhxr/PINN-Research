# Poster B — neural–mechanistic hybrid revision

This is the **second poster**, focused on neural–mechanistic hybrids. The first
poster in `../research-poster-latex/` focuses on dynamics, PINN reconstruction
and uncertainty. The approved original PDF, source ZIP and previews are
preserved locally in `../poster-versions/20260907_155254_approved/`.

The poster uses short headings and captions, with the plots and equations
carrying the results. Detailed interpretation remains in `SOURCES.md`.
The version before the 2026-09-17 text reduction is preserved in
`../poster-versions/20260917_195027_before_text_reduction/`.

The reading order is:

1. **Model and formulation:** original schematic, disease regimes,
   nondimensionalization, all seven ODEs, APC functioning and retinoid forcing.
   The additive ATRA treatment-pulse plot replaces the derived stemness index.
2. **PINNs and identifiability:** Fourier features, network architecture,
   forward loss and accuracy, inverse integral residual, recovery benchmark,
   APC parameter trade-off and local Fisher classification.
3. **Neural–mechanistic hybrid:** learned MYC activation within the ODE model,
   actual function-recovery curves, direct comparison against the mechanistic
   control, basal/neural compensation, and direct function/parameter error comparisons.

The title foregrounds identifiability and experimental design. Typography, garnet header,
three-column grid, print dimensions and original model artwork are retained.
Column 2 includes beta-catenin/HOXA5 forward trajectories and a compact
accuracy summary, alongside the PINN diagram and identifiability results.
The learned-function and error-comparison plots use saved full-model fits.
The current comparison version is also served at
`http://localhost:8003/second/poster_v1.pdf`.

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
- `assets/parity.pdf`: recovered vs. true value, integral inverse PINN, 36 parameters, Normal and Severe APC Loss
  (`figures/parity_plot.py`).
- `assets/normal_pinn.pdf`: Normal-regime MYC/APC inverse-PINN fit, untreated vs ATRA
  (`figures/treatment_plot.py`, data reused from the first poster).
- `assets/learned-myc.pdf`: actual learned and true MYC-activation curves.
- `assets/hybrid-errors.pdf`: mechanistic versus hybrid function NRMSE and
  basal-production parameter errors for β-catenin → MYC and RA → HOXA5,
  across all four regimes. The separate intervention screen has been removed.
- `assets/hybrid_error_audit.json`: the exact values shown in the comparison.
- `assets/tables.tex`: native LaTeX tables generated from numerical data.
- `assets/forward.pdf`: plain beta-catenin/HOXA5 curves in column 2;
  `assets/forward-window.tex` adds native LaTeX ATRA window arrows.
- `data/hybrid_comparison.json`: original learned curves and recovered/true
  parameter dictionaries for the matched-parameter comparisons.
- `SOURCES.md`: provenance, numerical interpretation and corrections.
- `PRESERVED_VERSION.md`: location of the saved approved version.

All equations, tables, headings and captions are editable LaTeX.
Fonts are standard TeX Gyre Pagella and Heros. Body text is 28 pt, captions
24 pt and tables 24–26 pt. Authors (2026-10-14 title bar, format set by the advisor): Amirkhan Aidarkhan¹ ·
Advisor: Pascal K. Kataboh¹, with ¹ Department of Mathematics & Statistics, Swarthmore
College, Swarthmore, PA 19081 on the line below (same institution, so one superscript; no
role descriptions). The title block is
left-aligned beside a white panel holding the Swarthmore College logo
(`assets/swarthmore-logo.pdf`, a vector trace of the official high-resolution JPG
`assets/swarthmore-logo-source.jpg` from swarthmore.edu/communications-office/college-logo,
made by `figures/vectorize_logo.py`; the College offers no vector or reversed version, and
the logo is a registered trademark licensed for College academic documents). All text
starts one inch inside the trim edge, which `verify.py` enforces.
The title matches Amirkhan Aidarkhan's abstract: “Identifiability and Experimental
Design for Neural-Mechanistic Hybrid Models Using Physics-Informed Neural Networks
in Colorectal Cancer.” The initial-condition multiplier
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
run in the `poster_hybrid_errors:build` tmux session and write new timestamped
directories under `builds/`.


## ATRA trajectory plots

The two beta-catenin and HOXA5 panels have white backgrounds and no grid. Native TikZ
annotations show a double-headed arrow from time 40 to 88, with ATRA centered
above it. `figures/forward_plot.py` regenerates the vector curves and the LaTeX
annotation wrapper from `data/forward_trajectories.npz`. Checkpoint hashes,
solver tolerances, and runtime parameters are in the adjacent provenance JSON.
These use the existing 40-observation run; no new training was performed.

## Direct mechanistic/hybrid error comparisons

`figures/compare_hybrid_errors.py` compares the saved full inverse-PINN control
with the β-catenin → MYC and RA → HOXA5 hybrids. It exports parameter-error
plots, regulatory-function NRMSE plots, all-shared-parameter plots, CSV data
and a downloadable gallery. Each model pair is scored on its common 34
parameters. The two fitted regulatory functions are evaluated on the same
saved observed-input grid.

Run in tmux, using a new timestamped output directory:

```bash
python3 research-poster-latex-hybrid/figures/compare_hybrid_errors.py \
  --out runs/<timestamp>_hybrid_error_comparison
```

The 2026-09-17 exports are in
`../runs/20260917_201321_hybrid_error_comparison/`, served at
`http://localhost:8003/hybrid_errors/`. The poster uses a compact four-panel
version of their function and basal-production errors, generated by
`poster_comparison()` in the same script. The gallery retains the complete
equation-parameter and all-shared-parameter comparisons.
