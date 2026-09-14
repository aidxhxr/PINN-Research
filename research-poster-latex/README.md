# First poster — PINN dynamics, parameter recovery and uncertainty

Editable LuaLaTeX poster, one landscape page, **48 × 36 inches**.
Authors: **Nathaniel Kim · Pascal Kataboh**.

1. **Model and formulation:** unchanged regulatory schematic, seven ODEs,
   nondimensionalization, forcing, disease regimes and stemness readout.
2. **Dynamics and PINN reconstruction:** Normal MYC/APC overlaid, untreated
   (`DR=0`) first and treated (`DR=1.5`) second; the corresponding saved inverse
   PINN fits, neural-network diagram, training losses and parameter errors.
3. **Identifiability and Bayesian uncertainty:** full Fisher diagnostics,
   selected posterior marginals from the presentation, and a Fisher analysis
   of eight named parameters with the remaining 28 fixed at their true values.

The second poster retains the neural–mechanistic hybrid and anchor-intervention
focus. Previous versions are preserved in `poster-versions/` in the repository.

## Files and regeneration

- `poster.pdf`: compiled PDF; `preview.png` and `preview-detail.png`: previews.
- `poster.tex`: editable source; `poster-latex-source.zip`: PDF plus all sources
  and numerical data required to regenerate the included figures.
- `assets/model-original.pdf`: original regulatory network, unchanged.
- `assets/pinn-architecture.tex`: native vector network and integral-loss diagram.
- `assets/normal_reference.pdf`, `normal_pinn.pdf`: Normal treatment comparisons.
  Their `*-window.tex` wrappers place editable double-headed ATRA arrows at
  exactly tau=40 and tau=88, with ATRA above the arrow.
- `assets/normal_training.pdf`: saved Adam loss components and parameter errors.
- `assets/fisher.pdf`: normalized Severe FIM and four-regime eigenvalue spectra.
- `assets/presentation_marginals.pdf`: six marginals replotted from the exact
  posterior-sample run used by the presentation.
- `assets/tables.tex`: numerical tables, including the eight-parameter Fisher analysis.
- `SOURCES.md`: numerical provenance, methods and interpretation limits.

Upload the ZIP to Overleaf and choose **LuaLaTeX**. All required generated
assets are included, so no Python step is needed. Locally, compile with:

```bash
lualatex -interaction=nonstopmode -halt-on-error poster.tex
```

To regenerate figures from the packaged arrays, compile, check layout/fonts,
render previews and package the sources:

```bash
bash build.sh
```

Run repository builds in tmux, recording the session in root `AGENTS.md`.
Each build creates a timestamped directory under `builds/`. The standard build
requires NumPy, Matplotlib and PyMuPDF; it does not train or solve an ODE.
`figures/export_normal_comparison.py` is a separate repository-only preparation
step using saved Torch checkpoints and Radau; the packaged arrays avoid needing
Torch or the research repository for ordinary poster builds.

## Reading the figures

Untreated dynamics retain the baseline circadian forcing (`AR=0.04`). Both
MYC and APC use the same concentration axis at their original amplitudes.
The plots show the full trajectories without oscillation insets.

The displayed networks are the **150-time-per-condition integral inverse PINN**
trained on ten conditions with additive noise sigma=0.002. They replace the
previous poster's separate 40-observation forward-PINN example. The Adam loss
and parameter-error plots are saved checkpoints, while the final state fits
also include L-BFGS. Reported state errors use the full [0,150] interval.

Fisher results are local to truth. The reduced eight-parameter analysis fixes
28 parameters at truth and does not demonstrate reduced-parameter PINN recovery.
Bayesian curves are marginals of the full 36-parameter frozen-state HMC run;
effective-sample-size and coverage checks fail, so interval widths are unreliable.
