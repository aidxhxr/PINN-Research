# Poster B — figure manifest (v3, as built)

The poster is a project walkthrough. **Nine of its eleven figure slots reuse
artwork that already existed in the repo**; only three matplotlib scripts still
run at build time. Prepared copies live in `research-poster/assets/ready/`
(sources are never modified); generated ones in `research-poster/assets/`.

## Reused, prepared into `assets/ready/`

| file | source | preparation |
|---|---|---|
| `schema.svg` | `network-diagram/schematic-better.pdf` | PDF→SVG (`mutool`), **no other change** |
| `forward_arch.svg` | `presentation-codex/output/latex/assets/forward_architecture.pdf` | PDF→SVG |
| `inverse_arch.svg` | `presentation-codex/output/latex/assets/inverse_architecture.pdf` | PDF→SVG |
| `forward_fit.png` | `research-paper/forward_pinn_hybrid.png` | internal title cropped (poster supplies its own) |
| `inv_recovery_top.png` | `research-paper/inv_recovery_bars_best8.png` | title cropped, then **top row of 3 panels only** — all 8 were illegible at column width |
| `bayes_marginals.png` | `PINN-bayesian/runs/20260713_204442_bayes/bayes_W_thetaP.png` | suptitle cropped |
| `bayes_miscal_top.png` | `research-paper/bayes_worst8_severe.png` | suptitle cropped, then **top row of 4 panels** — 8 panels made the axis numbers unreadable |
| *(unused)* `fim_spectra.png` | `research-paper/fim_cross_regime_spectra.png` | prepared, then dropped: 93 dpi at column width. Replaced by the vector `fig_fim.svg` |
| *(unused)* `reference_dynamics.png` | `research-paper/scipy_solutions.png` | prepared, then dropped: source is 504×300 px ⇒ ~50 dpi at column width. The Radau reference is visible anyway as the solid curves in `forward_fit.png` |

## Generated (three scripts in `figures/`)

| file | script | data |
|---|---|---|
| `fig_posterior.svg` | `fig_posterior.py` | `PINN-bayesian/runs/20260713_204442_bayes/Normal_posterior_samples.npz` |
| `fig_fim.svg` | `fig_fim.py` | `PINN-fisher-matrix/runs/20260711_203325_fisher/*_fim_summary.json` + `*_correlated_pairs.csv` |
| `fig_support.svg` | `fig_support.py` | `PINN-hybrid-ude/runs/20260802_anchor_reach/anchor_reach.txt` |
| `fig_dose_compact.svg` | `fig_dose_compact.py` | `PINN-hybrid-ude/runs/20260802_dose_response/dose_*.json` |

Retired but kept in the repo: `fig_regimes.py`, `fig_param_fail.py`,
`fig_attribution.py`, `fig_prospective.py`, `fig_dose.py` (the full four-panel
version), `schematic_poster.tex`.

## A correction the rebuild forced

`fig_posterior` originally annotated the `δ_P1`–`θ_P` degeneracy with
**corr = 0.998**. That figure came from the *smoke test* recorded in the
2026-07-07 note, not from the production run. In the production run
(`20260713_204442_bayes`, Normal) the **linear** correlation is **0.321** —
because the valley is hyperbolic, and a linear correlation understates a curved
one (in log space it is −0.99).

The figure and the poster now quote the honest statistic instead: the product
`δ_P1(1−θ_P)` has relative sd **0.19** against **0.91** for `δ_P1` alone, i.e.
the product is **4.7× better constrained** than the parameter. Do not
reintroduce 0.998 anywhere.

## Poster-scale legibility

Figure text must be ≥ 20 pt at printed size. Two ready figures failed this by
being multi-panel rather than low-resolution, and both were fixed by cropping to
one row rather than by scaling. Two others were dropped outright for resolution.
When a ready figure is placed at less than ~80% of column width, re-check its
smallest text before accepting it.
