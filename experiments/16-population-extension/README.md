# 16 · Population extension: a 14-state ATRA response model

What does a 48-hour ATRA (retinoic acid) exposure do to stem-like and
ALDH-like cell populations? This folder holds a direct SciPy reproduction of
the revised population-dynamics report
`wnt_hox_ra_revised_population_dynamics.pdf` (supplied privately; not in the repository). The report extends the
molecular network with four HOX-program scores and three cell populations,
14 states in total. There are no neural networks, fitted parameters or
copied curves. The reproduction matches the report's Tables 8, 11 and 12 Severe APC
Loss readouts to within 0.001: ATRA lowers the stem-like population by 30%
while raising the reduced HOX index. A Normal-regime run, which the report
does not plot, responds more strongly on every readout except the CYP26A1 fold change.

| | |
|---|---|
| **Status** | complete; maintained (runs through `wnt-pinn run`). A preliminary reproduction: the report's Figs 15-16 and its identifiability analysis are not rerun |
| **Period** | 2026-09 |
| **Builds on** | the seven-state model ([docs/model.md](../../docs/model.md)). This is an independent implementation, because the report revises its parameters, time scales, Hill functions and HOXA13 equation |
| **Paper** | not used |

## Result

Readouts at 88 h (end of exposure), ATRA against a matched untreated control:

| readout | Severe APC Loss | Normal |
|---|---:|---:|
| total population `NT` | -4.30% | -7.73% |
| stem-like population `NS` | -30.36% | -38.31% |
| ALDH-like fraction of all cells `FA` | -21.03% | -24.31% |
| reduced HOX index `IH`, control to ATRA | 0.2863 to 0.4334 (+0.147) | 0.2926 to 0.6318 (+0.339) |
| HOXA5 / HOXA13 / CYP26A1 fold change | 4.116 / 0.677 / 1.389 | 5.770 / 0.320 / 1.040 |
| maintenance index re-crosses 0.5 at | 94.905 h (6.905 h after washout) | 105.530 h (17.530 h after washout) |
| `NS` below control at re-crossing / at 168 h | 31.54% / 22.61% | 42.52% / 34.30% |
| 160 parameter samples keeping lower `FA` and higher `IH` | 160/160 | 160/160 |
| re-crossing time, 5th / median / 95th percentile | 94.450 / 94.922 / 95.504 h | 102.213 / 105.227 / 109.693 h |

The ALDH share of the residual stem pool rises (Severe 17.46% to 18.94%) even
though the absolute ALDH-like population falls, so the denominator matters.
The integration agrees with an independent, tighter Radau solve to a maximum
scaled state difference of 1.91e-8. Some magnitudes in the ablations and the
selective block do not match the report. One example: removing
differentiation gives an `FA` ratio of 1.052 against the report's 1.133. These
gaps are kept and listed in `tables/pdf_comparison.csv` rather than tuned
away.

## Method

- State order `[b, p, h5, h13, m, r, c, uBL, uCL, uBA, uCA, nL, nA, nD]`.
  Time is in hours. Molecular scales are 100, 150, 5, 5, 50, 10 and 25 nM.
  Populations are divided by an assumed carrying capacity of 10^6 cells.
- Readouts: `NT = nL + nA + nD`; `NS = nL + nA`; `FA = nA / NT`;
  `SA = nA / NS`; `IH = (1 + (nL uCL + nA uCA)/NS - H(h13))/2`; maintenance
  index `SSC = Jren / (Jren + Jexit)`. `SSC = 0.5` is exactly zero
  instantaneous net growth of `NS`.
- Every parameter set gets its own untreated 720-hour preconditioning and
  matched 168-hour control and treatment trajectories. Integration is BDF
  (rtol 1e-9, atol 1e-11, max step 0.5 h, output every 0.125 h), with
  continuous threshold detection and positivity and balance audits.
- The default study makes 202 matched comparisons:
  - 4 regimes;
  - a 5 x 5 dose-duration grid;
  - 4 extra schedules or blockades;
  - 7 ablation and Hill-shape variants;
  - 2 solver and clock checks;
  - 160 Latin-hypercube samples, log-uniform multipliers on [0.85, 1.15], seed
    20260915.
- Choices where the report is ambiguous:
  - Eq. (6) is used as printed, without the `dB` multiplier. This reproduces
    the reported beta-catenin ratio of 0.719; a physical-clock variant is
    also run.
  - Other molecular clocks use the exact stated half-lives. Population rates
    are already per hour.
  - The four 6-hour pulses are placed at 40-46, 54-60, 68-74 and 82-88 h.
  - The sampling seed is chosen here because the report gives none.
  - The sensitivity statistic is labelled Spearman, not partial correlation.
  - Crossing times are measured from the simulation start. Exposure begins
    at 40 h.
- The config key `regime` sets the base regime for every scenario except
  the four-regime comparison. The default is Severe APC Loss. Non-default runs
  compare only the report's
  Table 8 row, because every other printed target is a Severe value.

## Contents

| path | purpose |
|---|---|
| `../../src/wnt_pinn/population/model.py` | parameters (report Tables 3-7), forcing, 14-state right-hand side (Eqs 6-23), derived quantities (24-30), selective block (32) |
| `../../src/wnt_pinn/population/simulation.py` | BDF integration, preconditioning, matched controls, threshold detection, audits |
| `../../src/wnt_pinn/population/study.py` | scenario generation, Latin hypercube, Radau cross-check, source snapshot and exports |
| `../../src/wnt_pinn/population/plotting.py` | figures, results PDF and a short presentation brief (`meeting_brief.pdf`) |
| `../../configs/population_extension{,_normal}.json` | solver and sampling settings; `_normal` adds `"regime": "Normal"` |
| `../../tests/test_population_extension.py` | regression tests against the report's printed targets |
| `run_population.py` | standalone entry point, equivalent to `wnt-pinn run` |
| `wnt_hox_ra_revised_population_dynamics.pdf` | the source report; supplied privately and not in the repository |
| `population_extension_results_20260915.zip` | complete Severe run: figures, brief, tables, trajectories, source snapshot |
| `downloads/population_extension_plots_20260915.zip` | the 14 figures (PNG and PDF), a combined PDF and the brief |
| `build_web_preview.py` | writes a static plain-language explanation of the Severe run to `web/` (local only, not in the repository) |
| `LATEST_RUN.txt`, `LATEST_RUN_normal.txt`, `results` | pointers to the local run directories under the git-ignored root `runs/` |

Generated figures follow the report's numbering: `fig02_molecular` to
`fig14_sensitivity` and `fig17_hill_robustness`, 14 in all, plus the four-regime `fig_regimes` (not in the archived 2026-09-15 run). Figure 1 is a
schematic.

## Reproduce

```bash
# from the repository root; CPU only, NumPy/SciPy/Matplotlib
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python3 -m wnt_pinn.cli run configs/population_extension.json          # Severe APC Loss
PYTHONPATH=src python3 -m wnt_pinn.cli run configs/population_extension_normal.json   # Normal, about 80 s
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_population_extension.py'
```

Each run creates a new `runs/<timestamp>_population_extension[_normal]_<id>/`
at the repository root and never overwrites earlier runs. It uses four CPU
workers and has no resume. The run directory contains `RESULTS.md`,
`meeting_brief.pdf`, `all_figures.pdf`, `index.html`, PNG and vector-PDF
figures, CSV tables, `data/trajectories.npz`, resolved parameters, validation
results, dependency versions and a `source/` snapshot. To browse the archived
Severe run without rerunning:

```bash
d=$(mktemp -d)
python3 -m zipfile -e experiments/16-population-extension/population_extension_results_20260915.zip "$d"
xdg-open "$d/population_extension_results/index.html"
```

## Notes

- This is a mechanistic reproduction, not a fit to data or an experimental
  validation. The reduced HOX index is not the clinical eight-gene signature
  and says nothing about patient survival.
- The Latin-hypercube intervals describe sensitivity to parameter
  perturbations. They are not confidence intervals or a posterior.
- The 720-hour warm-up is followed exactly. It does not prove that the slow
  populations are at equilibrium; one-day residual changes are saved.
- In the no-RA-to-HOX ablation, the `IH` change is -0.000376 against the
  report's -0.027. Under the printed equations, no direct RA path remains to
  explain the report's larger value.
- The Normal-regime figures apply the report's Severe protocols unchanged.
  They are new model outputs, not reproductions of printed figures.
