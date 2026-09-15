# SciPy reproduction of the 14-state population extension

This is a direct numerical reproduction of
`wnt_hox_ra_revised_population_dynamics.pdf`, supplied for the population extension.
SciPy integrates the ODEs; NumPy holds arrays and Matplotlib draws the figures.
There are no neural networks, fitted parameters, or copied curves.

The implementation is independent of the existing seven-state model because the
PDF revises its parameters, time scales, Hill functions, and HOXA13 equation.
The historical `PINN-inverse-solve/` baseline is untouched.

## Run

From the repository root, with Python 3.11+ and NumPy, SciPy, and Matplotlib installed:

```bash
tmux new-session -s population_reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python3 -m wnt_pinn.cli run configs/population_extension.json
```

The standalone equivalent is `python3 extension/run_population.py`.
Record every new tmux session in root `AGENTS.md`. Each invocation creates a fresh
timestamped directory under root `runs/`; it never overwrites an earlier run.
`LATEST_RUN.txt` records the most recently completed directory. Four CPU workers
are used by default. The configuration includes the solver settings, sample count,
and random seed. There is no checkpoint resume; reruns get new directories.

The default study performs 202 matched comparisons: four regimes, 25 dose-duration
conditions, four extra schedules/blockades, seven ablation/Hill variants, two solver/
clock checks, and 160 parameter samples. Every parameter set gets its own untreated
720-hour preconditioning and matched 168-hour control/treatment trajectories.

## Implementation and model map

| Python file | Purpose and PDF equations |
|---|---|
| `src/wnt_pinn/population/model.py` | Fully documented parameters (Tables 3–7), state order, forcing, 14-state RHS (6–23), derived quantities (24–30), selective block (32) |
| `src/wnt_pinn/population/simulation.py` | BDF integration, preconditioning, matched controls, continuous threshold detection, balance and positivity audits |
| `src/wnt_pinn/population/study.py` | Deterministic scenario generation, 160-sample Latin hypercube, independent Radau comparison, source snapshots and numerical exports |
| `src/wnt_pinn/population/plotting.py` | Figures and automatically populated results/presentation PDFs |
| `configs/population_extension.json` | Reproducible solver and sampling configuration |

The exact state order is:

```text
[b, p, h5, h13, m, r, c, uBL, uCL, uBA, uCA, nL, nA, nD]
```

Time is in hours. The molecular scales are 100, 150, 5, 5, 50, 10, 25 nM,
respectively. Program scores are dimensionless. Populations are divided by
the assumed carrying capacity of one million cells. No scale represents
independently measured experimental abundances in this reproduction.

The key readouts deliberately use different denominators:

- Total abundance: `NT = nL + nA + nD`.
- Absolute stem-like abundance: `NS = nL + nA`.
- ALDH-like fraction of all cells: `FA = nA / NT`.
- ALDH-like share of the residual stem pool: `SA = nA / NS`.
- Reduced HOX index: `IH = (1 + (nL*uCL + nA*uCA)/NS - H(h13))/2`.
- Maintenance index: `SSC = Jren/(Jren+Jexit)`, with `SSC=0.5` exactly
  equivalent to zero instantaneous net growth of `NS`.

## Reproduction scope

| PDF figure | Generated file stem | Status |
|---|---|---|
| 1 | — | Regulatory schematic; not a numerical reproduction target |
| 2 | `fig02_molecular` | Seven molecular states and HOXC programs |
| 3 | `fig03_population` | Populations, fractions, and HOX index |
| 4 | `fig04_central_result` | ALDH/HOX response separation |
| 5 | `fig05_maintenance` | Fluxes, threshold and recovery |
| 6 | `fig06_timing_ensemble` | Crossing-time perturbation distribution |
| 7 | `fig07_endpoint` | End-of-exposure response ratios |
| 8 | `fig08_dose_duration` | 5 × 5 dose-duration scan |
| 9 | `fig09_grid_maintenance` | Maintenance during/after exposure |
| 10 | `fig10_duration` | 24, 48, 60, 72-hour trajectories |
| 11 | `fig11_schedules` | Equal-area schedules; pulse endpoints assumed below |
| 12 | `fig12_selective_block` | Counterfactual HOX-arm inhibition |
| 13 | `fig13_ablations` | Three mechanism removals |
| 14 | `fig14_sensitivity` | LHS scatter and Spearman correlations |
| 15–16 | — | Equilibrium/periodic-attractor multi-start scans not rerun in this preliminary package |
| 17 | `fig17_hill_robustness` | Nominal plus four alternative Hill shapes |

The numerical identifiability analysis is also not rerun. The outputs do not
claim that the independently implemented system has been proved monostable.
The figure layouts are newly rendered; no visual pixel matching is attempted.
`tables/pdf_comparison.csv` compares individual numerical targets. Some ablation
and selective-block magnitudes do not match the report; these discrepancies are
retained and discussed in the generated results, rather than tuned away.

## Explicit choices where the PDF is ambiguous

1. **Beta-catenin clock.** Printed Eq. (6) has no `dB` multiplier despite the
   physical-time statement. The primary configuration follows the printed equation;
   a separate variant multiplies by `dB=ln(2)/(5/6)` and quantifies its effect.
   The printed convention reproduces the reported beta-catenin ratio of 0.719.
2. **Exact molecular clocks.** Other rates use the stated half-lives rather than
   rounded four-digit rates. Population rates in Table 6 are already per hour;
   they are not multiplied by `dN` again.
3. **Pulse endpoints.** Four twofold 6-h pulses are placed at 40–46, 54–60,
   68–74, and 82–88 h. The report does not explicitly give these four endpoints.
4. **Sampling seed.** No numerical uncertainty seed was found in the PDF. This
   run explicitly uses 20260915. Multipliers are log-uniform on [0.85, 1.15].
   The ten ASCII parameter names are mapped in the `Parameters` docstring.
5. **Sensitivity statistic.** The PDF's plotted ordinary Spearman correlation
   is reproduced as Spearman correlation, not mislabeled as a partial correlation.
6. **Time origin.** Exposure starts at 40 h. Crossing near 94.9 h therefore means
   about 54.9 h after treatment start and 6.9 h after washout at 88 h.
7. **Preconditioning.** The prescribed finite 720-h warm-up is used. One-day
   residual population changes are saved, rather than claiming exact equilibrium.

## Output files

The checked-in `population_extension_results_20260915.zip` contains the completed
202-comparison run, including all plots, the meeting brief, numerical tables,
trajectories and the original source snapshot. To browse it from a fresh checkout
without rerunning simulations, extract it into a new directory:

```bash
population_preview=$(mktemp -d)
python3 -m zipfile -e extension/population_extension_results_20260915.zip "$population_preview"
xdg-open "$population_preview/population_extension_results/index.html"
```

`results` is a relative symlink to the original local directory under ignored
`runs/`; `LATEST_RUN.txt` records that directory. The directory is available only
on the machine that ran the study until its archived contents are restored there.
The archive's source snapshot and hashes describe the completed run; maintained
source files may subsequently receive formatting or documentation fixes.

The running localhost preview serves:

- `http://localhost:8003/extension_plots/index.html`: all 14 plots.
- `http://localhost:8003/extension_plots/all_figures.pdf`: complete figure PDF.
- `http://localhost:8003/extension_plots/meeting_brief.pdf`: six-page meeting brief.
- `http://localhost:8003/extenstion_report/html`: plain-language explanation,
  speaking script, figure interpretations, results and reproduction discrepancies.

The report route intentionally retains the requested spelling `extenstion`;
`/extension_report/html` also works. Rebuild the static explanation against the
latest completed results with the existing preview document root:

```bash
python3 extension/build_web_preview.py \
  --server-root notes/poster-preview_20260914_172407
```

Run this in the recorded tmux session. The script writes a dated HTML snapshot
under `extension/web/` and updates the preview aliases; it does not rerun simulations.

Each successful run contains `meeting_brief.pdf`, `all_figures.pdf`, `RESULTS.md`,
`index.html`, individual PNG/vector-PDF files under `figures/`, CSV tables,
raw trajectories, complete resolved parameters, source hashes, dependency versions,
and numerical validation results. `source/` preserves the code actually used.

Open `index.html` locally to browse all plots. A compressed array can be read with:

```python
import numpy as np
from pathlib import Path
run = Path('extension/LATEST_RUN.txt').read_text().strip()
data = np.load(Path(run) / 'data/trajectories.npz', allow_pickle=False)
t = data['regime:Severe APC loss__t']
y = data['regime:Severe APC loss__treated']  # shape: 14 × 1345
```

This is a mechanistic reproduction, not a fit to raw data or an independent
experimental validation. The reduced HOX index is not the clinical eight-gene
signature; it cannot be used here to infer effects on patient survival.
