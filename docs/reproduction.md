# Reproducing reported results

Install the project with the `publications` extra using
[the installation guide](installation.md). Run the commands below from the
repository root inside a recorded tmux session. These are saved-result
calculations, separate from [training new experiments](experiments.md).

## Rebuild a table and figure

```bash
wnt-pinn verify
wnt-pinn reproduce --result inverse_recovery --output runs/inverse-report
```

The output contains `inverse_recovery/table.csv`, `table.md`, `figure.png`,
`figure.pdf` and `result.json`. The table recomputes 37 versus 50 recovered
parameter-regime pairs from true and recovered dictionaries, using the same
strict 10% threshold and denominator of 144 as the posters.

For every registered result:

```bash
wnt-pinn reproduce --all --output runs/all-reported-results
```

Use a fresh output path on subsequent runs. `reproduction.json` records the
generated reports. The command fails on altered required inputs or changed
registered claims. Missing optional upstream artifacts produce warnings.

| Result ID | Recomputed quantity |
|---|---|
| `forward_accuracy` | Mean of 28 saved relative state errors per formulation |
| `inverse_recovery` | Full 36-parameter recovery counts, four regimes |
| `hybrid_comparison` | Common-parameter control/hybrid counts and function NRMSE |
| `fisher_information` | Full/reduced eigenvalues and local classifications |
| `anchor_interventions` | Saved MYC/HOXA13 basal errors and function scores |
| `hmc_diagnostics` | Truth coverage, historical ESS and gates from original draws |
| `bayesian_forward` | Reference coverage and RMSE from predictive arrays |
| `normal_treatment` | State errors from saved treated/untreated reference and PINN curves |
| `training_diagnostics` | Recorded weighted losses and W/thetaP errors |
| `constraint_screen` | Range of the historical note's transcribed five-form table |
| `figure_*` | Verified copies of the seven original README figure assets |

See [the registry guide](../results/README.md) for evidence levels. The old
README figures are preserved images; copying them is explicitly distinguished
from generating a new numerical figure. Reproduction generates new inverse
count, Fisher spectrum and Normal state-error figures. Both poster generators
also rebuild their current figures from pinned saved arrays.

## Build publications

```bash
wnt-pinn publications build --id poster_first --output build/review-1
wnt-pinn publications build --id poster_second --output build/review-1
```

These commands copy the declared inputs into isolated directories, run the
original build tools, and save a build manifest with source and output hashes.
Poster PDFs and editable ZIPs appear below each publication's `source/`
directory. Existing build directories with contents are rejected.

For a quicker figure-only check:

```bash
wnt-pinn publications build --id poster_first --figures-only --output build/figures-1
wnt-pinn publications build --id poster_second --figures-only --output build/figures-1
```

Publication IDs also include `paper`, `presentation`, and `network`. The
manuscript and presentation adapters compile their historical snapshots;
their complete scientific narratives have not been re-audited. Their catalog
entries list this narrower validation scope. The current poster builders
check page size, fonts, text bounds, and the preserved network schematic.

The adapters preserve the reviewed files at `research-poster-latex/poster.pdf`
and `research-poster-latex-hybrid/poster.pdf`, which remain linked to
`http://localhost:8003/first/poster.pdf` and
`http://localhost:8003/second/poster.pdf`. To publish an intentional poster edit,
use that project's `build.sh` in its recorded tmux session; it updates the PDF,
previews and editable ZIP together.

## What the checks establish

Hash checks identify the saved inputs. Metric checks reproduce the stated
calculations from those inputs. They do not recover undocumented historical
training environments, prove global identifiability, or validate a biological
model against measurements. Failed inverse-HMC calibration checks remain
failures in the generated reports.

No existing run or checkpoint is rewritten. The full five-constraint run
mapping and the precise checkpoint mapping for the old README forward PNG
remain unknown. The registry records those gaps. New runs save the resolved
configuration, independent seeds, code, environment and outcomes so their
provenance does not depend on narrative notes.
