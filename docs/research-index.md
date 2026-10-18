# Research index

Use this index to find the implementation, protocol and evidence behind a
result. Experiments live in `experiments/NN-name/` (see the
[experiments index](../experiments/README.md)); [`layout.md`](layout.md) maps
the folder names used before October 2026. Active numerical implementations
share `src/wnt_pinn/`, and the [registered results](../results/registry.json)
identify the saved runs used in current claims.

## Supported work

| Location | Purpose | Result/protocol |
|---|---|---|
| `src/wnt_pinn/`, `configs/`, `scripts/` | Shared model, supported experiment runner, reproducibility tools | [Experiment guide](experiments.md) |
| `paper/` | Final manuscript, figure scripts with audit files | [`paper/REPRODUCE.md`](../paper/REPRODUCE.md) |
| `experiments/17-sensitivity-pinn/` | Cramér–Rao bound of the benchmark design, converged maximum likelihood, SGN-PINN | Paper Figs 3–7, S1 |
| `experiments/10-inverse-integral-residual/` | Integral inverse PINN, saved ten-condition comparison, matched ablation scripts | `inverse_recovery`, `normal_treatment`, `training_diagnostics`; paper Fig 5 |
| `experiments/15-hybrid-ude/` | Learned regulatory terms, equation-level screens, depletion and dose-response designs, hybrid Fisher bound | `hybrid_comparison`, `anchor_interventions`; paper Figs 8–11 |
| `experiments/11-fisher-information/`, `experiments/12-fisher-top8/` | Full and conditional eight-parameter local Fisher analyses; `θP` design study | `fisher_information`; paper Fig 7 |
| `experiments/13-bayesian-inverse/` | Frozen-state inverse posterior experiments | `hmc_diagnostics`; reported chains fail calibration checks |
| `experiments/14-bayesian-forward/` | Forward network-weight HMC and predictive exports | `bayesian_forward` |
| `experiments/04-forward-pinn/forward-pinn-train-hybrid/` | Sparse forward PINN | `forward_accuracy`, sparse formulation; paper Fig 2 |
| `experiments/04-forward-pinn/forward_pinn_train/` | Dense supervised interpolation baseline | `forward_accuracy`, dense formulation |
| `experiments/16-population-extension/` | SciPy reproduction of the 14-state population model | `wnt-pinn run configs/population_extension.json` |

Supported here means that the implementation or saved outputs are used by
current workflows. It does not imply all old launch scripts have the same
seed protocol, defaults, dependencies or resume behavior as the new runner.

## Historical and exploratory work

| Location | Status and use |
|---|---|
| `experiments/05-inverse-single-condition/` | Protected historical inverse baseline. Do not edit. |
| `experiments/07-inverse-six-condition/` | Six-condition inverse and classical ODE-fit experiments |
| `experiments/09-inverse-excitation/` | Ten-condition WNT/MYC excitation comparison; original autodiff source for 37/144 |
| `experiments/06-inverse-three-condition/`, `experiments/08-inverse-small-networks/` | Earlier optimization and identifiability attempts; retained for historical comparison |
| `experiments/01-model-prototypes/`, `experiments/02-original-pinn/`, `experiments/03-global-sensitivity/` | Early prototypes, the original PINN notebooks, and sensitivity-analysis tables |
| `experiments/15-hybrid-ude/runs/*screen*`, `*protocol*`, `*dose*` | Equation-level or dose-specific screens; inspect their protocol before comparing to joint inverse fits |
| `docs/history/roadmap.md` | Chronological historical narrative, with superseded interpretations |
| [Ecology project](https://github.com/aidxhxr/ecosystem-engineer-model) | Separate ecology research topic; outside the WNT package and result registry |

New experiments go in fresh timestamped directories. An old folder's presence
does not establish that every run finished. Check its logs and expected
outputs. The first five-form constraint table (registry `constraint_screen`)
is transcribed from a dated note whose run mapping is incomplete; a complete
rerun of all five forms is tracked in
`experiments/15-hybrid-ude/runs/20260921_screen_5form/`.

## Publications and figures

| Location | Status |
|---|---|
| `paper/` | Final manuscript, Aidarkhan and Kataboh |
| `publications/posters/pinn-recovery/` | Current first poster, Nathaniel Kim, Amirkhan Aidarkhan and Pascal K. Kataboh; dynamics, inverse recovery, Fisher and Bayesian results |
| `publications/posters/hybrid-models/` | Current second poster, Amirkhan Aidarkhan and Pascal K. Kataboh; identifiability and experimental design for neural–mechanistic hybrids |
| `publications/papers/2026-08-pinn-identifiability/` | Historical LaTeX manuscript; overlapping numerical results registered, complete text not re-audited |
| `publications/slides/department-talk/` | Department talk, July 2026 |
| `publications/slides/archive/beamer-dark/` | Historical presentation; current first poster uses its original posterior draws with corrected coverage accounting |
| `publications/posters/archive/2026-08-typst/`, `publications/posters/archive/2026-09-typst-v1/` | Earlier poster versions |
| `publications/figures/network-diagram/` | TikZ regulatory network source and reviewed PDF |
| `docs/figures/` | Seven README figure snapshots with original-asset hashes |

[`publications/catalog.json`](../publications/catalog.json) defines the build
inputs, commands, outputs, result IDs and validation scope for the supported
publication adapters. See [reproduction instructions](reproduction.md).

## Decisions

- [Numerical model and historical compatibility](decisions/0001-model-and-protocols.md)
- [Comparison rules and uncertainty claims](decisions/0002-reported-results.md)
- [MYC and APC learned terms](decisions/0003-hybrid-protocols.md)
- [Artifact archives](decisions/0004-artifact-archives.md)

These tracked decisions contain the conventions needed to interpret and
extend the work.
