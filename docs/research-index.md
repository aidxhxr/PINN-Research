# Research index

Use this index to find the implementation, protocol and evidence behind a
result. Folder names are retained so historical run paths and publication
links remain valid. Active numerical implementations now share
`src/wnt_pinn/`; [registered results](../results/registry.json) identify the
saved runs used in current claims.

## Supported work

| Location | Purpose | Result/protocol |
|---|---|---|
| `src/wnt_pinn/`, `configs/`, `experiments/` | Shared model, supported experiment runner, reproducibility tools | [Experiment guide](experiments.md) |
| `experiments/10-inverse-integral-residual/` | Integral inverse PINN and saved ten-condition comparison | `inverse_recovery`, `normal_treatment`, `training_diagnostics` |
| `experiments/15-hybrid-ude/` | Learned regulatory terms and exact-state intervention screens | `hybrid_comparison`, `anchor_interventions` |
| `experiments/11-fisher-information/`, `experiments/12-fisher-top8/` | Full and conditional eight-parameter local Fisher analyses | `fisher_information` |
| `experiments/13-bayesian-inverse/` | Frozen-state inverse posterior experiments | `hmc_diagnostics`; reported chains fail calibration checks |
| `experiments/14-bayesian-forward/` | Forward network-weight HMC and predictive exports | `bayesian_forward` |
| `experiments/04-forward-pinn/forward-pinn-train-hybrid/` | Sparse forward PINN | `forward_accuracy`, sparse formulation |
| `experiments/04-forward-pinn/forward_pinn_train/` | Dense supervised interpolation baseline | `forward_accuracy`, dense formulation |

Supported here means that the implementation or saved outputs are used by
current workflows. It does not imply all old launch scripts have the same
seed protocol, defaults, dependencies or resume behavior as the new runner.

## Historical and exploratory work

| Location | Status and use |
|---|---|
| `experiments/05-inverse-single-condition/` | Protected historical inverse baseline. Do not edit. |
| `experiments/07-inverse-six-condition/` | Six-condition inverse and classical ODE-fit experiments |
| `experiments/09-inverse-excitation/` | Ten-condition WNT/MYC excitation comparison; original autodiff source for 37/144 |
| `experiments/06-inverse-three-condition/`, `experiments/08-inverse-small-networks/` | Earlier optimization and identifiability attempts; retain for historical comparison |
| `experiments/01-model-prototypes/`, `experiments/02-original-pinn/`, `experiments/03-global-sensitivity/` | Early prototypes, the original PINN notebooks, and sensitivity-analysis tables |
| `experiments/15-hybrid-ude/runs/*screen*`, `*protocol*`, `*dose*` | Equation-level or dose-specific screens; inspect their protocol before comparing to joint inverse fits |
| `ROADMAP.md` | Chronological historical narrative, with superseded interpretations |
| [Ecology project](https://github.com/aidxhxr/ecosystem-engineer-model) | Separate ecology research topic; outside the WNT package and result registry |

New experiments go in fresh timestamped directories. An old folder's presence
does not establish that every run finished. Check its logs and expected
outputs. The five-form constraint result is a known provenance exception:
the dated note exists, but its complete original run mapping is unavailable.

## Publications and figures

| Location | Status |
|---|---|
| `publications/posters/pinn-recovery/` | Current first poster, Nathaniel Kim and Pascal Kataboh; dynamics, inverse recovery, Fisher and Bayesian results |
| `publications/posters/hybrid-models/` | Current second poster, Amirkhan Aidarkhan and Pascal Kataboh; learned mechanisms and anchor-reaching interventions |
| `publications/papers/2026-08-pinn-identifiability/` | Historical LaTeX manuscript; overlapping numerical results registered, complete text not re-audited |
| `publications/slides/archive/beamer-dark/` | Historical presentation; current first poster uses its original posterior draws with corrected coverage accounting |
| `publications/posters/archive/2026-08-typst/`, `publications/posters/archive/2026-09-typst-v1/` | Earlier publication versions |
| `network-diagram/` | TikZ regulatory network source and reviewed PDF |
| `docs/figures/` | Seven README figure snapshots with original-asset hashes |

[`publications/catalog.json`](../publications/catalog.json) defines the build
inputs, commands, outputs, result IDs and validation scope for the supported
publication adapters. See [reproduction instructions](reproduction.md).

## Decisions retained outside local notes

- [Numerical model and historical compatibility](decisions/0001-model-and-protocols.md)
- [Comparison rules and uncertainty claims](decisions/0002-reported-results.md)
- [MYC and APC learned terms](decisions/0003-hybrid-protocols.md)

Long session narratives remain in dated local `notes/` files. These tracked
decisions contain the conventions needed to interpret and extend the work.
