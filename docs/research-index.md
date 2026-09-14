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
| `PINN-inverse-pinn-boost/` | Integral inverse PINN and saved ten-condition comparison | `inverse_recovery`, `normal_treatment`, `training_diagnostics` |
| `PINN-hybrid-ude/` | Learned regulatory terms and exact-state intervention screens | `hybrid_comparison`, `anchor_interventions` |
| `PINN-fisher-matrix/`, `PINN-fisher-matrix-top8/` | Full and conditional eight-parameter local Fisher analyses | `fisher_information` |
| `PINN-bayesian/` | Frozen-state inverse posterior experiments | `hmc_diagnostics`; reported chains fail calibration checks |
| `PINN-forward-bayesian/` | Forward network-weight HMC and predictive exports | `bayesian_forward` |
| `PINN-smaller/forward-pinn-train-hybrid/` | Sparse forward PINN | `forward_accuracy`, sparse formulation |
| `PINN-smaller/forward_pinn_train/` | Dense supervised interpolation baseline | `forward_accuracy`, dense formulation |

Supported here means that the implementation or saved outputs are used by
current workflows. It does not imply all old launch scripts have the same
seed protocol, defaults, dependencies or resume behavior as the new runner.

## Historical and exploratory work

| Location | Status and use |
|---|---|
| `PINN-inverse-solve/` | Protected historical inverse baseline. Do not edit. |
| `PINN-inverse-multicond/` | Six-condition inverse and classical ODE-fit experiments |
| `PINN-inverse-multicond-excite/` | Ten-condition WNT/MYC excitation comparison; original autodiff source for 37/144 |
| `PINN-inverse-better/`, `PINN-inverse-better-copy/`, `PINN-inverse-multicond-better/` | Earlier optimization and identifiability attempts; retain for historical comparison |
| `PINN-forward/`, `PINN/` | Early implementations, notebooks, and sensitivity-analysis tables |
| `PINN-hybrid-ude/runs/*screen*`, `*protocol*`, `*dose*` | Equation-level or dose-specific screens; inspect their protocol before comparing to joint inverse fits |
| `ROADMAP.md` | Chronological historical narrative, with superseded interpretations |
| `poster-context/` | August planning material, superseded by current poster source notes |
| [Ecology project](https://github.com/aidxhxr/ecosystem-engineer-model) | Separate ecology research topic; outside the WNT package and result registry |

New experiments go in fresh timestamped directories. An old folder's presence
does not establish that every run finished. Check its logs and expected
outputs. The five-form constraint result is a known provenance exception:
the dated note exists, but its complete original run mapping is unavailable.

## Publications and figures

| Location | Status |
|---|---|
| `research-poster-latex/` | Current first poster, Nathaniel Kim and Pascal Kataboh; dynamics, inverse recovery, Fisher and Bayesian results |
| `research-poster-latex-hybrid/` | Current second poster, Amirkhan Aidarkhan and Pascal Kataboh; learned mechanisms and anchor-reaching interventions |
| `research-paper/` | Historical LaTeX manuscript; overlapping numerical results registered, complete text not re-audited |
| `presentation/` | Historical presentation; current first poster uses its original posterior draws with corrected coverage accounting |
| `research-poster/`, `research-poster-v1/` | Earlier publication versions |
| `network-diagram/` | TikZ regulatory network source and reviewed PDF |
| `docs/figures/` | Seven README figure snapshots with original-asset hashes |
| `poster-versions/`, `poster-fable/` | Local proof archives and earlier draft, where available |

[`publications/catalog.json`](../publications/catalog.json) defines the build
inputs, commands, outputs, result IDs and validation scope for the supported
publication adapters. See [reproduction instructions](reproduction.md).

## Decisions retained outside local notes

- [Numerical model and historical compatibility](decisions/0001-model-and-protocols.md)
- [Comparison rules and uncertainty claims](decisions/0002-reported-results.md)
- [MYC and APC learned terms](decisions/0003-hybrid-protocols.md)

Long session narratives remain in dated local `notes/` files. These tracked
decisions contain the conventions needed to interpret and extend the work.
