# Repository layout

The repository was reorganised in October 2026: experiment folders moved under
`experiments/`, publications under `publications/`, and the final manuscript
to `paper/`. Folders were moved whole. Their contents are unchanged except
for the path references needed to keep code working and new README files.

## Tree

```text
PINN-Research/
├── README.md, LICENSE, LICENSE-CC-BY-4.0.txt, CITATION.cff, CONTRIBUTING.md
├── pyproject.toml, Makefile, requirements.txt, requirements-lock.txt
├── src/wnt_pinn/          shared model, networks, experiment runner, reporting
├── configs/               runner profiles (JSON)
├── scripts/               runner and benchmark entry points
├── tests/                 numerical, registry and archive checks
├── experiments/           one self-contained folder per experiment
│   ├── 01-model-prototypes/
│   ├── 02-original-pinn/
│   ├── 03-global-sensitivity/
│   ├── 04-forward-pinn/
│   ├── 05-inverse-single-condition/
│   ├── 06-inverse-three-condition/
│   ├── 07-inverse-six-condition/
│   ├── 08-inverse-small-networks/
│   ├── 09-inverse-excitation/
│   ├── 10-inverse-integral-residual/
│   ├── 11-fisher-information/
│   ├── 12-fisher-top8/
│   ├── 13-bayesian-inverse/
│   ├── 14-bayesian-forward/
│   ├── 15-hybrid-ude/
│   ├── 16-population-extension/
│   └── 17-sensitivity-pinn/
├── paper/                 final manuscript, figure scripts, REPRODUCE.md
├── publications/          posters, slides, the earlier report, shared figures
│   ├── catalog.json       build recipes
│   ├── papers/            2026-08-pinn-identifiability/
│   ├── posters/           pinn-recovery/, hybrid-models/, archive/
│   ├── slides/            department-talk/, archive/
│   └── figures/           network-diagram/
├── results/               result registry and its saved inputs
├── artifacts/             catalog of archived run outputs
├── docs/                  guides, decisions, figures, performance records, history
└── .github/               continuous-integration workflows
```

`runs/` at the root holds output of the `wnt-pinn` runner and is git-ignored.
See [`experiments/README.md`](../experiments/README.md) and
[`publications/README.md`](../publications/README.md) for what each folder
contains.

## Old to new paths

### Experiments

| Old path | New path |
|---|---|
| `matlab_code/` | `experiments/01-model-prototypes/matlab/` |
| `matlab_scripts/` | `experiments/01-model-prototypes/matlab/practice/` |
| `notebooks/` | `experiments/01-model-prototypes/notebooks/` |
| `Untitled.ipynb` | `experiments/01-model-prototypes/notebooks/Untitled.ipynb` |
| `hox_comparison.png`, `hox_components.png`, `wnt_comparison.png`, `wnt_dynamics.png` | `experiments/01-model-prototypes/figures/` |
| `atra_superimpose_plots/` | `experiments/01-model-prototypes/figures/atra-superimpose/` |
| `PINN/` | `experiments/02-original-pinn/` |
| `SA-big/` | `experiments/03-global-sensitivity/` |
| `PINN-smaller/` | `experiments/04-forward-pinn/` |
| `PINN-inverse-solve/` | `experiments/05-inverse-single-condition/` |
| `PINN-inverse-better/` | `experiments/06-inverse-three-condition/` |
| `PINN-inverse-multicond/` | `experiments/07-inverse-six-condition/` |
| `PINN-inverse-multicond-better/` | `experiments/08-inverse-small-networks/` |
| `PINN-inverse-multicond-excite/` | `experiments/09-inverse-excitation/` |
| `PINN-inverse-pinn-boost/` | `experiments/10-inverse-integral-residual/` |
| `PINN-fisher-matrix/` | `experiments/11-fisher-information/` |
| `PINN-fisher-matrix-top8/` | `experiments/12-fisher-top8/` |
| `PINN-bayesian/` | `experiments/13-bayesian-inverse/` |
| `PINN-forward-bayesian/` | `experiments/14-bayesian-forward/` |
| `PINN-hybrid-ude/` | `experiments/15-hybrid-ude/` |
| `extension/` | `experiments/16-population-extension/` |
| `PINN-sensitivity/` | `experiments/17-sensitivity-pinn/` |
| `experiments/run.py`, `experiments/benchmark_kernels.py`, `experiments/benchmark_reduction.py` | `scripts/run.py`, `scripts/benchmark_kernels.py`, `scripts/benchmark_reduction.py` |

A path inside a moved folder keeps its tail: for example
`PINN-sensitivity/runs/20261017_paper/` is now
`experiments/17-sensitivity-pinn/runs/20261017_paper/`.

### Publications and history

| Old path | New path |
|---|---|
| `research-paper-final/` | `paper/` |
| `research-paper/` | `publications/papers/2026-08-pinn-identifiability/` |
| `research-poster-latex/` | `publications/posters/pinn-recovery/` |
| `research-poster-latex-hybrid/` | `publications/posters/hybrid-models/` |
| `research-poster/` | `publications/posters/archive/2026-08-typst/` |
| `research-poster-v1/` | `publications/posters/archive/2026-09-typst-v1/` |
| `presentation-codex/` | `publications/slides/department-talk/` |
| `presentation/` | `publications/slides/archive/beamer-dark/` |
| `network-diagram/` | `publications/figures/network-diagram/` |
| `ROADMAP.md`, `ROADMAP.tex`, `ROADMAP.pdf` | `docs/history/roadmap.md`, `docs/history/roadmap.tex`, `docs/history/roadmap.pdf` |

Not moved: `src/`, `tests/`, `configs/`, `docs/`, `results/`, `artifacts/`,
`publications/catalog.json`, `publications/README.md`, `.github/` and the
root files.

## Paths cited in the paper

The manuscript in [`paper/`](../paper/) cites paths in the old layout
([`paper/REPRODUCE.md`](../paper/REPRODUCE.md) uses the new one). Git tag
[`paper-v1`](https://github.com/aidxhxr/PINN-Research/tree/paper-v1) preserves
that layout: after `git checkout paper-v1`, cited paths read as written (a few
run directories named in the S3 Table caption were never committed). On later
commits, translate a cited path with the
tables above. Older run manifests, logs and figure audit files also record
paths in the old layout; the same translation applies.
