# Estimator or experiment?

**Estimator or experiment? A Cramér–Rao accounting of parameter recovery by
physics-informed and hybrid neural–mechanistic models of WNT–retinoic
acid–HOX signalling in colorectal cancer**
Amirkhan Aidarkhan and Pascal K. Kataboh, Department of Mathematics and
Statistics, Swarthmore College, Swarthmore, PA, USA

[Manuscript (PDF)](main.pdf) · [Reproduction map](REPRODUCE.md)

## Summary

When a physics-informed neural network (PINN) or a neural–mechanistic hybrid
returns wrong parameter values, either the estimator failed or the experiment
did not contain the information, and the two call for opposite remedies. The
paper separates them with the Cramér–Rao bound of the experiment actually
performed, applied to a seven-state model of WNT, retinoic acid and HOX
signalling across four regimes, from healthy crypt to severe APC loss. For
most parameters the information is present: the bound predicts near-complete
recovery, a converged maximum-likelihood fit attains it where the data are
informative, and the inverse PINNs fall well short. A Schur-complement
Gauss–Newton PINN closes that gap wherever its continuation converges. Where
the bound itself is wide, as for APC functionality when β-catenin saturates
or for the basal rate next to a learned regulatory term, no estimator helps;
a targeted knockdown restores most of the missing information, while
information-matched conditions do not.

## Build

```bash
cd paper && ./build.sh
```

`main.tex` includes `sections/*.tex`; the bibliography is `references.bib`.
`build.sh` runs `pdflatex`, `bibtex` and two more `pdflatex` passes, writes
`build.log` and lists undefined citations or references. It needs TeX Live
with the TeX Gyre fonts; `mutool`, if present, prints the page count.

## Figures

Each figure has a script, `figures/make_figNN.py`, that redraws it from saved
run outputs without training anything. From the repository root:

```bash
PYTHONPATH=src python3 paper/figures/make_fig03.py
```

A script writes `figNN_<name>.{pdf,png}` and `figNN_<name>_audit.json` into
`figures/`; the audit file lists every plotted number with the run artifact it
comes from. The Fig 1 network panel, `figures/fig01a_network.tex`, is
standalone TikZ (`pdflatex`). [`REPRODUCE.md`](REPRODUCE.md) maps every
figure, table and in-text number to its script and runs.

## Paths cited in the manuscript

Paths in the manuscript refer to the repository at git tag
[`paper-v1`](https://github.com/aidxhxr/PINN-Research/tree/paper-v1), the state
the paper was built from. Since then this folder has moved from
`research-paper-final/` to `paper/`, and experiment folders to
`experiments/NN-name/` (for example, `PINN-sensitivity/` is now
`experiments/17-sensitivity-pinn/`). Check out the tag to follow a cited path
as written, or translate it with [`docs/layout.md`](../docs/layout.md).
`REPRODUCE.md` and the figure scripts use the current layout.
