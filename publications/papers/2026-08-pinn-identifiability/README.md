# PINN approximation and identifiability report (August 2026)

**A Reduced WNT–HOX–Retinoic Acid Model of Colorectal Cancer Stemness and Its
Approximation by Physics-Informed Neural Networks**\
Amirkhan Aidarkhan, Nathaniel Kim and Pascal Kataboh, August 2026.

[PDF](paper.pdf) (52 pages, 47 references) · [source](paper.tex)

The work before the neural–mechanistic hybrids: model formulation and
nondimensionalization, the Radau reference solution, global sensitivity
analysis, forward and inverse PINNs, Fisher-information and profile-likelihood
identifiability, and Bayesian inference with Hamiltonian Monte Carlo. The
current manuscript in [`paper/`](../../../paper/) supersedes it.

| File | Contents |
|---|---|
| `paper.tex`, `paper.pdf`, `references.bib` | Report source, compiled PDF, bibliography |
| `*.png` | Figure snapshots copied from the runs that produced them |
| `schematic.tex` | Standalone TikZ schematic, not included by `paper.tex` |

## Build

```bash
bash build.sh --bib   # pdflatex, bibtex, then two pdflatex passes
bash build.sh         # two pdflatex passes, reusing paper.bbl (git-ignored; run --bib first)
```

`build.sh` reports the page count, overfull or underfull boxes and undefined
references. The catalog adapter `paper` in [`../../catalog.json`](../../catalog.json)
runs the same steps in an isolated copy. The figures are fixed snapshots.

## Status

Historical. Results it shares with the
[result registry](../../../results/registry.json) are recomputed by
`wnt-pinn reproduce`; the rest of the text has not been re-audited. One known
error: the Bayesian section attributes the 19/20/11/8 recovery counts to the
corrected HMC run, but they come from the first, miscalibrated run, and the
corrected run fails its calibration checks, so it supports no recovery count.
The folder was `research-paper/` before the layout change
([`docs/layout.md`](../../../docs/layout.md)).
