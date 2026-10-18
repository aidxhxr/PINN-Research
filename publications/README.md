# Publications

Posters, slide decks, the earlier report and shared figures from this
project. The current manuscript is in [`../paper/`](../paper/).

## Index

| Item | What | Authors | Status | Files |
|---|---|---|---|---|
| [`papers/2026-08-pinn-identifiability`](papers/2026-08-pinn-identifiability/) | *A Reduced WNT–HOX–Retinoic Acid Model of Colorectal Cancer Stemness and Its Approximation by Physics-Informed Neural Networks*: model, sensitivity analysis, forward and inverse PINNs, Fisher and profile-likelihood identifiability, Bayesian PINN; work before the hybrids (52 pp., 47 references) | Amirkhan Aidarkhan, Nathaniel Kim, Pascal Kataboh | Historical manuscript, August 2026 (catalog `paper`) | [PDF](papers/2026-08-pinn-identifiability/paper.pdf) · [source](papers/2026-08-pinn-identifiability/paper.tex) |
| [`posters/pinn-recovery`](posters/pinn-recovery/) | First poster, *Physics-Informed Neural Networks for Forward Modeling and Parameter Inference in a Reduced WNT-RA-HOX Model of Colorectal Cancer Initiation*: dynamics, PINN reconstruction, parameter recovery, Fisher and Bayesian uncertainty; 48 × 36 in | Nathaniel Kim, Amirkhan Aidarkhan, Pascal K. Kataboh | Current (catalog `poster_first`) | [PDF](posters/pinn-recovery/poster.pdf) · [source](posters/pinn-recovery/poster.tex) |
| [`posters/hybrid-models`](posters/hybrid-models/) | Second poster, *Identifiability and Experimental Design for Neural-Mechanistic Hybrid Models Using Physics-Informed Neural Networks in Colorectal Cancer*: model, PINN identifiability, learned MYC activation against a mechanistic control, designed knockdowns; 48 × 36 in | Amirkhan Aidarkhan; advisor Pascal K. Kataboh | Current (catalog `poster_second`) | [PDF](posters/hybrid-models/poster.pdf) · [source](posters/hybrid-models/poster.tex) |
| [`posters/archive/2026-08-typst`](posters/archive/2026-08-typst/) | *Physics-Informed Neural Networks and Neural-Mechanistic Hybrids for a WNT–RA–HOX Model of Colorectal Cancer Stemness*: four-column Typst poster walking the whole project; 48 × 36 in | Amirkhan Aidarkhan, Pascal Kataboh | Archived, August 2026 | [PDF](posters/archive/2026-08-typst/poster.pdf) · [source](posters/archive/2026-08-typst/poster.typ) |
| [`posters/archive/2026-09-typst-v1`](posters/archive/2026-09-typst-v1/) | Same title; shorter rebuild with an abstract, display equations and vector figures; 48 × 36 in | Amirkhan Aidarkhan, Pascal Kataboh | Archived, September 2026 | [PDF](posters/archive/2026-09-typst-v1/poster-v1.pdf) · [source](posters/archive/2026-09-typst-v1/poster-v1.typ) |
| [`slides/department-talk`](slides/department-talk/) | *Physics-Informed Neural Network for a Reduced WNT-HOX-Retinoic Acid Model of Colorectal Cancer*: department talk, 21 July 2026; 20-slide Beamer reconstruction of the talk | Amirkhan Aidarkhan, Nathaniel Kim; advisor Pascal K. Kataboh | Final | [PDF](slides/department-talk/output/latex/pinn_presentation_native.pdf) · [source](slides/department-talk/output/latex/pinn_presentation_native.tex) |
| [`slides/archive/beamer-dark`](slides/archive/beamer-dark/) | Same talk, earlier dark-theme Beamer rebuild | Amirkhan Aidarkhan, Nathaniel Kim; advisor Pascal K. Kataboh | Superseded (catalog `presentation`) | [PDF](slides/archive/beamer-dark/pinn_presentation.pdf) · [source](slides/archive/beamer-dark/pinn_presentation.tex) |
| [`figures/network-diagram`](figures/network-diagram/) | TikZ WNT–RA–HOX regulatory network used by the posters and slide decks | — | Current asset (catalog `network`) | [PDF](figures/network-diagram/schematic-better.pdf) · [source](figures/network-diagram/schematic-better.tex) |

Authors are as printed on each item (the slide decks print the advisor as
“Dr. Kataboh”); the network diagram carries no author
line. Each folder has a README with its contents, build command and known
limitations.

## Build recipes

[`catalog.json`](catalog.json) registers the two current posters, the
historical manuscript and presentation, and the regulatory network. For each
it records the source directory, result dependencies, input globs, build
commands, outputs and validation scope.

```bash
wnt-pinn publications build --id poster_first --output build/poster-review
wnt-pinn publications build --id poster_second --output build/poster-review
```

Each adapter makes an isolated copy of the source, verifies the registered
result inputs, compiles the copy and saves a build manifest. The tracked PDFs
in the publication folders are left unchanged. Publication IDs are
`poster_first`, `poster_second`, `paper`, `presentation` and `network`;
`--all` builds every entry.

`--figures-only` regenerates poster figures and native tables from saved data.
It needs the Python `publications` extra; full builds also need TeX. No
adapter trains a model, solves a new reference trajectory or reruns HMC.

Both current posters use saved numerical arrays and existing layout checks.
The manuscript and presentation adapters compile historical material, and
their catalog entries state that the full historical narratives have not been
re-audited. Compiling a publication does not certify its claims.

See [the reproduction guide](../docs/reproduction.md) for output paths and
evidence levels, and [`docs/layout.md`](../docs/layout.md) for the folders
these items occupied before the layout change.
