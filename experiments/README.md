# Experiments

Seventeen experiments, numbered in the order the work developed: from the
original 27-state model to the seven-ODE PINNs, the inverse problem,
identifiability, neural–mechanistic hybrids and the Cramér–Rao analysis behind
the [paper](../paper/). Each folder except the protected baseline (05) has its own README with the
method, how to run it and where its outputs are.

**Status.** *Maintained*: run by the current `wnt-pinn` runner. *Complete*:
finished and kept as run. *Superseded*: a later experiment replaces it.
*Archived*: early or prototype work kept for the record. *Protected*: historical
baseline whose files are never edited. *Paper* and *registry* mark experiments
whose runs back the [paper](../paper/REPRODUCE.md) or the
[result registry](../results/registry.json).

Recovery counts are parameters within 10% of truth, per regime in the order
Normal / Early adenoma / Advanced adenoma / Severe APC loss.

## Model and sensitivity

| # | Experiment | Question | Headline result | Status |
|---|---|---|---|---|
| 01 | [Model prototypes](01-model-prototypes/) | Where do the model and the code base come from? | The 27-state MATLAB model, its SciPy ports and the first notebooks; provenance only | Archived |
| 02 | [Original PINN](02-original-pinn/) | Can a PINN solve the 27-state system, and which parameters drive the seven-ODE outputs? | Corrected sensitivity analysis: with `θP ≤ 1`, Sobol S1(`θP`) falls from 0.750 to 0.154 | Archived |
| 03 | [Global sensitivity](03-global-sensitivity/) | Which of the 30 rate constants of the 27-state model could be fixed? | 22 keep, 8 fix; strongly interacting (Σ ST 1.9–4.9) | Complete |

## Forward PINN

| # | Experiment | Question | Headline result | Status |
|---|---|---|---|---|
| 04 | [Forward PINN](04-forward-pinn/) | With known parameters, can a PINN reproduce all seven states in all four regimes? | Mean relative L2 error 2.41% from 40 sparse observations (1.06% with 100 labels) | Complete; paper, registry |

## Inverse PINN ladder

| # | Experiment | Question | Headline result | Status |
|---|---|---|---|---|
| 05 | [Single condition](05-inverse-single-condition/) | Can an inverse PINN recover the parameters from one noisy trajectory? | `W` within 5.2% in every regime; all 36 unknowns: 26 of 144 | Protected |
| 06 | [Three conditions](06-inverse-three-condition/) | Do three conditions and better conditioning rescue it? | 30 of 144 | Superseded |
| 07 | [Six conditions](07-inverse-six-condition/) | Do six conditions help, and does a classical ODE fit do better? | PINN 26 of 144; ODE fit 61 of 144 | Superseded |
| 08 | [Small networks](08-inverse-small-networks/) | Is network over-capacity the cause? | 25 of 144: capacity is not the lever | Complete (negative) |
| 09 | [Excitation](09-inverse-excitation/) | Do WNT- and MYC-perturbing conditions add the missing information? | ODE fit 61 → 82 of 144; autodiff PINN 37 of 144 | Complete; registry |
| 10 | [Integral residual](10-inverse-integral-residual/) | Does a derivative-free integral residual lift the PINN itself? | 37 → 50 of 144; the matched ablation credits the state weighting, not the residual | Maintained; paper, registry |

## Identifiability and uncertainty

| # | Experiment | Question | Headline result | Status |
|---|---|---|---|---|
| 11 | [Fisher information](11-fisher-information/) | What does each regime's data identify, and which condition restores `θP`? | CRLB of `θP` 0.031% → 55% from Normal to Severe (design study; 0.04% → 57% on the paper's benchmark); one WNT knockdown shrinks the Severe bound 22-fold | Complete; paper, registry |
| 12 | [Fisher, eight parameters](12-fisher-top8/) | Is an eight-parameter subproblem well conditioned with the rest known? | 8 of 8 identifiable in every regime; condition numbers about 100–665 | Complete; registry |
| 13 | [Bayesian inverse](13-bayesian-inverse/) | Can an HMC posterior say which parameters are identified? | Negative: every run fails the calibration gate | Complete; registry |
| 14 | [Bayesian forward](14-bayesian-forward/) | How uncertain is the forward solution under HMC over network weights? | 95% band covers the reference at 91–96% of times; chains under-mixed | Complete; registry |

## Hybrids

| # | Experiment | Question | Headline result | Status |
|---|---|---|---|---|
| 15 | [Hybrid UDE](15-hybrid-ude/) | When a network learns one regulation, can the host equation's basal rate still be identified? | Lost under every network parameterisation (`a_M` 32%/52% off); one knockdown that reaches the anchor brings it to 0.003% | Maintained; paper, registry |

## Extensions

| # | Experiment | Question | Headline result | Status |
|---|---|---|---|---|
| 16 | [Population extension](16-population-extension/) | What does a 48-hour ATRA exposure do to stem-like cells in the 14-state population model? | Stem-like population −30.4% (Severe APC loss), −38.3% (Normal) | Maintained |

## Cramér–Rao bound and SGN-PINN

| # | Experiment | Question | Headline result | Status |
|---|---|---|---|---|
| 17 | [Cramér–Rao bound and SGN-PINN](17-sensitivity-pinn/) | Estimator or experiment: what does the bound allow, and can a PINN reach it? | Bound: 132.5 of 136; PINNs 2–21 of 34; SGN-PINN 34/34/33/30 on held-out data | Complete; paper |

## Conventions

- **Self-contained folders.** Each experiment keeps its own code, launch
  scripts and outputs. Historical folders keep their own copy of the model, so
  their saved runs stay reproducible with the code that made them.
- **Long jobs.** Training and sampling runs take hours; launch them in a
  persistent terminal session (for example `tmux`) so they survive a dropped
  connection.
- **Outputs.** Each run writes to the folder's `runs/<timestamp>/`, which is
  git-ignored. Runs cited by the paper or the result registry are kept: tracked
  in place, or recorded with hashes in
  [`artifacts/catalog.json`](../artifacts/catalog.json) (see
  [`docs/artifacts.md`](../docs/artifacts.md)).
- **Shared model code** lives in [`src/wnt_pinn/`](../src/wnt_pinn/).
  Maintained experiments run through `wnt-pinn run configs/<profile>.json` and
  write to the root `runs/` directory; see
  [`docs/experiments.md`](../docs/experiments.md).
- **Old paths.** Before the layout change these folders had other names (for
  example `PINN-inverse-pinn-boost/`); [`docs/layout.md`](../docs/layout.md)
  maps old paths to new ones, and the tag
  [`paper-v1`](https://github.com/aidxhxr/PINN-Research/tree/paper-v1) keeps
  the old layout.
