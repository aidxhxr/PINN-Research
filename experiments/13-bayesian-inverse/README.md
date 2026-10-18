# 13 · Bayesian inverse PINN: HMC over the 36 parameters

Can a posterior distribution, rather than a point estimate, say which
parameters the data identify? This folder samples the 36 kinetic parameters
by Hamiltonian Monte Carlo (HMC) with a dense mass matrix. The state networks
of the integral-residual PINN ([10](../10-inverse-integral-residual/)) are
frozen, and the derivative-free integral residual serves as a Gaussian physics
likelihood. It is the Bayesian version of that PINN's frozen-network
refinement stage. The answer is negative. The first run reports every
parameter as identifiable with pinpoint posteriors that miss the truth, which
is an artifact of the likelihood scaling. After the scaling was fixed, no run
passes the convergence and coverage gates in any regime. The posterior widths
are therefore not calibrated uncertainty.

| | |
|---|---|
| **Status** | complete. No chain passes both the ESS and coverage gates. The final paper's uncertainty results use the Cramér-Rao bound and Monte Carlo study in [17](../17-sensitivity-pinn/) instead |
| **Period** | 2026-07 |
| **Builds on** | [10 · inverse-integral-residual](../10-inverse-integral-residual/) |
| **Paper** | not used; marginals appear on both posters and the department talk (registry record `hmc_diagnostics`) |

## Result

A chain passes the gate if its median effective sample size (ESS) is at least
200 and the 95% interval covers the truth for at least 33 of the 36
parameters.

| run | likelihood | warmup / draws / leapfrog | median ESS (N / E / A / S) | truth covered, of 36 (N / E / A / S) | gate |
|---|---|---|---|---|---|
| `20260707_191047_bayes` | sum over all 209,930 residual terms | 800 / 1500 / 18 | about 3-6 per parameter | 1 / 0 / 1 / 1 | not computed; summary claims 36/36 IDENT |
| `20260712_204532_bayes` | mean residual x n_eff = 2,800 | 2000 / 2000 / 20 | 13 / 139 / 8 / 13 | 19 / 19 / 7 / 13 | fail in all four |
| `20260713_204442_bayes` | mean residual x n_eff = 2,800 | 3000 / 3000 / 48 | 96 / 1265 / 12 / 15 | 21 / 20 / 8 / 9 | fail in all four |

- **The first run is miscalibrated.** Summing about 210,000 residual terms
  with a small residual scale produces a pinpoint, biased posterior. In Severe
  APC Loss, for example, the posterior gives `W` = 1.120 ± 0.004 against a
  true value of 2.0, about 200 posterior sd away. Its `bayes_summary.md`
  verdict of "36/36 IDENT" in every regime is an artifact. Do not quote it.
- **The rescaled runs still fail.** Coverage falls with WNT drive (21/36 in
  Normal to 9/36 in Severe APC Loss in the last run). In high WNT the
  posterior concentrates away from the truth. Do not quote recovery counts or
  IDENT/WEAK verdicts from any of these chains.
- `20260712_204532_bayes` reproduces `20260708_003044_bayes` exactly (same ESS
  and coverage). It is a rerun after the regime rename.

## Method

- Unknowns `s` are log-ratios to the nominal values, plus a logit for
  `thetaP`. Prior: `s ~ N(0, 1)` for the log parameters and `N(0, 2^2)` for
  the logit of `thetaP`.
- Potential `U(s) = n_eff * mean(weighted residual^2) / (2 sigma_res^2) + prior`.
  The residual is the trapezoidal integral residual on frozen networks over
  4,000 collocation points per condition, with per-state relative weights.
  `sigma_res` is set to the RMS residual at the PINN's estimate. `n_eff`
  defaults to 10 conditions x 7 states x 40. The first run used the raw sum
  instead.
- With frozen networks the data term is constant in the parameters, so only
  the physics term enters.
- Sampler: self-contained PyTorch HMC. The dense mass matrix starts from the
  Laplace Hessian at the estimate, eigenvalue-floored to be positive definite,
  and is re-adapted from warmup covariance. The step size is tuned by dual
  averaging (target acceptance 0.8). Divergent steps are rejected.
- Each chain starts at the integral PINN's estimate (`<regime>_params.pt`).
- Outputs per regime: posterior samples (`.npz`), a summary JSON with ESS,
  coverage and gate flags, and marginal plots. `aggregate_marginals.py` writes
  the cross-regime figures and `bayes_summary.md`.

## Contents

| path | purpose |
|---|---|
| `bayesian_infer.py` | one regime: potential, Laplace metric, dense-metric HMC, diagnostics, gate, plots; flags `--seed-run --warmup --draws --leapfrog --colloc --n-eff --eff-per-traj --target-accept --sigma-res` |
| `run_bayes.sh` | runs all four regimes in parallel, then aggregates; env `SEED_RUN WARMUP DRAWS COLLOC LEAP TACC` |
| `aggregate_marginals.py` | cross-regime `W`/`thetaP` figure, verdict heatmap, `bayes_summary.md` |
| `plot_top8_marginals.py` | marginals of eight selected parameters sliced from the same 36-parameter chain (no resampling) |
| `regen_queue.sh`, `regen_queue.log` | the post-rename rerun and its printed summary |
| `config.py`, `model.py`, `odes.py`, `reference.py`, `residual.py` | model, frozen-network loader and integral residual |

## Reproduce

```bash
# needs a finished integral-PINN run with its *_net.pt and *_params.pt files
# (git-ignored), e.g. from ../10-inverse-integral-residual/run_boost.sh
pip install -e '.[ml,publications]'      # from the repository root: shared model, torch, matplotlib
cd experiments/13-bayesian-inverse
SEED_RUN=../10-inverse-integral-residual/runs/<timestamp>_integral \
WARMUP=3000 DRAWS=3000 LEAP=48 bash run_bayes.sh     # -> runs/<timestamp>_bayes/
```

Use one GPU and run it in a persistent terminal session. Each regime takes
about 2.7 hours (2000/2000 draws) to 3.4 hours (3000/3000 draws). `runs/` is
git-ignored. The registered samples and summaries
(`20260712_204532_bayes` Normal and Severe, `20260713_204442_bayes` Normal) are
tracked as `results/inputs/*_posterior_*`. The command
`wnt-pinn reproduce --result hmc_diagnostics --output runs/<new-dir>`
recomputes their coverage and ESS without resampling (see
[`results/README.md`](../../results/README.md)).

## Notes

- The posterior conditions on frozen state networks, so it omits all
  uncertainty in the state estimate. A joint posterior over network weights
  and parameters was planned but not built. The forward analogue, HMC over
  network weights with known parameters, is
  [14](../14-bayesian-forward/).
- The ESS is a historical single-chain estimate: raw autocorrelation summed
  until the first lag below 0.05. No R-hat is available. A passing gate would
  still need multi-chain checks.
- `plot_top8_marginals.py` and the department-talk slides show Normal coverage as 18/36 (a manual override). The
  saved samples give 19/36.
- Absolute widths depend on `sigma_res` and `n_eff`, which are modelling
  choices, not measured noise.
- What the data do and do not inform is answered without sampling by the
  Fisher bound and the converged MLE: [11](../11-fisher-information/),
  [17](../17-sensitivity-pinn/).
