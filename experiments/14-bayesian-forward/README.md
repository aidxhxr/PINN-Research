# 14 · Bayesian forward PINN: HMC over network weights

How well do 40 noisy observations and the known physics pin down the state
trajectory? This folder is the forward twin of [13](../13-bayesian-inverse/).
The kinetic parameters are known, and Hamiltonian Monte Carlo (HMC) samples
the weights of a small Fourier-feature PINN given sparse noisy data, the
initial condition and the derivative-free integral residual. The output is a
posterior band on each reconstructed state. In all four regimes the latent
95% band covers the noise-free reference at 91-96% of evaluation times, with
a mean relative RMSE of 0.07-0.15%. The weight chain itself mixes poorly
(energy ESS of 3-4), so the bands are a useful picture of where the sparse
fit is least constrained rather than calibrated intervals.

| | |
|---|---|
| **Status** | complete (exploratory; not used in the final paper) |
| **Period** | 2026-07 |
| **Builds on** | [04 · forward-pinn](../04-forward-pinn/) (sparse-data forward solve), [13 · bayesian-inverse](../13-bayesian-inverse/) |
| **Paper** | not in the final paper; earlier manuscript under `publications/papers/2026-08-pinn-identifiability/`; registry record `bayesian_forward` |

## Result

Run `runs/20260711_203325_fwdbayes/`: 40 observations per state with noise
sigma = 0.02, 1000 warmup and 1000 draws, 40 leapfrog steps.

| regime | acceptance | ESS of energy | ESS of beta-catenin prediction | latent band covers truth | mean relative RMSE | predictive band covers the 40 observations |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 0.88 | 4 | 400 | 0.96 | 0.002 | 0.95 |
| Early Adenoma | 0.87 | 3 | 104 | 0.91 | 0.001 | 0.94 |
| Advanced Adenoma | 0.87 | 4 | 195 | 0.96 | 0.001 | 0.94 |
| Severe APC Loss | 0.88 | 4 | 97 | 0.96 | 0.001 | 0.95 |

"Latent band covers truth" is the fraction of 2,000 evaluation times at which
the noise-free reference lies inside the 95% band of the reconstruction,
averaged over the seven states. The relative RMSE is normalised by each
state's maximum, floored at 0.05. The mean relative width of the latent band
is 0.3-0.6% of the same per-state scale. The predictive band adds the observation noise
and is 6.7-8.0% wide.

## Method

- Network: Fourier-feature GELU PINN with width 64 and depth 3 (10,951
  weights; 16 frequencies at scale 4), kept small so the weight posterior
  can be sampled. The Fourier features are fixed and not sampled.
- Data: 40 scattered observation times per regime, including t = 0, with
  Gaussian noise of sd 0.02 on all seven states. The constraints are the
  initial condition and the trapezoidal integral residual on an 800-point
  fixed grid.
- First a MAP fit by Adam (4,000 epochs, learning rate 2e-3). Each likelihood
  scale (data, initial condition, physics) is then set to the residual RMS at
  the MAP, so the physics likelihood does not grow with the collocation
  count. That growth is the failure of the first inverse run in
  [13](../13-bayesian-inverse/).
- Prior on the weights: N(0, 3^2). Sampler: HMC with a diagonal mass matrix
  adapted Stan-style and a dual-averaging step size (target acceptance 0.75).
- Prediction: up to 400 posterior draws evaluated on a 2,000-point grid. The
  script reports the latent (reconstruction-only) 95% band and a predictive
  band that adds the observation noise.

## Contents

| path | purpose |
|---|---|
| `bayesian_forward.py` | one regime: sparse data, MAP pre-fit, noise calibration, diagonal-metric HMC, predictive bands, summary; flags `--n-data --noise-std --colloc --width --depth --warmup --draws --leapfrog --sigma-w --n-eval --max-pred-draws` |
| `run_bayes_forward.sh` | runs all four regimes in parallel, then aggregates; env `NOISE NDATA WARMUP DRAWS COLLOC LEAPFROG` |
| `aggregate_forward.py` | beta-catenin band figure across regimes, coverage heatmap, `fwd_bayes_summary.md`; replots from saved arrays without rerunning HMC |
| `config.py`, `model.py`, `odes.py`, `reference.py`, `residual.py` | model, network and integral residual |

## Reproduce

```bash
cd experiments/14-bayesian-forward
bash run_bayes_forward.sh                 # 40 obs, noise 0.02, 1000/1000 draws -> runs/<timestamp>_fwdbayes/
python3 aggregate_forward.py --run runs/<timestamp>_fwdbayes   # re-aggregate or replot only
```

Use one GPU and run it in a persistent terminal session. Each regime takes
about 50 minutes, and the four run in parallel. `runs/` is git-ignored. The
registered predictive arrays and summaries of `runs/20260711_203325_fwdbayes/`
are tracked as `results/inputs/*_predictive*`. The repository README's figure
`docs/figures/bayesian_forward_bands.png` is a snapshot of its
`fwd_beta_bands.png`. The command
`wnt-pinn reproduce --result bayesian_forward --output runs/<new-dir>`
recomputes RMSE and coverage from them. The `runs/20260708_*_fwdbayes/` runs
use the regime names from before the rename. In the earliest of them the
predictive band did not include the observation noise.

## Notes

- The energy ESS is 3-4 in every regime, so the weight chain is far from
  mixed. The ESS of the beta-catenin prediction is higher (97-400) because
  predictive functionals are insensitive to redundant weight directions. Full
  weight-chain histories were not saved.
- The "predictive coverage" is measured on the same 40 observations used for
  training, not on held-out data. The wording "held-out" in the generated
  `fwd_bayes_summary.md` is inaccurate. Coverage of one synthetic trajectory
  is also not repeated-experiment calibration.
- The likelihood scales are set from the MAP fit (reduced chi-square = 1),
  so the band widths depend on that choice.
- The observation noise here (0.02) is ten times that of the inverse
  benchmark (0.002), and the network is far smaller than the 256x4 networks
  of the inverse PINNs.
