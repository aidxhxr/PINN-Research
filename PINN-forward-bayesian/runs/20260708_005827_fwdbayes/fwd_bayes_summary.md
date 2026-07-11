# Bayesian forward PINN — posterior-predictive UQ

Forward B-PINN (Yang/Karniadakis 2020): HMC over the network WEIGHTS with the ODE parameters KNOWN, given ~sparse noisy observations + the derivative-free trapezoidal physics residual. Deliverable = a 95% posterior-predictive band on the reconstructed trajectory (reconstruction uncertainty (+) observation noise sigma) — UQ on the forward solve.

## Per-regime summary

| regime | accept | ess(U) | ess(pred) | pred cov (obs) | mean relRMSE | rel pred band | epi cov (truth) | rel epi band |
|---|---|---|---|---|---|---|---|---|
| Normal | 0.87 | 6 | 322 | 0.95 | 0.001 | 0.080 | 0.98 | 0.005 |
| Early adenoma | 0.87 | 3 | 247 | 0.95 | 0.001 | 0.073 | 0.97 | 0.004 |
| Cancer-like | 0.85 | 4 | 98 | 0.94 | 0.001 | 0.069 | 0.92 | 0.003 |
| Strong APC-mutant | 0.86 | 3 | 280 | 0.95 | 0.001 | 0.067 | 0.94 | 0.003 |

## Per-state predictive coverage vs obs (rows = state, cols = regime)

| state | Normal | Early adenoma | Cancer-like | Strong APC-mutant |
|---|---|---|---|---|
| b | 0.95 | 0.93 | 0.95 | 0.95 |
| apc | 0.93 | 0.93 | 0.93 | 0.93 |
| h5 | 0.97 | 0.97 | 0.97 | 0.97 |
| h13 | 0.95 | 0.95 | 0.95 | 0.95 |
| m | 0.97 | 0.97 | 0.95 | 0.97 |
| r | 0.90 | 0.90 | 0.90 | 0.90 |
| c | 0.95 | 0.97 | 0.95 | 0.97 |

_Reading: 'pred cov (obs)' near 0.95 = the predictive band (reconstruction (+) noise sigma) is calibrated against the held-out noisy observations. 'epi cov (truth)' is the thin reconstruction-only band's coverage of the noise-free truth (~1.0 when the mean is well-pinned); a large 'rel pred band' flags states where the sparse data + physics leave the forward solve least pinned down._

