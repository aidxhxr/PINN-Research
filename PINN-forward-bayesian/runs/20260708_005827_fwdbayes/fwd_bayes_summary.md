# Bayesian forward PINN — posterior-predictive UQ

Forward B-PINN (Yang/Karniadakis 2020): HMC over the network WEIGHTS with the ODE parameters KNOWN, given ~sparse noisy observations + the derivative-free trapezoidal physics residual. Deliverable = a 95% credible band on the reconstructed trajectory (UQ on the forward solve).

## Per-regime summary

| regime | accept | ess(U) | ess(pred) | mean cov95 | mean relRMSE | mean rel band |
|---|---|---|---|---|---|---|
| Normal | 0.87 | 6 | 322 | 0.98 | 0.001 | 0.005 |
| Early adenoma | 0.87 | 3 | 247 | 0.97 | 0.001 | 0.004 |
| Cancer-like | 0.85 | 4 | 98 | 0.92 | 0.001 | 0.003 |
| Strong APC-mutant | 0.86 | 3 | 280 | 0.94 | 0.001 | 0.003 |

## Per-state 95% coverage (rows = state, cols = regime)

| state | Normal | Early adenoma | Cancer-like | Strong APC-mutant |
|---|---|---|---|---|
| b | 0.98 | 0.96 | 0.87 | 0.94 |
| apc | 0.99 | 0.95 | 0.90 | 0.95 |
| h5 | 0.99 | 0.96 | 0.90 | 0.89 |
| h13 | 0.99 | 0.99 | 0.95 | 0.97 |
| m | 0.99 | 0.97 | 0.94 | 0.97 |
| r | 0.98 | 0.96 | 0.94 | 0.95 |
| c | 0.97 | 0.97 | 0.93 | 0.95 |

_Reading: coverage near 0.95 = the band is calibrated; a state whose band is wide (high rel band width) but still tracks the truth is where the sparse data + physics leave the forward solve least pinned down._

