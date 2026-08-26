# Bayesian forward PINN — posterior-predictive UQ

Forward B-PINN (Yang/Karniadakis 2020): HMC over the network WEIGHTS with the ODE parameters KNOWN, given ~sparse noisy observations + the derivative-free trapezoidal physics residual. Deliverable = a 95% posterior-predictive band on the reconstructed trajectory (reconstruction uncertainty (+) observation noise sigma) — UQ on the forward solve.

## Per-regime summary

| regime | accept | ess(U) | ess(pred) | pred cov (obs) | mean relRMSE | rel pred band | epi cov (truth) | rel epi band |
|---|---|---|---|---|---|---|---|---|
| Normal | 0.88 | 4 | 400 | 0.95 | 0.002 | 0.080 | 0.96 | 0.006 |
| Early Adenoma | 0.87 | 3 | 104 | 0.94 | 0.001 | 0.073 | 0.91 | 0.004 |
| Advanced Adenoma | 0.87 | 4 | 195 | 0.94 | 0.001 | 0.069 | 0.96 | 0.003 |
| Severe APC Loss | 0.88 | 4 | 97 | 0.95 | 0.001 | 0.067 | 0.96 | 0.003 |

## Per-state predictive coverage vs obs (rows = state, cols = regime)

| state | Normal | Early Adenoma | Advanced Adenoma | Severe APC Loss |
|---|---|---|---|---|
| b | 0.93 | 0.93 | 0.93 | 0.93 |
| p | 0.93 | 0.93 | 0.93 | 0.93 |
| $h_5$ | 0.97 | 0.97 | 0.97 | 0.97 |
| $h_{13}$ | 0.95 | 0.95 | 0.95 | 0.95 |
| m | 0.97 | 0.97 | 0.95 | 0.97 |
| r | 0.90 | 0.90 | 0.90 | 0.90 |
| c | 0.97 | 0.95 | 0.97 | 0.97 |

_Reading: 'pred cov (obs)' near 0.95 = the predictive band (reconstruction (+) noise sigma) is calibrated against the held-out noisy observations. 'epi cov (truth)' is the thin reconstruction-only band's coverage of the noise-free truth (~1.0 when the mean is well-pinned); a large 'rel pred band' flags states where the sparse data + physics leave the forward solve least pinned down._
