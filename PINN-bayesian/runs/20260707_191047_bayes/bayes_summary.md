# Bayesian inverse PINN — posterior identifiability

Tier-1 B-PINN: HMC over the 36 parameters with the pinn-boost state nets frozen and the derivative-free integral residual as the physics likelihood.

## Per-regime summary

| regime | accept | IDENT | WEAK | NON-IDENT | W post | thetaP post (true) |
|---|---|---|---|---|---|---|
| Normal | 0.90 | 36 | 0 | 0 | 0.792±0.000 | 1.000±0.000 (1.00) |
| Early adenoma | 0.90 | 36 | 0 | 0 | 0.840±0.001 | 0.716±0.002 (0.75) |
| Cancer-like | 0.90 | 36 | 0 | 0 | 0.922±0.002 | 0.643±0.002 (0.50) |
| Strong APC-mutant | 0.92 | 36 | 0 | 0 | 1.120±0.004 | 0.613±0.001 (0.25) |

## W and thetaP across regimes

| regime | W true | W post (CI95) | shrink | thetaP true | thetaP post (CI95) | shrink | verdict |
|---|---|---|---|---|---|---|---|
| Normal | 0.80 | 0.792 [0.79,0.79] | 0.00 | 1.00 | 1.000 [1.00,1.00] | 0.07 | IDENT |
| Early adenoma | 1.00 | 0.840 [0.84,0.84] | 0.00 | 0.75 | 0.716 [0.71,0.72] | 0.00 | IDENT |
| Cancer-like | 1.50 | 0.922 [0.92,0.93] | 0.00 | 0.50 | 0.643 [0.64,0.65] | 0.00 | IDENT |
| Strong APC-mutant | 2.00 | 1.120 [1.11,1.13] | 0.00 | 0.25 | 0.613 [0.61,0.62] | 0.00 | IDENT |
