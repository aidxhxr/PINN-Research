# Bayesian inverse PINN — posterior identifiability

Tier-1 B-PINN: HMC over the 36 parameters with the pinn-boost state nets frozen and the derivative-free integral residual as the physics likelihood.

> **⚠ GATE FAILED for one or more regimes — the IDENT/NON-IDENT counts below are NOT trustworthy.** A chain needs ESS(median) ≥ 200 and CI coverage ≥ 33/36 before its verdicts mean anything.

## Per-regime summary

| regime | gate | ESS med | covers | accept | IDENT | WEAK | NON-IDENT | W post | thetaP post (true) |
|---|---|---|---|---|---|---|---|---|---|
| Normal | ❌FAIL | 96 | 21/36 | 0.79 | 34 | 1 | 1 | 0.759±0.009 | 0.973±0.053 (1.00) |
| Early Adenoma | ❌FAIL | 1265 | 20/36 | 0.68 | 34 | 2 | 0 | 0.842±0.018 | 0.586±0.175 (0.75) |
| Advanced Adenoma | ❌FAIL | 12 | 8/36 | 0.70 | 36 | 0 | 0 | 0.818±0.021 | 0.151±0.028 (0.50) |
| Severe APC Loss | ❌FAIL | 15 | 9/36 | 0.67 | 34 | 2 | 0 | 0.596±0.041 | 0.998±0.001 (0.25) |

## W and thetaP across regimes

| regime | W true | W post (CI95) | shrink | thetaP true | thetaP post (CI95) | shrink | verdict |
|---|---|---|---|---|---|---|---|
| Normal | 0.80 | 0.759 [0.74,0.77] | 0.01 | 1.00 | 0.973 [0.90,1.00] | 0.51 | WEAK |
| Early Adenoma | 1.00 | 0.842 [0.80,0.88] | 0.02 | 0.75 | 0.586 [0.31,0.87] | 0.40 | WEAK |
| Advanced Adenoma | 1.50 | 0.818 [0.78,0.86] | 0.03 | 0.50 | 0.151 [0.12,0.22] | 0.10 | IDENT |
| Severe APC Loss | 2.00 | 0.596 [0.54,0.70] | 0.07 | 0.25 | 0.998 [1.00,1.00] | 0.16 | IDENT |
