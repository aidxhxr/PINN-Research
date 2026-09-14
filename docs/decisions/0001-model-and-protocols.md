# Model and protocol compatibility

Status: accepted. Recorded 2026-09-14 from the repository guide, active model
implementations and current publication provenance.

The state order is `[b, p, h5, h13, m, r, c]`: beta-catenin, APC, HOXA5,
HOXA13, MYC, retinoic acid, CYP26A1. The initial state is
`[0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40]`.

Runtime parameters use `rho5=1.10`, `rhoB=1.10`, `rho13=1.30`. A different
dimensional realization appears in the manuscript; it must not silently
replace these runtime values. Four regimes use `(W, thetaP)` values
`(0.8,1.0)`, `(1.0,0.75)`, `(1.5,0.5)` and `(2.0,0.25)`.

References use SciPy Radau over `[0,150]`. Default ATRA pulse endpoints are
40 and 88. Untreated means `DR=0`, preserving circadian forcing. Timescale
parameters are part of the nondimensional equations, not plot rescaling.

Shared-code extraction must preserve forcing, state order, parameter order,
network output scaling and Fourier-feature buffers. Numerical parity and
checkpoint predictions are checked before active callers migrate. The
protected historical `PINN-inverse-solve/` directory remains untouched.

New independent-seed protocols are labeled separately from the historical
runs. Observation samples must stay fixed across optimization starts when
the purpose is to compare initialization. Preserving old results does not
require reproducing their seed coupling in new experiments.
