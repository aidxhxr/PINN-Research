# Model implementation

The maintained model is in `src/wnt_pinn/model/`. Its version identifier is
`wnt-ra-hox-runtime-v1`. Importing model definitions does not select a device,
change Torch's default dtype, or set CPU thread counts.

State order is `[b, p, h5, h13, m, r, c]`, with initial values
`[0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40]`. The runtime values
`rho5=1.10`, `rhoB=1.10`, and `rho13=1.30` are retained. They differ from a
dimensional realization in the manuscript; that distinction is intentional.

`parameters(regime, overrides)` returns an independent dictionary. The ten
standard intervention conditions and 36-parameter order match the historical
excitation analysis. Known depletion multipliers are fixed inputs and do not
enter the inverse unknown set.

## Numerical backends

`numpy_rhs.py` implements the reference equations, including the known WNT,
MYC and depletion inputs. It retains the original nonnegative-state clipping
used by the reference solver. Torch residuals retain their original treatment
of the network outputs; numerical agreement is tested on nonnegative states.

`integral_rhs.py` and `hybrid_rhs.py` keep the two original operation orders.
The hybrid version supports the learned regulatory terms, including modulator
and APC-production terms. Algebraically equivalent multiplication orders can
produce different floating-point roundoff, so the two implementations are
kept explicit within the same package.

The reference solver uses Radau with `rtol=1e-10` and `atol=1e-12`. The default
horizon is 150 and ATRA pulse is [40, 88]. Setting `DR=0` removes treatment
and retains circadian forcing.

## Migration and verification

Active integral, hybrid and inverse Bayesian modules now import shared
networks and equations through compatibility modules. Both Fisher analyses
use shared parameter definitions and reference equations without importing an
experiment folder through `sys.path`. Earlier experiment implementations and
`PINN-inverse-solve/` remain historical records.

The parity fixtures in `tests/fixtures/legacy_model/` preserve original
implementations. Tests compare equations at treatment boundaries, parameter
gradients, parameter ordering, all four regimes, seeded initialization and
checkpoint state keys. State networks are constructed before learned terms,
as required by the historical random-number sequence.

Fisher sensitivities scale every column by its parameter value, including
`thetaP`. The eight-parameter analysis fixes the other 28 parameters at truth.
This conditional local analysis does not establish global identifiability or
eight-parameter PINN recovery.
