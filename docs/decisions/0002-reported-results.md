# Rules for reported comparisons

Status: accepted. Recorded 2026-09-14 from current poster source notes and
saved numerical inputs.

Parameter recovery means absolute relative error strictly below 10%. Counts
always include their denominator. The matched ten-condition inverse comparison
is 37 versus 50 out of 144 parameter-regime pairs. This is a gain of 13 pairs,
not a doubling. Residual formulation, relative weights, collocation and starts
changed together; the result cannot be attributed to one change in isolation.

Hybrid comparisons use the intersection of remaining parameter sets per regime
and verify matching true values. RA-HOXA5 is 51 versus 53 out of 136; beta-MYC
is 48 versus 43 out of 136; APC loss is 52 versus 53 out of 140. Repeating the
unadjusted control total of 53 in every row is incorrect.

Function NRMSE is RMS(learned-true)/RMS(true) over saved grids within observed
regulator ranges. Basal recovery and function recovery are separate outcomes.
Equation-level exact-state screens are not full inverse PINNs and do not
demonstrate performance with noisy, sparse state estimates.

Fisher classifications are local sensitivity diagnostics. Log-parameter
sensitivities include thetaP despite an older code docstring. Near-null
eigen-directions differ from individual parameter verdicts. The eight-parameter
analysis fixes the other 28 at truth; no reduced-parameter PINN recovery result
has been established by that calculation.

Inverse HMC conditions on frozen state networks. Saved chains fail ESS and
coverage checks, so their posterior widths cannot be interpreted as calibrated
uncertainty. The first poster's presentation-derived Normal coverage is 19/36
from the saved samples, not the plotting script's manual 18/36 override.

The five-constraint MYC range remains a historical transcription until its
complete original run mapping is recovered. Explicit provenance gaps are
retained in the registry. Hash verification establishes input identity;
recomputing a metric establishes agreement with that input, not biological
validity or a complete reconstruction of undocumented training.
