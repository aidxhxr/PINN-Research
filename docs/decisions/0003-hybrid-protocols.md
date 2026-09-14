# MYC and APC learned terms

Status: accepted. Tracked 2026-09-14 from the implementation and the dated
2026-07-28 MYC/APC protocol note.

The default learned activation is a shared two-hidden-layer tanh network
with five units per layer, nonnegative output and exact zero-input anchor.
For the MYC example it replaces beta-catenin activation; etaBM and kappaBM
leave the biological inverse set, reducing its size from 36 to 34.

Construct learned-term networks after the condition-specific state networks.
Changing this order changes random-number consumption and invalidates a
matched initialization comparison. The neural L2 penalty is `1e-8` on this
implementation's loss scale. Larger literature-scale penalties previously
overwhelmed the objective and collapsed the learned function.

The APC-loss protocol has two stages. First calibrate a shared, monotone
degradation function against known mutation-severity information, with an
exact healthy anchor. Then freeze that function during four-regime inverse
inference. It has additional calibration information compared with a jointly
fitted activation hybrid and must be labeled accordingly.

Frozen-state refinement is effectively a no-op in the reported hybrid runs.
Do not attribute learned-term performance to that stage. An exact `f(0)=0`
constraint does not determine basal production if the observed regulator
values remain far from zero. On observed support a neural offset can compensate
for a changed basal term.

Interventions should reach the anchor while retaining the parameter being
estimated in the active equation. Turning off a forcing term can remove its
parameter from the equation and make recovery impossible, even if it reaches
the desired state range. Saved model-intervention screens are synthetic
design studies, not validated laboratory knockdown protocols.
