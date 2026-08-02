# Pre-registered predictions — anchor dose-response

**Written 2026-08-02, BEFORE any fit in `runs/20260802_dose_response/` was run.**
The achieved anchor ratios below come from `anchor_doses.py`, which needs only
reference trajectories and no training, so they were knowable in advance. The
error columns are what the experiment measures and are predicted here.

## Why this design

The 2026-08-01 A/B is a two-point comparison: 10 conditions versus 10 + two
knockouts. Two points cannot distinguish "the anchor matters" from "these two
particular experiments help", and the information-matched arm
(`HYBRID_INFOCTL`) rules out only the generic-extra-data alternative.

This design removes the confound structurally instead of by a control arm:
**the number of conditions is fixed at eleven at every dose.** Ten base
conditions plus exactly one graded depletion. Only the *depth* of that one
depletion varies. `k = 1.0` is a near-null dose — it duplicates an existing
condition — so each arm carries its own internal zero.

## The measured quantity

Two statistics, both recorded, both fit on the same eleven conditions with the
same seeds and the same budget:

* **primary — hybrid basal error**: the relative error in the basal-production
  parameter of the equation hosting the network (`a5` or `aM`). This is the
  quantity an unvisited anchor lets the network absorb a constant out of, and
  it is the number the 2026-08-01 A/B moved from 17.0% to 0.1%.
* **secondary — excess basal error** = hybrid − mechanistic control on the same
  data. The subtraction removes information that helps the equation generally,
  leaving what the network specifically costs.

**Clarification recorded before any real fit was run.** An earlier draft of
this file named the excess as the sole primary statistic. A plumbing smoke test
at a 40-iteration budget — whose numbers are not interpretable and are not used
anywhere — showed that the excess can go *negative* when the mechanistic
control is itself badly conditioned, which the 2026-08-01 write-up already
documented for Severe APC Loss (three of four hybrids beat their control
there). A signed quantity that can flip sign makes the threshold predictions
below ambiguous. Both statistics are therefore reported at every dose, and the
thresholds in P2/P3/P4 are scored on the primary; the excess is scored
alongside and any disagreement between the two is reported rather than
resolved. Nothing else about the design changed, and no fit at a usable budget
had been run when this was written.

## Achieved anchor ratios (measured, no training)

| dose k | `ra` (r), Normal | `ra` (r), Severe | `wnt` (b), Normal | `wnt` (b), Severe | `bcat` (b), Normal | `bcat` (b), Severe |
|---|---|---|---|---|---|---|
| 1.00 | 0.1206 | 0.1434 | 0.0250 | 0.0665 | 0.0250 | 0.0665 |
| 0.30 | 0.0417 | 0.0464 | 0.0250 | 0.0661 | 0.0250 | 0.0619 |
| 0.10 | 0.0142 | 0.0159 | 0.0225 | 0.0312 | 0.0097 | 0.0161 |
| 0.03 | 0.0043 | 0.0048 | 0.0159 | 0.0144 | 0.0029 | 0.0045 |
| 0.01 | 0.0014 | 0.0016 | 0.0140 | 0.0109 | 0.0009 | 0.0015 |
| 0.00 | **0.0000** | **0.0000** | **0.0131** | **0.0092** | **0.0000** | **0.0000** |

The `wnt` arm cannot reach zero at any dose: at `kW = 0` the WNT drive is gone
entirely and `b` still floors at 0.019 / 0.028, because the HOXA13 → β-catenin
feedback keeps producing it. `bcat` knocks that feedback down too and reaches
the anchor exactly. This is the mechanism the plateau is attributed to, and the
`bcat` arm is the test of that attribution.

## Predictions

Scored on `ra_h5` (basal `a5`, equation `dh5`) for the `ra` arm and on `bm_myc`
(basal `aM`, equation `dm`) for the `wnt` and `bcat` arms.

**P1 — monotonicity.** Within each arm and regime, excess basal error is
monotone non-decreasing in the achieved anchor ratio. Scored as Spearman
ρ ≥ 0.8 between ratio and excess error over the six doses.

**P2 — the `ra` arm collapses.** Excess basal error falls from 15–30% at
`k = 1.0` to **< 2%** at `k = 0`, and is already **< 3%** by `k = 0.03`
(ratio ≤ 0.005). The 2026-08-01 A/B saw only the two endpoints of this curve
(17.0% → 0.1% in Normal); the prediction is that the interior is smooth and
monotone, not a step.

**P3 — the `wnt` arm plateaus and never gets there.** Excess basal error
flattens below `k = 0.1`, mirroring the ratio plateau, and **does not fall
below 2% at any dose** — including at complete WNT knockout. Expect a floor
around 2–5% in Normal and higher in Severe APC Loss.

**P4 — the `bcat` arm breaks the plateau.** At `k = 0` (ratio exactly 0)
excess basal error is **< 2%**, materially below the `wnt` arm's `k = 0` value
in the same regime. This is the direct test of the mechanism claimed for P3: if
the plateau is caused by the feedback arm, removing the feedback must break it.

**P5 — collapse onto one curve.** Plotted against *achieved anchor ratio*
rather than dose, the `ra` and `bcat` arms lie on approximately the same curve,
despite scoring different edges, in different equations (`dh5` vs `dm`), with
different basal parameters (`a5 = 0.15` vs `aM = 0.18`), under physically
different knockdowns (retinoid restriction vs a double genetic knockout). If
the anchor ratio is the governing variable, it should not matter how the ratio
was achieved.

**P6 — cross-arm consistency at matched ratio.** `wnt` at `k = 0`
(ratio 0.0131, Normal) and `bcat` at `k = 0.1` (ratio 0.0097, Normal) reach
comparable ratios by different means. Their excess basal errors should be
comparable — within a factor of ~2 — even though the `wnt` condition is a far
larger perturbation of the system.

## Falsifiers — what would sink the claim

* **P3 fails**, i.e. the `wnt` arm's excess error collapses below 2% despite
  its ratio plateauing at 0.013. That would show depth of knockdown, not anchor
  visitation, is doing the work, and the anchor account should be dropped. This
  is a real risk: `kW = 0` is a complete WNT knockout and is highly informative
  about the network generally.
* **P1 fails** in either direction-reaching arm, i.e. the response is flat or
  non-monotone in ratio. Then the 2026-08-01 endpoints were a coincidence of
  the two chosen conditions.
* **P5 fails**, i.e. the two arms trace clearly different curves against ratio.
  Then the anchor ratio is not the governing variable and something
  edge-specific is.

## What was *not* pre-registered

The functional NRMSE of the learned term, and the recovered-equation-parameter
counts, are recorded but not predicted here. They are secondary; the basal
error is the sharp statistic because it is the specific quantity an unvisited
anchor lets the network absorb.
