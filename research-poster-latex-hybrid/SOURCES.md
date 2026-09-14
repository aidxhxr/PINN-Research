# Numerical provenance and interpretation

This is a separate revision of the approved garnet poster, with model/math in
column one, PINNs and identifiability in column two, and neural–mechanistic
hybrids in column three. The first poster in `research-poster-latex/` has since
been revised separately. The approved original PDF/source ZIP are preserved
locally under `poster-versions/20260907_155254_approved/`.
The results are from this repository.
No model training, resampling or ODE solving was performed.

## Source map

| Element | Original source, relative to repository root |
|---|---|
| Original schematic, Fig. 1 | `network-diagram/schematic-better.pdf`, also `poster-fable/assets/ready/schema.pdf` |
| Equations, state order, forcing, regimes | `PINN-hybrid-ude/odes.py`, `config.py`, `residual.py`; root `AGENTS.md` |
| Nondimensionalization and stemness | `research-paper/paper.tex`, equations ndb–ndc and stemness |
| Fourier network and forward losses | `PINN-smaller/forward-pinn-train-hybrid/model.py`, `pinn_core_dynamics.py`; `research-paper/paper.tex` |
| Beta-catenin/HOXA5 trajectories, Fig. 2 | `data/forward_trajectories.npz`; checkpoints in `PINN-smaller/forward-pinn-train-hybrid/runs/20260712_204546/`; `figures/forward_plot.py` |
| Dense forward accuracy | `PINN-smaller/forward_pinn_train/runs/20260711_203325/forward_error_table.json` |
| Sparse forward accuracy | `PINN-smaller/forward-pinn-train-hybrid/runs/20260712_204546/forward_error_table.json` |
| Autodiff recovery, Table 1 | `PINN-inverse-multicond-excite/runs/20260702_153336/*_recovered.json` |
| Integral recovery, Table 1 | `PINN-inverse-pinn-boost/runs/20260711_203325_integral/*_recovered.json` |
| Fisher table | `PINN-fisher-matrix/runs/20260711_203325_fisher/*_fim_summary.json` |
| HMC geometry, Fig. 2 | `PINN-bayesian/runs/20260713_204442_bayes/Normal_posterior_samples.npz` |
| Learned MYC activation, Fig. 3 | `PINN-hybrid-ude/runs/20260727_012448_bm_myc/{Normal,Severe_APC_Loss}_term.json` |
| Mechanistic control, Table 2 | `PINN-hybrid-ude/runs/20260726_195830_control/*_recovered.json` |
| RA → HOXA5 hybrid, Table 2 | `PINN-hybrid-ude/runs/20260726_214316_ra_h5/*_{recovered,term}.json` |
| Beta-catenin → MYC hybrid, Table 2 | `PINN-hybrid-ude/runs/20260727_012448_bm_myc/*_{recovered,term}.json` |
| APC-loss hybrid, Table 2 | `PINN-hybrid-ude/runs/20260728_233450_apc_mutation_frozen/*_{recovered,term}.json`; calibration in `runs/20260728_232743_apc_calibration/` |
| Five constraint forms | `notes/2026-08-01-hybrid-edge-atlas-and-anchor-visiting.md`, Result 1; `PINN-hybrid-ude/runs/20260801_204046_screen_atlas/` |
| MYC intervention comparison, Table 3 | `PINN-hybrid-ude/runs/20260802_dose_response/dose_{wnt,bcat}_bm_myc.json` |
| Prospective HOXA13 test, Table 3 | `PINN-hybrid-ude/runs/20260802_protocol_test/protocol_m_h13.json` |
| Failed HOXA13 → beta-catenin intervention | `PINN-hybrid-ude/runs/20260802_protocol_test/protocol_h13_b.json` |
| Reachability analysis | `PINN-hybrid-ude/runs/20260802_anchor_reach/anchor_reach.txt` |

The regulatory schematic's SHA-256 is
`6836f765e455581fdfdf8bff7f8b65423f9fa1c051da3ed7afed3d80576ab086`.
Its geometry, colors, labels, legend and edges are unchanged.

## Formulation

- The seven-state runtime model is typeset in full using Hill shorthand.
  A dot denotes derivative with respect to nondimensional time.
- The poster uses characteristic concentration scales, rather than claiming
  that the numerical initial state is all ones. The runtime initial state is
  `[0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40]`.
- The equations retain the runtime structure, including the nonlinear
  beta-catenin and HOXA5 losses. No runtime parameter was changed to match a
  different dimensional realization in the manuscript.
- The disease labels Early/Advanced abbreviate Early/Advanced Adenoma.
  Severe means Severe APC Loss. The reference trajectory horizon is 150 and
  the base ATRA pulse is [40,88].

## Corrections and numerical meaning

### Forward and inverse PINNs

- Forward errors are the grand mean relative L2 across seven states and four
  regimes, read directly from the saved JSON. Dense supervised: 1.0554%
  (displayed 1.06%); sparse forward PINN: 2.4074% (2.41%). The two formulations
  differ in their objectives as well as their numbers of labels.
- The forward trajectory figure shows beta-catenin and HOXA5 from the same saved
  sparse-PINN checkpoints (20260712_204546), evaluated on 6000 plot times.
  References use Radau with rtol=1e-10 and atol=1e-12. No model was retrained.
  Reusable curves and checkpoint hashes are included in `data/`; the build
  only plots these saved arrays. Native LaTeX arrows mark the exact [40,88]
  ATRA window above plain, unshaded axes.
- Table 1 is computed from actual recovered and true parameter dictionaries.
  The historical labels `Early_adenoma`, `Cancer-like` and
  `Strong_APC-mutant` map to Early Adenoma, Advanced Adenoma and Severe APC Loss.
- The matched-condition-count comparison is 16/9/6/6 → 17/16/10/7:
  **37 → 50 out of 144 regime–parameter pairs**, with 10 conditions per method.
  The old poster incorrectly printed the six-condition counts 10/4/5/7
  (total 26) beside a 37 → 50 claim.
- The comparison is between complete implementations. Integral residual,
  relative weighting, fixed collocation and multi-start change together; the
  gain is not isolated to the residual alone. The planned component ablation
  did not complete. No error bars or structural-identifiability claim are made.
- A later 20260712 excitation rerun gives 17/8/7/6 (38), so it is not silently
  substituted for the documented 37-parameter comparator.

### Fisher and HMC diagnostics

- Fisher counts are local eigenvector-participation classifications, not
  estimates within 10% of truth. The source uses log-parameter coordinates
  except for thetaP, with assumed observation noise sigma = 0.002.
- These counts do not deteriorate monotonically with WNT. The poster reports
  the actual table and the specific loss of APC-functional sensitivity.
- Fig. 2 uses actual Normal-regime HMC samples and an iso-product guide.
  The run fails effective-sample-size and coverage gates. It illustrates
  parameter trade-off, not calibrated posterior uncertainty.
- No marginal credible widths, 36/36 identifiability count, smoke-test
  correlation, or claim of calibrated HMC recovery is reproduced.
- The APC product relation is a property of the equation. At thetaP = 1,
  deltaP1 drops out; elsewhere, constant deltaP1*(1-thetaP) traces a ridge.

### Hybrid formulation, learned functions and matched parameter sets

- The MYC example replaces etaBM * H(b;kappaBM) by f_phi(b). The network has
  two hidden layers of five tanh units (46 weights), shared across conditions.
  Basal production, degradation and the timescale remain mechanistic.
  etaBM and kappaBM leave the inverse unknown set, reducing it from 36 to 34.
- State networks, remaining biological parameters and the learned term are
  fitted jointly using the integral residual. Neural regularization is 1e-8,
  calibrated to this implementation's objective scale.
- Fig. 3 uses the actual saved Normal and Severe APC Loss function arrays.
  Solid gray is the true mechanistic activation; dashed blue is the learned
  term. No extrapolated curve is drawn through the unobserved gap.
  Dots at (0,0) denote the exact architectural anchor, not measured inputs.
- NRMSE is RMSE divided by the RMS of the true function on the saved grid,
  as implemented in `PINN-hybrid-ude/training.py`. The figure generator
  recomputes both RMSE and NRMSE from the arrays. Scoring uses uniform grids
  over each regime's observed regulator range. Displayed MYC errors are
  7.9% (Normal) and 8.3% (Severe); across all four regimes the range is 6.2–9.5%.
- Table 2 is computed from the original recovered/true dictionaries stored in
  `data/hybrid_comparison.json`. For each variant and regime, both models are
  scored on the intersection of their recovered-parameter keys. True values
  agree on every intersected key. Counts are summed over the four regimes.

| Learned term | Mechanistic control | Hybrid | Denominator |
|---|---:|---:|---:|
| RA → HOXA5 | 51 | 53 | 4 × 34 = 136 |
| Beta-catenin → MYC | 48 | 43 | 4 × 34 = 136 |
| APC loss | 52 | 53 | 4 × 35 = 140 |

- The APC control count is **52**, not the unadjusted total of 53 repeated in
  an earlier narrative. Removing deltaP1 from the comparison removes one
  correctly recovered control parameter in Early Adenoma. This correction
  follows the common-parameter rule used for the other two rows.
- The RA/MYC NRMSE ranges summarize four separately trained functions.
  APC uses one shared function calibrated across known mutation severities,
  then frozen for inverse inference; its full-curve NRMSE is 0.1195% (0.12%).
  The displayed APC number is not four independent functional estimates.
- The rows represent different learned relationships and protocols; they are
  not an isolated ablation of joint training versus calibrate-then-freeze.
  In particular, APC has additional calibration information and a different
  restart protocol. No causal attribution to freezing alone is made.
- Function recovery and parameter recovery are reported separately. MYC's
  function error stays below 10% in all four regimes while five fewer
  parameter–regime pairs meet the parameter-error threshold.

### Compensation and experimental design

- The earlier illustrative anchor plot is replaced by the algebraic
  compensation identity in equation (9). The identity applies on observed
  support; the shifted function can return to the zero anchor in the
  unobserved gap. It does not claim a nonzero global shift preserves f(0)=0.
- The five-form constraint screen uses one start; the reported Normal basal
  errors span 13.8–202.8%.
- The intervention screens use exact reference states, fit one host equation,
  and fix parameters outside that equation at truth. They use two starts
  (seeds 100 and 117), selecting the lower physics loss. They do not establish
  performance with noisy, sparse state estimates or biological measurements.
- Table 3 shows 100 * hybrid_basal_err, absolute relative error against truth.
  These are not hybrid-minus-control differences.
- The MYC arms each use 11 conditions at every dose. Table 3 shows k=0:
  Normal 2.170 → 0.021%; Severe 1.550 → 0.006%. None is rounded to zero.
  The underlying dose series remains in the packaged data.
- The prospective HOXA13 baseline has 10 conditions and the two depletion
  protocols each have 11. The baseline, beta-catenin depletion and MYC
  depletion errors are respectively 12.844/51.516/0.111% in Normal and
  95.564/59.843/0.049% in Severe.
- Functional NRMSE is scored on each design's observed support. It can worsen
  despite improved basal recovery, so functional identification is not claimed.
- Reaching an anchor is insufficient if the intervention deletes the target
  parameter: switching off W reaches the HOXA13 anchor but leaves Normal W
  error at 16.2%. This counterexample motivates the final design criterion.
- The old poster's universal cross-arm collapse, exact-zero recovery and
  extra-siRNA claims are not reproduced. The screens implement model
  interventions, not a validated laboratory knockdown protocol.

## References

Checked against primary sources on 2026-09-07:

1. Raissi, Perdikaris & Karniadakis (2019). Physics-informed neural networks:
   A deep learning framework for solving forward and inverse problems
   involving nonlinear partial differential equations. J. Comput. Phys. 378,
   686–707. [Publisher](https://doi.org/10.1016/j.jcp.2018.10.045);
   [authors' project](https://maziarraissi.github.io/PINNs/).
   The poster uses a shortened title.
2. Rackauckas et al. Universal Differential Equations for Scientific Machine
   Learning. [arXiv:2001.04385](https://arxiv.org/abs/2001.04385), 2020,
   revised 2021.
3. Loman & Baker. Functional and parametric identifiability for universal
   differential equations applied to chemical reaction networks.
   [arXiv:2510.14140](https://arxiv.org/abs/2510.14140), 2025 preprint.

The references support the methods and distinction between functional and
parametric identifiability. The numerical findings are this repository's work.
