# Numerical provenance and interpretation

This revision keeps the garnet layout of the earlier version, with model/math in
column one, PINNs and identifiability in column two, and neural–mechanistic
hybrids in column three. The first poster in `publications/posters/pinn-recovery/` has since
been revised separately. Earlier versions are in the git history.
The results are from this repository.
No model training, resampling or ODE solving was performed.

The 2026-09-17 revision shortens the visible text and enlarges the plots.
It omits the dense supervised comparison and the numerical five-form
constraint-screen range. Their provenance and
limitations remain below. Fig. 6 now compares full-model mechanistic and hybrid
fits. The intervention-screen figure was replaced by Table 3 (designed
knockdowns), and an abstract now opens column one.

## Source map

| Element | Original source, relative to repository root |
|---|---|
| Original schematic, Fig. 1 | `publications/figures/network-diagram/schematic-better.pdf` |
| Equations, state order, forcing, regimes | `experiments/15-hybrid-ude/odes.py`, `config.py`, `residual.py`; `docs/model.md` |
| Nondimensionalization | `publications/papers/2026-08-pinn-identifiability/paper.tex`, equations ndb–ndc |
| Additive ATRA treatment plot (`assets/atra-treatment.pdf`, not placed) | `src/wnt_pinn/model/numpy_rhs.py::_ra_input`; saved baseline in `data/forward_trajectories_provenance.json` |
| Fourier network and forward losses | `experiments/04-forward-pinn/forward-pinn-train-hybrid/model.py`, `pinn_core_dynamics.py`; `publications/papers/2026-08-pinn-identifiability/paper.tex` |
| Beta-catenin/HOXA5 trajectories, Fig. 2 | `data/forward_trajectories.npz`; checkpoints in `experiments/04-forward-pinn/forward-pinn-train-hybrid/runs/20260712_204546/`; `figures/forward_plot.py` |
| Dense forward accuracy | `experiments/04-forward-pinn/forward_pinn_train/runs/20260711_203325/forward_error_table.json` |
| Sparse forward accuracy | `experiments/04-forward-pinn/forward-pinn-train-hybrid/runs/20260712_204546/forward_error_table.json` |
| Autodiff recovery, Table 1 | `experiments/09-inverse-excitation/runs/20260702_153336/*_recovered.json` |
| Integral recovery, Table 1 | `experiments/10-inverse-integral-residual/runs/20260711_203325_integral/*_recovered.json` |
| Fisher table | `experiments/11-fisher-information/runs/20260711_203325_fisher/*_fim_summary.json` |
| Treatment fit, Fig. 3 | `experiments/10-inverse-integral-residual/runs/20260711_203325_integral` (Normal, `noATRA`/`ctrl`), cached in `data/normal_treatment_comparison.npz` (copied from the first poster) |
| Parity plot, Fig. 4 | `data/inverse_recovery.json` (integral rows); `figures/parity_plot.py` |
| Learned MYC activation, Fig. 5 | `experiments/15-hybrid-ude/runs/20260727_012448_bm_myc/{Normal,Severe_APC_Loss}_term.json` |
| Mechanistic control, Table 2 | `experiments/15-hybrid-ude/runs/20260726_195830_control/*_recovered.json` |
| RA → HOXA5 hybrid, Table 2 | `experiments/15-hybrid-ude/runs/20260726_214316_ra_h5/*_{recovered,term}.json` |
| Beta-catenin → MYC hybrid, Table 2 | `experiments/15-hybrid-ude/runs/20260727_012448_bm_myc/*_{recovered,term}.json` |
| APC-loss hybrid, Table 2 | `experiments/15-hybrid-ude/runs/20260728_233450_apc_mutation_frozen/*_{recovered,term}.json`; calibration in `experiments/15-hybrid-ude/runs/20260728_232743_apc_calibration/` |
| Full-model error comparison, Fig. 6 | The MYC, HOXA5 and control runs above; cached in `data/hybrid_comparison.json`, plotted by `figures/compare_hybrid_errors.py::poster_comparison` |
| Five constraint forms | `results/inputs/constraint_screen_transcription.json`; `experiments/15-hybrid-ude/runs/20260801_204046_screen_atlas/`; full five-form rerun in `experiments/15-hybrid-ude/runs/20260921_screen_5form/` |
| MYC knockdown rows, Table 3 (k=0) | `experiments/15-hybrid-ude/runs/20260802_dose_response/dose_{wnt,bcat}_bm_myc.json` |
| HOXA13 knockdown rows, Table 3 | `experiments/15-hybrid-ude/runs/20260802_protocol_test/protocol_m_h13.json` |
| Failed HOXA13 → beta-catenin intervention | `experiments/15-hybrid-ude/runs/20260802_protocol_test/protocol_h13_b.json` |
| Reachability analysis | `experiments/15-hybrid-ude/runs/20260802_anchor_reach/anchor_reach.txt` |
| Information-matched control (abstract) | `experiments/15-hybrid-ude/runs/20260801_infoctl/comparison.txt` |
| Profile likelihood (abstract) | `publications/papers/2026-08-pinn-identifiability/paper.tex` |

Run directories under `experiments/*/runs/` that are absent from the checkout
are archived; restore one with
`wnt-pinn artifacts restore github-history --select <path>` (`docs/artifacts.md`).

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
- The derived stemness index has been removed. Eq. (3) gives the RA input; the packaged but unplaced `assets/atra-treatment.pdf` plots
  the additive treatment term DR/2*(tanh(q*(t-tau1))-tanh(q*(t-tau2))), with
  DR=1.5, q=0.3, tau1=40, tau2=88, from the saved baseline. This is the
  treatment contribution to muR, not the RA state or the total RA forcing.
  The companion first poster shows baseline plus cosine input. The column-one
  heading and thetaP description use “APC functioning”; the disease table
  abbreviates the Severe APC Loss regime as “Severe.”

## Corrections and numerical meaning

### Forward and inverse PINNs

- The forward trainer sets lam_ic=20 and comments “strong anchor on the
  initial condition” (`experiments/04-forward-pinn/forward-pinn-train-hybrid/training.py`).
  The poster explains this as a heuristic weight emphasizing the known initial
  state. No ablation or optimality claim for 20 is supported by these results.
- Forward errors are the grand mean relative L2 across seven states and four
  regimes, read directly from the saved JSON. Dense supervised: 1.0554%
  (formerly displayed 1.06%); sparse forward PINN: 2.4074% (2.41%). The two formulations
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
  gain is not isolated to the residual alone. A later matched ablation (paper
  Fig 5A, seed 42) credits the weights and fixed collocation, not the residual. No error bars or structural-identifiability claim are made.
- A later 20260712 excitation rerun gives 17/8/7/6 (38), so it is not silently
  substituted for the documented 37-parameter comparator.

### Fisher and HMC diagnostics

- Fisher counts are local eigenvector-participation classifications, not
  estimates within 10% of truth. The implementation uses log-parameter
  sensitivities for every parameter, including thetaP, with assumed
  observation noise sigma = 0.002. An older code docstring incorrectly
  described thetaP as an exception.
- These counts do not deteriorate monotonically with WNT. The poster reports
  the actual table and the specific loss of APC-functional sensitivity.
- The former HMC scatter (Normal-regime thetaP/deltaP1 draws from a run that
  fails its ESS and coverage gates) was removed on 2026-09-24. Fig. 4 now shows
  recovered vs. true value for all 36 parameters, Normal and Severe APC Loss (integral method);
  the count inside the ±10% margin reproduces Table 1's integral row exactly
  (asserted in `figures/parity_plot.py`). The Fisher counts table stays beside it.
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
- Fig. 5 uses the actual saved Normal and Severe APC Loss function arrays.
  Solid gray is the true mechanistic activation; dashed blue is the learned
  term. No extrapolated curve is drawn through the unobserved gap.
  Dots at (0,0) denote the exact architectural anchor, not measured inputs.
- NRMSE is RMSE divided by the RMS of the true function on the saved grid,
  as implemented in `experiments/15-hybrid-ude/training.py`. The figure generator
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

### Direct error comparison in Fig. 6

- The two columns are β-catenin → MYC and RA → HOXA5. The top row shows
  regulatory-function NRMSE; the bottom row shows errors in the corresponding
  basal-production parameters aM and a5. Gray is the fitted mechanistic
  control; blue is the fitted hybrid. All four regimes are shown.
- These are the saved full multi-condition inverse-PINN runs used in Table 2,
  not the single-equation intervention screens. Both approaches fit state
  networks; the hybrid additionally replaces one biological activation law.
- Both function estimates are scored on the same saved 400-point grid over
  the hybrid's observed input range. The mechanistic curve uses the control's
  recovered Hill coefficients. MYC uses etaBM*b^2/(kappaBM^2+b^2); HOXA5 uses
  etaR*r/(kappaR+r). The closed-form truth is checked against the saved true
  curve. Function NRMSE is 100 * RMS(error) / RMS(truth).
- Parameter error is 100 * abs(estimate - truth) / abs(truth). The basal
  parameter is retained in both models. Choosing the same parameter role for
  both relationships keeps the compact comparison consistent. All remaining
  equation parameters and all 34 shared parameters are available in the
  gallery (`index.html`) that `figures/compare_hybrid_errors.py` writes to its `--out` directory.
- Display labels use one decimal place; exact values are recorded in
  `assets/hybrid_error_audit.json`. The graph shows selected saved fits,
  without uncertainty bars or an independent-replicate interpretation.

### Compensation and historical experimental-design screens

- The earlier illustrative anchor plot is replaced by the algebraic
  compensation identity in equation (7). The identity applies on observed
  support; the shifted function can return to the zero anchor in the
  unobserved gap. It does not claim a nonzero global shift preserves f(0)=0.
- The five-form constraint screen uses one start; the reported Normal basal
  errors span 13.8–202.8%. This range is transcribed from an unpublished dated
  research log. The complete original five-form run mapping has not been recovered;
  the cited screen-atlas directory contains gated/sc cases and cannot verify
  all five forms. A tracked transcription and source-note hash are in
  `results/inputs/constraint_screen_transcription.json`. This provenance
  limit is retained here; the range is omitted from the shortened poster.
- The intervention screens use exact reference states, fit one host equation,
  and fix parameters outside that equation at truth. They use two starts
  (seeds 100 and 117), selecting the lower physics loss. They do not establish
  performance with noisy, sparse state estimates or biological measurements.
- Table 3 (formerly an intervention figure) shows 100 * hybrid_basal_err, absolute relative error against truth.
  These are not hybrid-minus-control differences.
- The MYC arms each use 11 conditions at every dose. Table 3 uses k=0:
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
  error at 16.2%. This counterexample remains in the documentation; the
  shortened poster does not state a general intervention-design criterion.
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
