# Numerical provenance and interpretation

The first poster emphasizes PINN reconstruction, parameter recovery and
uncertainty. No model was retrained or HMC chain rerun for this revision.
Matched Normal references were evaluated with Radau during the one-time export;
ordinary builds only read saved arrays. Archived hybrid/forward data remain in
`data/` and dated snapshots, but are not used by the current poster figures.

## Source map

| Element | Original source, relative to repository root |
|---|---|
| Regulatory schematic, Fig. 1 | `network-diagram/schematic-better.pdf` |
| Model, state order and forcing | `PINN-inverse-pinn-boost/config.py`, `odes.py`; root `AGENTS.md` |
| Normal Radau and PINN curves, Figs. 2–3 | `PINN-inverse-pinn-boost/runs/20260711_203325_integral/Normal_{noATRA,ctrl}_net.pt`; exported to `data/normal_treatment_comparison.npz` |
| Network and training recipe | `PINN-inverse-pinn-boost/model.py`, `training.py`, `run_boost.py`, same run's `Normal.log` |
| Training diagnostics, Fig. 4 | same run's `Normal_history.json`, `Normal_recovered.json`; copied into `data/normal_inverse_*.json` |
| Autodiff recovery, Table 1 | `PINN-inverse-multicond-excite/runs/20260702_153336/*_recovered.json` |
| Integral recovery, Table 1 | `PINN-inverse-pinn-boost/runs/20260711_203325_integral/*_recovered.json` |
| Fisher diagnostics, Fig. 5 | `PINN-fisher-matrix/runs/20260711_203325_fisher/*_FIM.npy`; `data/fisher_matrices.npz` |
| Presentation Bayesian marginals, Fig. 6 | `PINN-bayesian/runs/20260712_204532_bayes/{Normal,Severe_APC_Loss}_posterior_samples.npz` and summaries; source mapping in `presentation/sync_figures.sh` |
| Reduced Fisher comparison, Table 2 | `PINN-fisher-matrix-top8/runs/20260711_203325_fisher_top8/*_fim_summary.json`; `data/fisher_top8.json` |

The original regulatory schematic SHA-256 remains
`6836f765e455581fdfdf8bff7f8b65423f9fa1c051da3ed7afed3d80576ab086`.

## Treatment comparison and trained networks

- State order is `[b,p,h5,h13,m,r,c]`; initial state is
  `[0.20,1.00,0.80,0.30,0.30,0.60,0.40]`. Concentrations use the original scales.
- Both panels use Normal parameters W=0.8 and thetaP=1. Only DR changes:
  noATRA has DR=0 and ctrl has DR=1.5. Baseline AR=0.04, period TR=24,
  mu0=0.35 and all other parameters remain identical. Pulse endpoints are 40/88.
- References use Radau, rtol=1e-10, atol=1e-12, 6000 evaluation times over
  [0,150]. Checkpoints and source-code hashes, exact parameters and exported
  relative L2 errors are in `data/normal_treatment_provenance.json`.
- Both comparisons show full trajectories at their original amplitudes.
  The oscillation insets from the previous proof were removed at user request.
- The saved networks come from the integral inverse-PINN run: ten conditions,
  one 4 × 256 GELU state MLP each, 16 Fourier frequencies (33 inputs), seven
  scaled outputs, 36 shared biological parameters, 150 observation times per
  condition, additive Gaussian noise sigma=0.002. The two displayed conditions
  are members of that jointly fitted ten-condition experiment.
- The selected Normal start is 0 (seed 42), chosen by final physics residual
  among two starts. Training uses 2000 Adam epochs, 150 L-BFGS steps, and a
  frozen-state parameter-refinement stage. No gain is attributed to the final
  refinement stage, which is effectively a no-op here.
- Relative L2 error is norm(PINN-reference)/norm(reference), evaluated over
  all 6000 times, not an oscillation-relative or held-out-support score.
  MYC/APC errors are 0.310657%/0.746920% untreated and 0.404763%/0.541723% treated.
- These state fits replace the previous 40-observation forward-PINN example.
  The old dense/sparse accuracy table is not reused for these new panels.

## Training diagnostics and recovery

- Fig. 4 uses eleven actual saved Adam checkpoints from the selected Normal
  run. Lines connect recorded samples; no intermediate history is synthesized.
  The loss curves are Ldata, lambda_phys*Lphys and 20*LIC, averaged over all ten
  conditions. Their sum is checked against the saved total loss.
- Selected parameter curves show 100*abs(estimate-truth)/abs(truth) for W and
  thetaP at the same Adam checkpoints. They are not the final L-BFGS estimates.
  L-BFGS follows Adam and is included in the final state fits in Fig. 3.
- Table 1 counts actual parameters within 10% of truth: autodiff 16/9/6/6,
  integral 17/16/10/7; totals 37 → 50 of 144 regime–parameter pairs. Both methods
  use ten conditions. Residual, relative weights, deterministic collocation and
  multi-start differ together; no isolated residual ablation is claimed.

## Fisher identifiability

- Fig. 5 uses full 36-parameter matrices, at truth, with 7 observed states,
  10 conditions, 200 times and assumed sigma=0.002. Every sensitivity column
  is theta_i*dy/dtheta_i, including thetaP (the code's older docstring differs).
- Matrix order is `UNKNOWN` in `PINN-inverse-multicond-excite/config.py`, not the
  participation-sorted CSV. Display order groups all 36 parameters by ODE.
- The Severe heatmap is Cij=Iij/sqrt(Iii*Ijj), normalized sensitivity overlap,
  not posterior covariance correlation. Large off-diagonal magnitudes flag
  similar or opposed sensitivity patterns.
- Each regime has one near-null eigen-direction (lambda<1). Values below 1e-4
  are explicitly displayed at that floor; numerical negatives are roundoff.
  Individual-parameter verdict counts and eigen-direction counts are distinct.
- The eight-parameter subset is lambdaC, etaBM, kappaBM, kappaBC, etaBC, etaRC,
  kappaB13 and kappaR, selected for lowest worst-case flat-subspace participation.
  Other 28 parameters are fixed at true regime values. All eight are locally
  identifiable with zero near-null directions in each regime. Condition numbers
  are 99.6552, 120.8990, 229.4061 and 665.1968, rounded to one decimal place.
- These conditional, local diagnostics neither prove global identifiability
  nor demonstrate reduced-parameter PINN recovery. Uncertainty in the fixed
  parameters is not included. A reduced-parameter PINN fit remains a next step.

## Bayesian distributions from the presentation

- Fig. 6 replots selected columns of the exact 2000-draw posterior-sample files
  used for the presentation's Normal top-eight and Severe worst-eight figures.
  Normal shows etaBM, kappaBM and kappaR; Severe shows etaBM, aM and lambdaP.
  They illustrate concentration near truth and concentration away from truth.
- All six are marginals of the full 36-parameter posterior conditional on
  frozen state networks and the integral likelihood. They are not HMC fits of
  the reduced eight-parameter model and do not include state-network uncertainty.
- Histograms use the original 40 density bins; solid lines indicate truth and
  dashed lines sample means, checked against the saved summaries. Source hashes
  and selection are in `data/presentation_posterior_provenance.json`.
- This is the presentation's 20260712_204532 run, distinct from the second
  poster's 20260713_204442 Bayesian scatter. No runs are blended.
- Both regimes fail ESS and coverage checks. Median ESS is 12.68 (Normal) and
  13.04 (Severe); posterior concentration is not evidence of calibrated
  uncertainty. The poster states that interval widths are unreliable.
- The presentation plotting script manually overrides Normal total coverage
  to 18/36, while the saved summary reports 19/36. That overridden total and
  the script's “confidently wrong” label are not reproduced in the poster.

## References

The current first-poster references support PINNs, inverse biological ODE
inference and Bayesian PINNs. Bibliographic details checked against primary
sources on 2026-09-14 (Raissi citation retained from the approved version):

1. Raissi, Perdikaris & Karniadakis (2019). Physics-informed neural networks.
   J. Comput. Phys. 378, 686–707. [Publisher](https://doi.org/10.1016/j.jcp.2018.10.045).
2. Yazdani, Lu, Raissi & Karniadakis (2020). Systems biology informed deep
   learning for inferring parameters and hidden dynamics. PLoS Comput. Biol.
   16, e1007575. [Publisher](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007575).
3. Yang, Meng & Karniadakis (2021). B-PINNs: Bayesian physics-informed neural
   networks for forward and inverse PDE problems with noisy data. J. Comput.
   Phys. 425, 109913. [Publisher](https://doi.org/10.1016/j.jcp.2020.109913).

The figures and numerical results are this repository's work. The actual
frozen-state HMC implementation is described above; it is not equated with
full Bayesian inference over neural-network weights.
