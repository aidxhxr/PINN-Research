# 17 · Cramér–Rao accounting, converged MLE and the Schur-complement Gauss–Newton PINN

When an inverse PINN fails to recover a parameter, is the estimator at fault or
does the experiment lack the information? This folder answers that for the
ten-condition benchmark design. It computes the Fisher information and
Cramér–Rao bound (CRLB) of the design and checks by Monte Carlo that a converged
maximum-likelihood fit attains it. It then places three PINN formulations
against the bound and introduces a new estimator, the **Schur-complement
Gauss–Newton PINN (SGN-PINN)**, which closes most of the gap from inside the
PINN framework. The answer: the data support nearly every parameter. An
efficient estimator would recover 132.5 of 136 regime–parameter pairs, but the
PINNs recover 2–21 of 34 per regime, so the shortfall is an **estimator gap**.
SGN-PINN matches the classical fit wherever its continuation converges.

| | |
|---|---|
| **Status** | complete |
| **Period** | 2026-10 |
| **Builds on** | [10 · inverse-integral-residual](../10-inverse-integral-residual/) (the integral PINN that SGN-PINN starts from, and the ablation variants), [11 · fisher-information](../11-fisher-information/) (earlier local Fisher analysis) |
| **Paper** | Figs 3–7 and S1 Fig; Sections 2.3–2.7, Methods 4.2–4.4 and 4.6 ([paper](../../paper/main.pdf), [figure map](../../paper/REPRODUCE.md)) |

## Result

Counts are parameters within 10% of truth among the **34 regular parameters**
per regime, i.e. all 36 minus the structurally degenerate pair
(θ<sub>P</sub>, δ<sub>P1</sub>), which enter the model only through the product
δ<sub>P1</sub>(1−θ<sub>P</sub>). Benchmark data (seed 42); held-out replicate
(seed 7) in parentheses where it was run.

| estimator | Normal | Early Adenoma | Advanced Adenoma | Severe APC Loss |
|---|---|---|---|---|
| PINN, autodiff residual (1 start) | 14 | 10 | 7 | 2 |
| PINN, + relative weights, fixed collocation | 21 | 19 | 10 | 10 |
| PINN, + integral residual | 17 (18) | 15 (16) | 10 (13) | 7 (9) |
| **SGN-PINN** (from the integral PINN) | **34 (34)** | **34 (34)** | 28 (**33**) | 24 (30) |
| classical MLE, best of 4 starts | 34 (34) | 34 (34) | 34 (33) | 34 (31) |
| CRLB-expected count E[N<sub>10</sub>] | 34.0 | 34.0 | 33.4 | 31.1 |

- **Information budget (Fig 3, Section 2.3).** The 36-parameter Fisher matrix
  has 35 informative eigenvalues that span 5.7–8.3 orders of magnitude, and one
  structural null direction that lies in the (log θ<sub>P</sub>, log
  δ<sub>P1</sub>) plane to within 10<sup>−6</sup>. With δ<sub>P1</sub> known,
  every parameter in the two low-WNT regimes has a relative CRLB below 1.3%.
  31 of 35 are below 5% in Advanced Adenoma and 25 of 35 in Severe APC Loss,
  where θ<sub>P</sub> (57%), ρ<sub>13</sub> (21%), ε<sub>P</sub> (18%) and
  a<sub>C</sub> (16%) are the least informed. The CRLB-expected count is 132.5
  of 136.
- **The bound is attained (Fig 4, Section 2.4).** We ran unit-weight MLE fits,
  started at the truth with δ<sub>P1</sub> known, on 24 independent datasets
  per regime (seeds 101–124). In Normal and Early Adenoma the standardised
  errors have variance 1.03 and 1.13, 95% intervals cover the truth in 95% and
  92% of cases, and twice the cost drop has mean 36.3 and 33.1 against
  χ²<sub>35</sub>. In the high-WNT regimes the estimates scatter *less* than
  the bound (variance 0.88 and 0.70, likelihood-ratio means 31.9 and 31.0).
  This is non-asymptotic shrinkage in a weakly informed likelihood, not a
  failure to converge.
- **The estimator gap (Fig 5, Section 2.5).** The integral PINN's median errors
  are 13–35 times the bound across both datasets. Parameters that the design
  pins to 10<sup>−3</sup> come back at 10<sup>−1</sup>. In the matched
  ablation, the derivative-free integral residual adds nothing on top of
  relative weighting and fixed collocation (17/15/10/7 against 21/19/10/10).
  **Weighting and fixed collocation carry the gain, not the integral residual.**
- **SGN-PINN (Fig 6, Section 2.6).** On the held-out seed-7 data it recovers
  34/34/33/30, with median errors 0.11%, 0.20%, 1.1% and 1.5%. Where the λ
  continuation converges (Normal, Early and Advanced on seed 7; Normal and
  Early on seed 42), its per-parameter errors coincide with the classical fit
  (rank correlation of |log error| 0.99–1.00). In Severe APC Loss on seed 7,
  and in Advanced and Severe on seed 42, the physics residual stalls at
  7×10<sup>3</sup>–1.5×10<sup>4</sup>. The stalled residual flags the problem
  without needing the truth. Its self-reported standard errors, the inverse
  Schur complement, track the CRLB at a median 1.2–1.6 times the bound.
- **The classical fit needs care too.** From the common start, the unit-weight
  fit in Early Adenoma stops in a genuine local minimum (19 of 34 recovered, at
  18 times the global cost). 34 of the 36 jittered starts reach the global
  minimum. Selecting among four starts by cost, which never uses the truth,
  recovers the bound's count on both datasets.
- **θ<sub>P</sub> is an information gap (Fig 7, Section 2.7).** The fits behind
  Fig 7B (`ms_tight/*_s42_none{,_info,_dep}.json`) and the design bounds
  (`fim2_*`) live here. With the depletion pair, the bound on θ<sub>P</sub> in
  Severe APC Loss falls from 57% to 1.8% and all 35 parameters are below 5%.
  The information-matched pair gives 53%. The training-free design study is
  in [11 · fisher-information](../11-fisher-information/).

## Method

- **One benchmark for every estimator** (`bench.py`). The model and conditions
  come from `wnt_pinn.model` (36 unknowns, 10 conditions, 4 regimes). The
  reference is Radau at rtol 10<sup>−10</sup> on a 5000-point grid. Each
  condition has the t=0 point plus 149 random grid times, all 7 states, and iid
  N(0, 0.002²) noise, with data seed + condition index. The weights are
  1/max(max|y|, 0.05). `make_data(regime, seed, design)` also builds the added
  designs `info`, `dep` and `depb` (`bench.EXTRA`). The ten base conditions are
  byte-identical in every design.
- **Fisher information and CRLB** (`fim.py`). F = JᵀJ/σ², with J from central
  differences in log-parameters on Radau solves. It gives the unweighted CRLB
  and the sandwich covariance of the relative-weighted objective. The weighted
  estimator's standard deviations exceed the bound by a median 7–9% (at most
  47%). `--save-jac` keeps per-condition Jacobians for leave-one-out,
  partial-observation and noise-scaling analyses (S1 Fig).
- **Converged classical MLE** (`mle.py`). Single shooting with LSODA and
  trust-region-reflective least squares. Forward-difference Jacobians are
  computed in a process pool. Each fit runs to convergence (at most 400
  evaluations), with a 30 s alarm per ODE solve. Flags: `--start
  init|truth|pinn`, `--jitter-seed` for N(0, 0.35²) log-jitter, `--weights
  rel|none`, `--fix-deltaP1`, `--design`, `--rtol`, `--fd-step`. Paper fits use
  `--rtol 1e-11 --fd-step 1e-5`.
- **Monte Carlo check** (`mc_costdrop.py`, `mc_restart_check.py`). The first
  compares 2×(cost at truth − cost at MLE) with χ²<sub>35</sub>. The second
  restarts a fit from its own estimate with stricter stopping rules.
- **SGN-PINN** (`sgn.py`). The trained integral PINN's 256 last-hidden features
  are frozen. To them we add 512 random Fourier features (ω uniform on [0, π]),
  24 decaying exponentials e<sup>−t/τ</sup> with τ log-spaced in [0.05, 20],
  and a constant. Each condition's state is then *linear* in its head matrix,
  and the problem in (heads, θ), about 5.6×10<sup>4</sup> unknowns, is
  separable nonlinear least squares. Its Gauss–Newton matrix is
  block-arrowhead, so each Levenberg–Marquardt step costs ten Cholesky
  factorisations plus a 36×36 Schur-complement solve. λ is continued from
  10<sup>−2</sup> up to 10<sup>3</sup>. The undamped Schur complement gives
  the estimator's own standard errors. The work is in float64 on a GPU.
- **Truth-free selection** (`reselect.py`). Two schedules run per regime: S1
  steps λ through 10<sup>−2</sup>…10<sup>3</sup>, and S2 runs a 15-iteration
  head presolve and then λ = 1…10<sup>3</sup>. The reported one has the lower
  *shooting* misfit, the weighted misfit of the ODE solution at its
  parameters. Basis and schedules were fixed on seed 42, and seed 7 is held
  out.
- **Tangent PINN** (`tpinn.py`, negative result). The network outputs states
  *and* their parameter sensitivities, with z = Z + S(x − x<sub>ref</sub>),
  plus integral residuals for both. At small λ the network absorbed the data
  and θ barely moved (Normal 6/36 after 4250 epochs, Early stalled). A
  physics-only solve drifted to errors up to 90% of state scale after 2000
  epochs, the long-horizon forward-PINN failure. Abandoned in favour of
  SGN-PINN.

## Experiments in this folder

| run directory | question | in git |
|---|---|---|
| `runs/20261002_integral_seed42/`, `runs/20261002_integral_seed7/` | Matched baseline: the integral PINN of [10](../10-inverse-integral-residual/) on byte-identical benchmark and replicate data. These are the start points for SGN-PINN. | JSON and `summary.txt` (network weights `*.pt` are not committed) |
| `runs/20261017_paper/fim_seed42/` | Information budget of the benchmark design (Fig 3) | yes |
| `runs/20261017_paper/fim2_{base,info,dep,depb}/` | Per-condition Jacobians (S1 Fig) and bounds under added designs (Section 2.7) | `fim_summary.json` for all four; Jacobians (`*_Jcond.npy`) for `base` only |
| `runs/20261017_paper/mc_tight/`, `mc_tight_costdrop.json` | Is the bound attained? 96 truth-start MLE fits (Fig 4) | yes |
| `runs/20261017_paper/ms_tight/`, `ms_tight_cap/` | Multistart MLE from the common start and 3 jittered starts (seed 42 relative and unit weights, seed 7 relative), plus the three-design θ<sub>P</sub> fits (Figs 5, 7B) | yes |
| `runs/20261017_paper/sgn_seed42/`, `sgn_seed7/` | SGN-PINN v2, both schedules, with `reselect.json` (Figs 5, 6) | yes |
| `runs/20261017_paper/ablation_seed42/{base,plumb}/` | Matched PINN ablation: autodiff and weighted rungs (Fig 5) | yes |
| `runs/20261017_paper/wallclock.json` | Measured run times (Fig 6E) | yes |
| `runs/20261017_paper/mc/`, `ms/`, `mc_costdrop.json` | The same fits at the default tolerance (LSODA 10<sup>−8</sup>, FD step 10<sup>−6</sup>). Superseded by `*_tight`. | yes (kept for comparison) |
| `runs/20261017_paper/apc_turnover.json`, `restart_checks.jsonl` | Numbers quoted only in the text (APC share of β-catenin removal, restart checks) | yes |
| `runs/20261002_mle/`, `20261002_sgn_from_pinn0711/`, `20261002_sgn_lampath/`, `20261002_sgn_v2/`, `20261002_tpinn_v1/`, `20261002_ablate/` | Development rounds: first MLE, SGN v1 basis, λ-path tests, v2 on the earlier PINN, tangent PINN, an SGN ablation that was cancelled | not committed |

## Contents

| path | purpose |
|---|---|
| `bench.py` | Shared benchmark: true parameters, Radau reference, `make_data`, added designs, log coordinates |
| `fim.py` | Fisher information and CRLB of a design (unweighted and weighted sandwich) |
| `mle.py` | Converged classical least-squares and MLE reference estimator |
| `sgn.py` | SGN-PINN: feature head, block-arrowhead LM, λ continuation, profiled standard errors (also `--opt adam/vp` variants) |
| `reselect.py` | Truth-free schedule selection by shooting misfit |
| `tpinn.py`, `solve_test.py` | Tangent PINN (negative result) and its physics-only solve test |
| `mc_costdrop.py`, `mc_restart_check.py` | Monte Carlo likelihood-ratio check and restart convergence check |
| `adequacy.py`, `floorcheck.py`, `headcheck.py`, `feat_test.py`, `feat_test2.py` | Diagnostics: is the head basis adequate at the true θ, and where in time does the physics residual floor sit (this is what motivated the decaying exponentials) |
| `apc_turnover.py` | APC level and APC share of β-catenin removal per regime (Section 2.7) |
| `aggregate.py`, `make_figures.py`, `figures/` | First-round comparison table and exploratory figures (SGN v1 basis, default-tolerance MLE, counts out of 36). Superseded by the paper figures. |
| `run_v2.sh` | SGN-PINN v2, schedules S1 and S2 for all regimes (env `PINN_DIR`, `OUT`, `DATA_SEED`, `TRUTH`) |
| `run_paper_sgn7.sh`, `run_paper_sgn42.sh` | Paper SGN runs on seeds 7 and 42, then `reselect.py` |
| `run_paper_fim2.sh` | `fim.py --save-jac` for designs base/info/dep/depb |
| `run_paper_cpu.sh`, `run_paper_cpu_resume.py` | MLE job list (multistart and Monte Carlo) and a resumable executor (4 fits at a time, 3 h limit per fit) |
| `run_paper_restart.sh` | Restart checks into `restart_checks.jsonl` |
| `run_ablate.sh`, `run_multistart.sh` | SGN ablation and variable-projection multistart on the v1 basis. Not used for reported results. |

## Reproduce

All data are synthetic and regenerated from seeds. Run from the folder with the
shared package on the path. Long jobs belong in a persistent terminal session.
Choose fresh output directories: the `run_paper_*` scripts write into
`runs/20261017_paper/`, which holds the committed results. They set
`PYTHONPATH` to the repository's `src/` themselves.

```bash
cd experiments/17-sensitivity-pinn
export PYTHONPATH=../../src
R=("Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss")

# 1. Fisher information / CRLB (CPU; ~4 min per regime on 24 cores)
python3 fim.py runs/<timestamp>_fim --workers 24
for d in base info dep depb; do python3 fim.py runs/<timestamp>_fim2_$d --design $d --save-jac --workers 24; done

# 2. Monte Carlo test of the bound (CPU; 96 fits, 2-32 min each with 14 workers)
for s in $(seq 101 124); do for r in "${R[@]}"; do
  python3 -u mle.py runs/<timestamp>_mc "$r" --start truth --seed $s --weights none \
      --fix-deltaP1 --rtol 1e-11 --fd-step 1e-5 --workers 14 --tag _s$s
done; done
python3 mc_costdrop.py ../<timestamp>_mc   # argument is relative to runs/20261017_paper/; writes runs/<timestamp>_mc_costdrop.json

# 3. Multistart MLE (common start + jitter seeds 1-3); rel weights shown, also --weights none
for r in "${R[@]}"; do for j in "" 1 2 3; do
  python3 -u mle.py runs/<timestamp>_ms "$r" --start init ${j:+--jitter-seed $j} --seed 42 \
      --weights rel --rtol 1e-11 --fd-step 1e-5 --workers 14 --tag _s42_rel
done; done
#    theta_P design fits (Fig 7B): add --weights none --design info|dep (tags _s42_none_info/_dep)

# 4. SGN-PINN (one CUDA GPU, float64; 15-27 min per regime for both schedules).
#    Needs the saved integral-PINN networks, which are not committed. Train them first
#    (CPU, 40-43 min per regime and start at 16 threads):
( cd ../10-inverse-integral-residual && for r in "${R[@]}"; do
    python3 run_boost.py --regime "$r" --variant integral --starts 1 --data-seed 7 \
        --device cpu --threads 16 --out ../17-sensitivity-pinn/runs/<timestamp>_integral_seed7; done )
PINN_DIR=runs/<timestamp>_integral_seed7 OUT=runs/<timestamp>_sgn_seed7 DATA_SEED=7 TRUTH=0 bash run_v2.sh
python3 reselect.py runs/<timestamp>_sgn_seed7 --seed 7
```

The benchmark-seed integral PINN used `--starts 3 --data-seed 42`. The paper's
integral rung is start 0 of that run. The autodiff and weighted rungs come from
`run_boost.py --variant base|plumb --starts 1 --data-seed 42`, launched by
`run_paper_ablation_cpu.sh` and `run_paper_ablation_plumb.sh` in
[10](../10-inverse-integral-residual/), with outputs written here. Paper
figures are rebuilt from the committed runs by `paper/figures/make_fig03.py`–`make_fig07.py`
and `make_figS1.py` (see [REPRODUCE.md](../../paper/REPRODUCE.md)).

## Notes

- **Solver tolerance matters.** With LSODA at 10<sup>−8</sup> and an FD step of
  10<sup>−6</sup> (the `mle.py` defaults), the finite-difference Jacobian is too
  noisy along the flattest directions, and high-WNT fits stop short of the
  optimum. Every reported fit uses rtol 10<sup>−11</sup> and step
  10<sup>−5</sup>. The default-tolerance `mc/` and `ms/` directories are kept
  only for comparison. One relative-weighted Early Adenoma fit hit the
  400-evaluation cap. It was rerun with 1500 evaluations and converged at 536
  (`ms_tight_cap/`).
- **Superseded numbers.** Earlier rounds reported counts out of 36 (out of 144
  overall): MLE from the common start 35/34/33/20 = 122/144 at default
  tolerance, and SGN v1 (no decaying exponentials) 34/35/17/11 = 97/144 from
  the earlier integral PINN. They are superseded by the table above. The
  "13–18 of 36 ceiling by any method" stated in
  [07](../07-inverse-six-condition/) and [09](../09-inverse-excitation/) came
  from an under-converged ODE fit (80 evaluations, noisy FD Jacobians). It is
  not an information limit.
- **The CRLB is local and asymptotic.** In the high-WNT regimes it is
  conservative for the weakest parameters, and the likelihood has more than one
  basin, so every local method, SGN-PINN included, is start-dependent there.
- **SGN-PINN standard errors are only as good as the convergence check.** In
  the unconverged seed-42 Severe fit, one parameter is reported 2500 times too
  precise. Check the physics residual before trusting them. When one schedule
  ended far from an ODE solution (physics residual above 10<sup>5</sup>), its
  shooting misfit was 0.5–19% higher, and the selection discarded it.
- **Starting λ.** Never start the continuation at small λ from an imperfect
  trajectory. In development, SGN started at the truth drifted away at
  λ = 10<sup>−2</sup> (36 → 18 of 36 in Advanced Adenoma) unless a head-only
  presolve came first.
- **Synthetic, generous design.** All seven states are observed at 150 times
  with noise of 0.1–2.6% of their mean levels. With only β-catenin, the two
  HOX genes and MYC observed, the expected count falls to 16–30 of 34 per
  regime. With β-catenin and MYC alone it is 4–8.5 (S1 Fig).
- Related: [11 · fisher-information](../11-fisher-information/) (θ<sub>P</sub>
  design study), [15 · hybrid-ude](../15-hybrid-ude/) (the same accounting for
  a learned term, `hybrid_fim.py`).
