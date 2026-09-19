# PINN-Research: agent guide

Repository root: `/home/29/aidahxr/PINN-Research`
Primary branch: `main`

This is the tracked agent guide for the repository. Historical, detailed
session notes live in the git-ignored `notes/` directory when it is available.

## Standing operating rules

1. Run every new experiment, training job, or newly created approach in
   `tmux`. A short error fix or read-only diagnostic does not require a new
   session.
2. Immediately record every launched session in the active-session table
   below, including its purpose, output/log path, start date, and status.
3. Keep long research narratives in dated `notes/` files. Keep this guide
   operational and concise.
4. Preserve old run directories and checkpoints. New experiments get new
   timestamped run directories.
5. Do not edit `PINN-inverse-solve/`; prior user instructions designate it as
   the untouched historical inverse baseline.

## Active tmux sessions

| session | purpose | run/log | started | status |
|---|---|---|---|---|
| `kernel_smoke` | Optional float64 Triton/compiled physics loss; 75 checks and 10 matched runs pass; full-workload Triton 1.003x, small compiled 1.442x steady; default eager retained | `runs/20260919_153746_kernel_comparison/summary.md`; `notes/kernel-smoke_20260919_152936/` | 2026-09-19 | complete / idle |
| `commit_validation` | Pre-push checks: 89 tests and eight subtests, lint, result reproduction and wheel/sdist builds pass; poster PDFs, previews and archives verified | `notes/commit-validation_20260915_224147/` | 2026-09-15 | complete / idle |
| `population_extension` | SciPy 14-state reproduction: 202 matched comparisons, 160 LHS samples, 14 figure pairs, six-page meeting brief; five scientific tests and BDF/Radau checks pass | `runs/20260915_213018_population_extension_5128/`; `extension/results/`; `notes/population_extension_20260915_212214/` | 2026-09-15 | complete / idle |
| `poster_professor_updates` | Both posters: stemness index replaced by RA/cosine and ATRA plots; IC weight explained; APC wording, dose labels and author corrected; PDFs, ZIPs and live previews verified | `notes/poster-professor-updates_20260915_211836/`; snapshot in `poster-versions/20260915_211836_before_professor_feedback/` | 2026-09-15 | complete / idle |
| `research_reporting_validate` | Registry verification, numerical reproduction and cached poster figure builds | `notes/research-reporting-validation.log`; `notes/research-reporting-adapter-check.log` | 2026-09-14 | complete / idle |
| `repo_organization` | Package, numerical parity, run resume, artifact restoration and publication validation | `notes/organization_20260914_193541/` | 2026-09-14 | complete / idle |
| `poster_first_dynamics` | Exported Normal noATRA/ATRA inverse-PINN fits, matched Radau references and errors from saved networks | `research-poster-latex/builds/20260914_183805_normal_export/export.log` | 2026-09-14 | complete / idle |
| `poster_preview` | Serve both posters plus `/extension_plots/` and the meeting explanation at `/extenstion_report/html` on `http://localhost:8003/` through the SSH tunnel | `notes/poster-preview_20260914_172407/server.log`; extension HTML in `extension/web/` | 2026-09-14 | running |
| `poster_b_hybrid` | Beta-catenin/HOXA5 plots added with plain axes and native ATRA arrows; title names physics-informed neural networks; PDF and source ZIP verified | `research-poster-latex-hybrid/builds/20260914_175211_pshcCg/build.log`; prior version in `poster-versions/20260914_174740_before_atra_plots/` | 2026-09-14 | complete / idle |
| `poster_b_latex` | First poster: epsilonM error curve removed; eight-parameter Fisher block rewritten as methods and results; Bayesian plots enlarged; PDF/ZIP/live links verified | `research-poster-latex/builds/20260914_190110_59ULJ2/build.log`; snapshot in `poster-versions/20260914_190008_before_plain_fisher_block/` | 2026-09-14 | complete / idle |
| `hybrid_ude` | Completed control, `ra_h5` and `ra_h5_nc` queue; session closed | `PINN-hybrid-ude/runs/queue.log` | 2026-07-26 | complete / idle |
| `hybrid_bm` | Completed `bm_myc` and `bm_myc_nc` chain; session closed | `PINN-hybrid-ude/runs/chain_bm_myc.log` | 2026-07-26 | complete / idle |
| `hybrid_apc` | Calibrated a shared monotone APC-loss neural degradation term; completed the frozen four-regime inverse PINN (3 starts); session closed | `PINN-hybrid-ude/runs/20260728_232743_apc_pipeline.log`; calibration in `runs/20260728_232743_apc_calibration/`; inverse in `runs/20260728_233450_apc_mutation_frozen/` | 2026-07-28 | complete / session closed |

## Source-of-truth model

The project studies a seven-state nondimensional WNT–RA–HOX colorectal-cancer
model with state order:

```text
[b, p, h5, h13, m, r, c]
```

These are beta-catenin, APC, HOXA5, HOXA13, MYC, retinoic acid, and CYP26A1.
The maintained equations and parameter definitions are in `src/wnt_pinn/model/`.
Active integral, hybrid and inverse Bayesian modules import shared implementations;
historical experiment folders retain their recorded configurations.

- Initial state: `[0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40]`.
- Four regimes: `Normal`, `Early Adenoma`, `Advanced Adenoma`, and
  `Severe APC Loss`.
- Regime parameters `(W, thetaP)`:
  `(0.8,1.0)`, `(1.0,0.75)`, `(1.5,0.5)`, `(2.0,0.25)`.
- Time horizon: `T=150`.
- Base ATRA pulse: `tau1=40`, `tau2=88`.
- Reference solver: SciPy `solve_ivp` Radau with tight tolerances.
- The checked-in runtime configuration intentionally uses
  `rho5=1.10`, `rhoB=1.10`, `rho13=1.30`, although the manuscript contains a
  different dimensional realization. Do not silently replace runtime values
  from the manuscript.

## Current hybrid/UDE implementation

`PINN-hybrid-ude/` is the current neural-mechanistic implementation. It is
based on the successful inverse PINN with:

- sparse multi-condition data;
- one Fourier-feature state network per experimental condition;
- shared inverse biological parameters;
- a derivative-free trapezoidal/integral physics residual;
- relative state weighting, deterministic collocation, and multi-start;
- a small learned mechanistic term shared across conditions;
- honest recovery comparisons on intersected parameter sets.

The learned regulatory term is a 2×5 tanh MLP by default. Activation terms use
nonnegative output and an exact zero-input anchor. Neural L2 is calibrated to
the repository loss scale (`1e-8` by default); literature-scale values such as
`0.1–10` overwhelm this objective and collapse the learned function.

Important reproducibility constraints:

- Construct learned-term networks after state networks so their random-number
  draws do not change the state-network initializations.
- Score learned functions only over input ranges actually covered by data.
- Keep recovery denominators dynamic; a replaced mechanistic parameter must
  leave `UNKNOWN`.
- Stage-3 frozen-state refinement is effectively a no-op in the completed
  hybrid runs. Do not attribute learned-term performance to it.

Completed hybrid results at commit `6c0e70f`:

- Mechanistic control: `18/15/12/8` parameters under 10%.
- RA→HOXA5 learned term: total recovery unchanged; functional NRMSE
  `7.6–23.7%`.
- Beta-catenin→MYC learned term: functional NRMSE `6.2–9.5%`; recovery loss is
  concentrated in the MYC equation, consistent with local neural/parameter
  compensation.
- APC mutation calibration: held-out and full-curve functional NRMSE `0.12%`;
  the learned term has an exact healthy anchor and is strictly increasing.
  The completed frozen inverse run is included in the hybrid result registry.
- The `f(0)=0` anchor helps only when observed regulator values approach zero.

The current MYC/APC implementation decision and APC two-stage protocol are
recorded in tracked `docs/decisions/`, with historical working notes in `notes/`.

## Repository map

| path | purpose |
|---|---|
| `extension/` | Supplied 14-state population-model PDF, documented SciPy reproduction entry point, meeting brief, plots and numerical discrepancy audit; implementation in `src/wnt_pinn/population/` |
| `PINN-smaller/forward_pinn_train/` | supervised forward interpolation baseline |
| `PINN-smaller/forward-pinn-train-hybrid/` | sparse-data forward PINN |
| `PINN-inverse-solve/` | historical inverse baseline; do not edit |
| `PINN-inverse-multicond/` | six-condition inverse PINN and classical ODE fit |
| `PINN-inverse-multicond-excite/` | WNT/MYC information-excitation experiments |
| `PINN-inverse-pinn-boost/` | derivative-free integral-residual inverse PINN |
| `PINN-hybrid-ude/` | current learned-mechanism/UDE work |
| `PINN-fisher-matrix*/` | full and reduced Fisher identifiability analyses |
| `PINN-bayesian/` | inverse Bayesian PINN/HMC experiments |
| `PINN-forward-bayesian/` | forward posterior-predictive experiments |
| `network-diagram/` | TikZ regulatory-network schematic |
| `PINN/` | original notebooks; `run_sa_7ode.py` + `sa_results/` are the sensitivity analysis as numeric tables |
| `research-paper/` | LaTeX write-up of everything except the UDE work |
| `research-poster-latex/` | First poster: Normal treatment dynamics, inverse PINN fits, training errors, Fisher analysis and Bayesian marginals; Nathaniel Kim / Pascal K. Kataboh |
| `research-poster-latex-hybrid/` | Revised advanced poster: model/math → PINNs and identifiability → neural–mechanistic hybrid; LuaLaTeX, PDF and source ZIP |
| `poster-versions/` | Local dated snapshots of poster PDFs, sources and previews |

## Recurring findings

- Full 36-parameter recovery is ill-conditioned; low loss does not imply
  correct parameters.
- More informative perturbations outperform larger or differently regularized
  state networks.
- The matched ten-condition inverse comparison recovers 37 versus 50 of 144
  parameter-regime pairs. Integral residuals, weighting, collocation and starts
  differ together, so the gain is attributed to the full implementation.
- High WNT saturates beta-catenin and makes `thetaP` practically
  non-identifiable. FIM, profile likelihood, point recovery, and Bayesian runs
  all expose this wall.
- A learned UDE term can fit trajectories while compensating for mechanistic
  parameters in the same equation. Always report functional and parametric
  identifiability separately.

## Running experiments

New maintained experiments write to timestamped directories below root `runs/`
through `wnt-pinn run configs/<profile>.json`. Historical launchers retain their
experiment-local `runs/` paths. See `docs/experiments.md` for manifest and resume
rules. A standard durable launch is:

```bash
tmux new-session -d -s <session> -n run
tmux send-keys -t <session>:run \
  'cd /home/29/aidahxr/PINN-Research/<folder> && bash <runner>.sh' C-m
```

Inspect with:

```bash
tmux list-sessions
tmux capture-pane -pt <session>:run -S -80
tail -f <run-dir>/<log>
nvidia-smi
```

When a run finishes, mark its row complete and retain the session only if it is
useful for inspection.
