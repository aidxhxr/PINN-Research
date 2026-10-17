# Reproducing the manuscript

`main.tex` + `sections/*.tex`, built by `./build.sh` (pdflatex + bibtex, three
passes). Every figure is made by `figures/make_*.py` from run artifacts and
writes `figures/<name>_audit.json` listing every plotted number and its source.
Shared style: `figures/style.py`; shared loaders: `figures/paperdata.py`.
All paths below are relative to the repository root. `PYTHONPATH=src` for every
script.

## Model and data
- Model: `src/wnt_pinn/model/{numpy_rhs.py,parameters.py}` (36 kinetic
  parameters, 4 regimes, 10 base conditions).
- Benchmark data and added-condition designs: `PINN-sensitivity/bench.py`
  (`make_data(regime, seed, design)`; seeds 42 benchmark, 7 held out, 101-124
  Monte Carlo).

## Figures and the runs behind them
| figure | script | runs |
|---|---|---|
| Fig 1 | `make_fig01.py`, `fig01a_network.tex` | reference solutions (computed in script) |
| Fig 2 | `make_fig02.py` | `PINN-smaller/forward-pinn-train-hybrid/runs/20260712_204546`, `PINN-smaller/forward_pinn_train/runs/20260711_203325` |
| Fig 3 | `make_fig03.py` | `PINN-sensitivity/runs/20261017_paper/fim_seed42` (`PINN-sensitivity/fim.py`) |
| Fig 4 | `make_fig04.py` | `.../20261017_paper/mc_tight` (`mle.py --start truth --weights none --fix-deltaP1 --rtol 1e-11 --fd-step 1e-5`), `mc_tight_costdrop.json` (`mc_costdrop.py`) |
| Fig 5 | `make_fig05.py` | `.../20261017_paper/ablation_seed42` (`PINN-inverse-pinn-boost/run_boost.py --variant base/plumb`), `PINN-sensitivity/runs/20261002_integral_seed{42,7}`, `.../20261017_paper/{sgn_seed42,sgn_seed7,ms_tight}` |
| Fig 6 | `make_fig06.py`, `wallclock.py` | `.../20261017_paper/sgn_seed{42,7}` (`sgn.py`, `run_v2.sh`, `reselect.py`), `ms_tight`, `wallclock.json` |
| Fig 7 | `make_fig07.py` | `PINN-fisher-matrix/runs/20260921_thetaP_design`; `.../20261017_paper/ms_tight/*_s42_none{,_info,_dep}.json` |
| Fig 8 | `make_fig08.py` | `PINN-hybrid-ude/runs/{20260727_012448_bm_myc, 20260801_204046_screen_atlas, 20260802_screen_atlas_rest, 20260921_screen_5form}` |
| Fig 9 | `make_fig09.py` | `PINN-hybrid-ude/runs/20261017_hybrid_fim` (`hybrid_fim.py`, `hybrid_fim_analysis.py`) |
| Fig 10 | `make_fig10.py` | `PINN-hybrid-ude/runs/{20260801_depletion_ab, 20260801_infoctl}` |
| Fig 11 | `make_fig11.py` | `PINN-hybrid-ude/runs/{20260802_dose_response, 20260802_dose_starts5, 20260921_dose_seeds, 20260802_protocol_test}` |
| S1 Fig | `make_figS1.py` | `.../20261017_paper/fim2_base` (`fim.py --save-jac`) |

## Tables
- Table 1 (regimes), S1 Table (parameters), S2 Table (edges): model definitions.
- S3 Table: `PINN-hybrid-ude/runs/20260921_aggregations/depfull_*.txt`.
- S4 Table: `PINN-hybrid-ude/runs/20260802_anchor_reach/anchor_reach.txt`.

## Order of computation (2026-10-17 confirmatory runs)
1. `PINN-sensitivity/fim.py` (Fisher information, ~4 min per regime on 24 cores).
2. `PINN-sensitivity/run_paper_cpu.sh` job list, executed by
   `run_paper_cpu_resume.py` (multistart and Monte Carlo MLE fits).
3. `PINN-sensitivity/run_paper_sgn7.sh`, `run_paper_sgn42.sh` (SGN-PINN, GPU).
4. `PINN-inverse-pinn-boost/run_paper_ablation_cpu.sh` (autodiff `base`, GPU) and `run_paper_ablation_plumb.sh` (weighted `plumb`, GPU); seed 42 only.
5. `PINN-hybrid-ude/hybrid_fim.py --out runs/20261017_hybrid_fim`, then
   `hybrid_fim_analysis.py`.
6. `research-paper-final/figures/make_*.py`, then `./build.sh`.

## Sources of numbers quoted only in the text

- APC level and APC share of beta-catenin removal (results_c): `PINN-sensitivity/apc_turnover.py` -> `runs/20261017_paper/apc_turnover.json`.
- Restart-convergence checks (results_b): `PINN-sensitivity/run_paper_restart.sh` -> `runs/20261017_paper/restart_checks.jsonl`.
- Full-pipeline RA->HOXA5 and MYC hybrids vs control (results_d): `PINN-hybrid-ude/runs/20260726_195830_control`, `20260726_214316_ra_h5`.
- Pre-registered dose and prospective predictions: `PREREGISTERED.md`, `dose.log`, `protocol.log` in `PINN-hybrid-ude/runs/20260802_{dose_response,protocol_test}`.
- Screen budget: the 2026-08-01 Normal atlas and depletion A/B screens were launched inline; the follow-up runs built to match them (`run_infoctl.sh`, `run_atlas_rest.sh`) pass `--adam 3000 --lbfgs 300`.
