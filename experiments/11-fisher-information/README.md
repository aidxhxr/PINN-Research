# 11 · Fisher information and a design study for APC functionality

Which parameters does the ten-condition design actually inform, before any
estimator is trained? This folder computes the local Fisher information
matrix (FIM) of all 36 log-parameters at the truth in each regime, and,
in a later training-free design study, the Cramér-Rao bound (CRLB) of the
APC functionality `thetaP` under alternative designs. Every regime has exactly
one near-null direction: the structural `thetaP`-`deltaP1` degeneracy. With
`deltaP1` known, the bound on `thetaP` widens from 0.031% (Normal) to 55%
(Severe APC Loss). Conditions that perturb without lowering beta-catenin
barely change it. One WNT knockdown (`kW=0.1`) narrows it 22-fold in Severe APC Loss, a full knockout 33-fold.

| | |
|---|---|
| **Status** | complete. The paper's information budget (Fig 3, S1 Fig) is recomputed on the benchmark design in [17](../17-sensitivity-pinn/) |
| **Period** | 2026-07 (36-parameter FIM), 2026-09 (`thetaP` design study) |
| **Builds on** | [09 · inverse-excitation](../09-inverse-excitation/) (ten-condition design) |
| **Paper** | [Fig 7A, Section 2.7](../../paper/main.pdf) (`thetaP_design.py`) |

## Result

**Full 36-parameter FIM** (`fisher_matrix.py`, run `runs/20260711_203325_fisher/`,
registry record `fisher_information`): 7 states x 10 conditions x 200 times =
14,000 observations, sigma = 0.002.

| regime | near-null directions (eigenvalue < 1) | IDENT / WEAK / NON-IDENT | NON-IDENT parameters |
|---|---:|---|---|
| Normal | 1 | 13 / 22 / 1 | `deltaP1` |
| Early Adenoma | 1 | 17 / 18 / 1 | `deltaP1` |
| Advanced Adenoma | 1 | 14 / 20 / 2 | `thetaP`, `deltaP1` |
| Severe APC Loss | 1 | 14 / 21 / 1 | `thetaP` |

**`thetaP` design study** (`thetaP_design.py`, run `runs/20260921_thetaP_design/`):
CRLB relative standard deviation of `thetaP` with `deltaP1` known and the other
34 parameters free, 200 uniform times per condition.

| design | Normal | Early | Advanced | Severe |
|---|---:|---:|---:|---:|
| base design, 10 conditions (`base10`) | 0.031% | 0.21% | 6.3% | 55% |
| retinoid conditions only (`ra6`) | 1.29% | 5.29% | 51.6% | 148% |
| `base10` without the WNT-inhibition step | x2.71 | x6.30 | x2.41 | x1.32 |
| `base10` + one information-matched condition | x0.965-0.973 | x0.957-0.965 | x0.977-0.980 | x0.969-0.974 |
| `base10` + WNT knockdown `kW=0.5` | x0.90 | x0.94 | x0.47 | x0.25 |
| `base10` + WNT knockdown `kW=0.1` | x0.75 | x0.57 | x0.078 | x0.045 |
| `base10` + WNT knockout `kW=0` | x0.75 | x0.55 | x0.063 | x0.030 |

The sustained WNT-inhibition step is the only base condition whose removal
matters for `thetaP`. Leaving out any of the other three WNT/MYC conditions
changes the bound by at most x1.14. In Severe APC Loss the knockdown gains
4x at `kW=0.5`, 22x at `kW=0.1` and 33x at `kW=0`, in proportion to how far
it lowers beta-catenin. A beta-catenin double knockout (`kW=k13=0`) gives the
same gain as `kW=0`.

## Method

- Log-parameter sensitivities `S[:, j] = p_j dy/dp_j` by central differences
  (relative step 1e-3) on LSODA solutions (rtol 1e-9, atol 1e-11). The FIM is
  `S^T S / sigma^2`, evaluated at the true parameters of each regime.
- The verdict in `fisher_matrix.py` comes from the eigendecomposition. "Null"
  means eigenvalue < 1, that is relative sd > 100%. "Soft" means eigenvalue
  < 1e-4 x the largest. A parameter is NON-IDENT if its participation in the
  null subspace exceeds 0.5. It is WEAK if that participation exceeds 0.2, or
  its participation in the soft subspace exceeds 0.5.
- `thetaP_design.py` fixes `deltaP1`, which removes the one structural null
  direction (`thetaP` and `deltaP1` enter only as `deltaP1 (1 - thetaP)`). It
  then reads `sqrt((F^-1)_thetaP,thetaP)` for designs built from per-condition
  sensitivity blocks:
  - leave-one-out of the four WNT/MYC conditions;
  - an information-matched `midATRA` or `wntPulse2`, which moves the
    regulators up without leaving the beta-catenin plateau;
  - graded WNT knockdowns `kW` in {0.5, 0.25, 0.1, 0.02, 0};
  - `bcatKO` (`kW = k13 = 0`).

## Contents

| path | purpose |
|---|---|
| `fisher_matrix.py` | 36-parameter FIM, spectra, correlations, verdict CSV/JSON and figures per regime |
| `run.sh` | runs `fisher_matrix.py` for all regimes into `runs/<timestamp>_fisher/` |
| `thetaP_design.py` | training-free CRLB of `thetaP` under the designs above |
| `runs/20260921_thetaP_design/thetaP_design.json` | tracked result of the design study, the source of the paper's Fig 7A |

## Reproduce

```bash
pip install -e '.[publications]'          # from the repository root: shared model + matplotlib
cd experiments/11-fisher-information
bash run.sh                               # all regimes -> runs/<timestamp>_fisher/ (about 45 s per regime, CPU)
python3 thetaP_design.py runs/<timestamp>_thetaP_design   # about 12 s on a multi-core CPU
```

No GPU is needed. Of this folder's runs, only
`runs/20260921_thetaP_design/thetaP_design.json` is tracked. The registered
36-parameter matrices (`runs/20260711_203325_fisher/*_FIM.npy`) are git-ignored
but hash-recorded in [`results/registry.json`](../../results/registry.json) and
[`artifacts/catalog.json`](../../artifacts/catalog.json). Copies used by the
first poster are kept with its sources under
`publications/posters/pinn-recovery/data/`. The earlier
`runs/20260706_*_fisher/` runs use the regime names from before the rename.

## Notes

- These are local, asymptotic diagnostics at the truth, not global or
  structural identifiability. The condition numbers that `fisher_matrix.py`
  prints include the structural null eigenvalue, which sits at round-off
  level, so they are not a measure of conditioning.
- WEAK is a sloppiness label relative to the stiffest direction, not an
  absolute precision. Read as a CRLB with `deltaP1` known, the paper's
  benchmark-design matrices ([17](../17-sensitivity-pinn/), Fig 3) put every
  parameter below 1.3% relative sd in the two low-WNT regimes. In Advanced
  Adenoma 31 of 35 are below 5%, and 25 of 35 in Severe APC Loss.
- The design study uses 200 uniform times per condition, so its absolute
  values differ slightly from the 150-point benchmark design. On that design
  the paper reports 0.04%, 0.23%, 7.2% and 57% for the base design. Its full
  ten-condition leave-one-out (S1 Fig: x2.8, 6.3, 2.4, 1.25) and its
  two-condition pairs (Severe: 57% base, 53% information-matched, 1.8%
  depletion) were computed in [17](../17-sensitivity-pinn/).
- The fitted counterpart, a converged MLE under base, information-matched and
  depletion designs scored on the identifiable product `deltaP1 (1 - thetaP)`,
  is in [17](../17-sensitivity-pinn/) (Fig 7B).
- The eight-parameter conditional analysis is [12](../12-fisher-top8/). The
  Fisher bound of a learned term is in [15](../15-hybrid-ude/) (`hybrid_fim.py`).
