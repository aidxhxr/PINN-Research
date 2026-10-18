# 12 · Fisher information of an eight-parameter subproblem

If only the best-informed parameters were treated as unknown, would the
local inverse problem become well conditioned? This folder reruns the Fisher
analysis of [11](../11-fisher-information/) on eight of the 36 parameters,
with the other 28 fixed at their true values. In every regime the reduced
matrix has no near-null direction, all eight parameters are locally
identifiable, and the condition number is between about 100 and 665. That is
a statement about a conditional, local problem. No PINN has been fitted to
the reduced problem.

| | |
|---|---|
| **Status** | complete (exploratory; not used in the final paper) |
| **Period** | 2026-07 |
| **Builds on** | [11 · fisher-information](../11-fisher-information/) |
| **Paper** | not used; shown on the first poster (`publications/posters/pinn-recovery/`) |

## Result

Run `runs/20260711_203325_fisher_top8/`, registry record `fisher_information`
(`reduced` block): 7 states x 10 conditions x 200 times = 14,000 observations,
sigma = 0.002.

| regime | near-null directions | IDENT / WEAK / NON-IDENT | condition number |
|---|---:|---|---:|
| Normal | 0 | 8 / 0 / 0 | 99.7 |
| Early Adenoma | 0 | 8 / 0 / 0 | 120.9 |
| Advanced Adenoma | 0 | 8 / 0 / 0 | 229.4 |
| Severe APC Loss | 0 | 8 / 0 / 0 | 665.2 |

For comparison, every full 36-parameter matrix has one near-null direction.

## Method

- The subset is `lambdaC`, `etaBM`, `kappaBM`, `kappaBC`, `etaBC`, `etaRC`,
  `kappaB13`, `kappaR`. These are the eight parameters with the lowest
  worst-case participation in the flat subspace of the full FIM across the
  four regimes. They are mostly the beta-catenin-to-MYC, beta-catenin-to-CYP26
  and RA-to-CYP26 terms that the WNT/MYC conditions drive directly.
- The other 28 parameters are held at their true regime values.
- Otherwise the recipe matches [11](../11-fisher-information/):
  log-parameter central-difference sensitivities (relative step 1e-3) on LSODA
  solutions, `F = S^T S / sigma^2` at the truth, and the same
  null/soft/verdict thresholds. `fisher_matrix.py` here is the full-model
  script with `UNKNOWN` restricted to the subset and figure labels resized.

## Contents

| path | purpose |
|---|---|
| `fisher_matrix.py` | reduced-parameter FIM, spectra, correlations, verdicts and figures per regime |
| `run.sh` | runs all regimes into `runs/<timestamp>_fisher_top8/` |

## Reproduce

```bash
pip install -e '.[publications]'      # from the repository root
cd experiments/12-fisher-top8
bash run.sh                           # -> runs/<timestamp>_fisher_top8/, about 10 s per regime on CPU
```

`runs/` is git-ignored. The registered reduced matrices are tracked as
`results/inputs/*_FIM.npy` and hash-recorded in
[`results/registry.json`](../../results/registry.json). The summary used by the
first poster is `publications/posters/pinn-recovery/data/fisher_top8.json`.
`runs/20260706_174533_fisher_top8/` is the same analysis under the regime
names from before the rename.

## Notes

- The calculation conditions on the 28 fixed parameters and omits their
  uncertainty. The reduced problem is well conditioned partly because 28
  sloppy directions were removed by assumption.
- The selection uses the full-model FIM at the truth, so the subset is chosen
  with knowledge that a real analysis would not have.
- No reduced-parameter PINN or classical fit was run. Whether a PINN recovers
  these eight parameters when the other 28 are known remains untested.
- The same eight names are used to slice the full 36-parameter inverse-HMC
  posterior in `../13-bayesian-inverse/plot_top8_marginals.py`. Nothing is
  re-sampled there.
- The final paper reads the full matrix as a Cramér-Rao bound instead
  ([17](../17-sensitivity-pinn/), Fig 3). With only `deltaP1` fixed, most of
  the 36 parameters are already pinned to better than 5% under the benchmark
  design.
