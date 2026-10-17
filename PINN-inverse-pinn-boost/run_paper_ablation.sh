#!/usr/bin/env bash
# Matched estimator ablation for the final paper (2026-10-17): the integral
# PINN's levers peeled off one at a time on the shared benchmark data
# (bench.py: data seeds 42 and 7), 1 start each so every rung is comparable:
#   base  = autodiff residual, unweighted, random collocation
#   plumb = + relative weighting + fixed collocation
#   (integral = + derivative-free residual: already in PINN-sensitivity/runs/20261002_integral_seed{42,7})
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/home/29/aidahxr/PINN-Research/src:$PWD
for seed in 42 7; do
  OUT=../PINN-sensitivity/runs/20261017_paper/ablation_seed$seed
  mkdir -p $OUT/base $OUT/plumb
  for v in base plumb; do
    for r in "Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do
      safe=${r// /_}
      python3 -u run_boost.py --regime "$r" --variant $v --out $OUT/$v --starts 1 \
        --data-seed $seed --threads 3 --device cuda > $OUT/$v/$safe.stdout 2>&1 &
    done
  done
  wait
done
echo ABLATION_DONE
