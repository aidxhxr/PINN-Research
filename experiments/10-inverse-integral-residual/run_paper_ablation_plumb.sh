#!/usr/bin/env bash
# Weighted-variant (plumb) rung of the ablation, seed 42, on CPU in parallel with
# the autodiff rung on the GPU (2026-10-17). Seed-7 ablation dropped for time.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd)/src:$PWD"
OUT=../17-sensitivity-pinn/runs/20261017_paper/ablation_seed42/plumb; mkdir -p $OUT
for r in "Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do
  safe=${r// /_}
  python3 -u run_boost.py --regime "$r" --variant plumb --out $OUT --starts 1 \
    --data-seed 42 --threads 4 --device cuda > $OUT/$safe.stdout 2>&1 &
done
wait
echo PLUMB_DONE
