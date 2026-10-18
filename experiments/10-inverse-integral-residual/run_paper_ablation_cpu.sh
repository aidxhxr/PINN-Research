#!/usr/bin/env bash
# Matched PINN ablation on CPU (the GPU is saturated by SGN-PINN): waits for the
# MLE queues to finish, then 16 single-start runs (base/plumb x 4 regimes x
# data seeds 42, 7), 4 at a time x 16 threads. Skips runs already finished.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd)/src:$PWD"
Q=../17-sensitivity-pinn/runs/20261017_paper
# (no longer waits for the MLE queues: runs alongside them at reduced parallelism)
JOBS=()
for seed in 42 7; do for v in base plumb; do for r in "Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do JOBS+=("$seed|$v|$r"); done; done; done
run_one() {
  IFS='|' read -r seed v r <<< "$1"
  OUT=../17-sensitivity-pinn/runs/20261017_paper/ablation_seed$seed; mkdir -p $OUT/$v
  safe=${r// /_}
  [ -f "$OUT/$v/${safe}_recovered.json" ] && return
  python3 -u run_boost.py --regime "$r" --variant $v --out $OUT/$v --starts 1 \
    --data-seed $seed --threads 4 --device cuda > $OUT/$v/$safe.stdout 2>&1
  echo "done $1"
}
export -f run_one
printf '%s\n' "${JOBS[@]}" | xargs -P 4 -I{} bash -c 'run_one "{}"'
echo ABLATION_DONE
