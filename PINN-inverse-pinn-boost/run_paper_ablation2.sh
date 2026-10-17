#!/usr/bin/env bash
# Requeue of run_paper_ablation.sh after a GPU OOM (8 concurrent was too many
# next to SGN): waits for the surviving seed-42 jobs, then runs the rest at
# most 4 at a time.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/home/29/aidahxr/PINN-Research/src:$PWD
while pgrep -f "python3 -u run_boost.py.*ablation_seed42" > /dev/null; do sleep 60; done
JOBS=()
for r in "Normal" "Early Adenoma" "Severe APC Loss"; do JOBS+=("42|base|$r"); done
for v in base plumb; do for r in "Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do JOBS+=("7|$v|$r"); done; done
run_one() {
  IFS='|' read -r seed v r <<< "$1"
  OUT=../PINN-sensitivity/runs/20261017_paper/ablation_seed$seed; mkdir -p $OUT/$v
  safe=${r// /_}
  python3 -u run_boost.py --regime "$r" --variant $v --out $OUT/$v --starts 1 \
    --data-seed $seed --threads 4 --device cuda > $OUT/$v/$safe.stdout 2>&1
  echo "done $1"
}
export -f run_one
printf '%s\n' "${JOBS[@]}" | xargs -P 4 -I{} bash -c 'run_one "{}"'
echo ABLATION_DONE
