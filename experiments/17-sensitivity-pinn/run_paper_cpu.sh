#!/usr/bin/env bash
# Paper confirmatory runs (2026-10-17), CPU. 4 fits at a time x 14 workers.
#  MC : unweighted (true iid) MLE from truth, deltaP1 known, data seeds 101-124
#       -> empirical sd vs Cramer-Rao bound (is the bound attained?)
#  MS : multistart MLE (+50% init and 3 log-jittered starts), 36 params,
#       data seeds 42 and 7, weights rel (shared objective) and none (true MLE)
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd)/src"
OUT=runs/20261017_paper
mkdir -p $OUT/mc $OUT/ms
JOBS=$OUT/cpu_jobs.txt; : > $JOBS
for s in 42 7; do for w in rel none; do for r in "Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do
  echo "python3 -u mle.py $OUT/ms '$r' --start init --seed $s --weights $w --workers 14 --tag _s${s}_$w" >> $JOBS
  for j in 1 2 3; do
    echo "python3 -u mle.py $OUT/ms '$r' --start init --jitter-seed $j --seed $s --weights $w --workers 14 --tag _s${s}_$w" >> $JOBS
  done
done; done; done
for s in $(seq 101 124); do for r in "Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do
  echo "python3 -u mle.py $OUT/mc '$r' --start truth --seed $s --weights none --fix-deltaP1 --workers 14 --tag _s$s" >> $JOBS
done; done
cat $JOBS | xargs -P 4 -I{} bash -c '{} > /dev/null 2>&1; echo "done: {}"' > $OUT/cpu_queue.log 2>&1
echo CPU_DONE >> $OUT/cpu_queue.log
