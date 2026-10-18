#!/bin/bash
# Restart convergence checks behind the paper's "restart moves it by less than ..." claims.
# Writes one JSON line per fit to runs/20261017_paper/restart_checks.jsonl.
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/../../src${PYTHONPATH:+:$PYTHONPATH}"
D=runs/20261017_paper
OUT=$D/restart_checks.jsonl
for r in Advanced_Adenoma Severe_APC_Loss; do for s in 101 102 103 104 105 106; do
  echo "$D/mc_tight/${r}_truth_s${s}.json"; done; done |
  xargs -P 3 -I{} sh -c "python3 -u mc_restart_check.py {} --workers 16 2>>$D/restart.err | grep '^{' >> $OUT"
python3 -u mc_restart_check.py $D/ms_tight/Early_Adenoma_init_s42_none.json --workers 16 2>>$D/restart.err | grep '^{' >> $OUT
echo RESTART_DONE
