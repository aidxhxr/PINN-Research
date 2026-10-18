#!/usr/bin/env bash
# SGN-PINN v2 on seed 42 initialised from the integral PINN trained on exactly
# the benchmark data (runs/20261002_integral_seed42, --data-seed 42), so the
# seed-42 and seed-7 pipelines are identical. Waits for run_paper_sgn7.sh.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd)/src"
while ! grep -q SGN7_DONE runs/20261017_paper/sgn7_queue.log 2>/dev/null; do sleep 60; done
PINN_DIR=runs/20261002_integral_seed42 OUT=runs/20261017_paper/sgn_seed42 DATA_SEED=42 TRUTH=0 bash run_v2.sh
python3 -u reselect.py runs/20261017_paper/sgn_seed42 --seed 42 > runs/20261017_paper/sgn_seed42/reselect.log 2>&1
echo SGN42_DONE
