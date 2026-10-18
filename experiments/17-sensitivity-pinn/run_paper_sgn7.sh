#!/usr/bin/env bash
# SGN-PINN v2 on the HELD-OUT replicate (data seed 7), whose integral PINN was
# trained independently; basis/schedules were frozen on seed 42. Then the
# truth-free shooting-misfit selection (reselect.py) on both seeds.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd)/src"
PINN_DIR=runs/20261002_integral_seed7 OUT=runs/20261017_paper/sgn_seed7 DATA_SEED=7 TRUTH=0 bash run_v2.sh
python3 -u reselect.py runs/20261017_paper/sgn_seed7 --seed 7 > runs/20261017_paper/sgn_seed7/reselect.log 2>&1
python3 -u reselect.py runs/20261002_sgn_v2 --seed 42 > runs/20261017_paper/reselect_seed42.log 2>&1
echo SGN7_DONE
