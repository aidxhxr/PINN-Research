#!/usr/bin/env bash
# Fisher information of the benchmark design with per-condition Jacobians
# saved (leave-one-out, partial observation, noise scaling), and of the three
# added-condition designs (info / dep / depb), seed 42. Timed (fim seconds).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/home/29/aidahxr/PINN-Research/src
for d in base info dep depb; do
  python3 -u fim.py runs/20261017_paper/fim2_$d --design $d --save-jac --workers 24
done
echo FIM2_DONE
