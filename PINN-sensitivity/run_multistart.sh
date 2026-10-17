#!/usr/bin/env bash
# Equal-budget multi-start in the multimodal high-WNT regimes: VP-SGN from the
# SAME starting thetas as the classical MLE (+50% init and log-jitter seeds
# 1-3), PINN heads as the initial trajectory basis.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/home/29/aidahxr/PINN-Research/src
P=../PINN-inverse-pinn-boost/runs/20260711_203325_integral
OUT=runs/20261002_multistart; mkdir -p $OUT
for r in "Advanced Adenoma" "Severe APC Loss"; do
  python3 -u sgn.py $OUT "$r" --pinn-dir $P --x-start init --opt vp --lam 1e2 --iters 30 --inner 3 --tag _vp >/dev/null 2>&1
  for j in 1 2 3; do
    python3 -u sgn.py $OUT "$r" --pinn-dir $P --x-start init --jitter-seed $j --opt vp --lam 1e2 --iters 30 --inner 3 --tag _vp_j$j >/dev/null 2>&1
  done
done
echo MULTISTART_DONE
