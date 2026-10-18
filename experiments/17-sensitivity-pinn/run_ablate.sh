#!/usr/bin/env bash
# Ablations of SGN-PINN on the regimes where the PINN start is in the MLE basin.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/home/29/aidahxr/PINN-Research/src
P=../10-inverse-integral-residual/runs/20260711_203325_integral
OUT=runs/20261002_ablate; mkdir -p $OUT
L=1e-2,1e-1,1,10,100
for r in "Normal" "Early Adenoma"; do
  python3 -u sgn.py $OUT "$r" --pinn-dir $P --x-start pinn --lam $L --iters 10 --tag _full >/dev/null 2>&1
  python3 -u sgn.py $OUT "$r" --pinn-dir $P --x-start pinn --lam $L --iters 10 --n-rff 0 --tag _norff >/dev/null 2>&1
  python3 -u sgn.py $OUT "$r" --pinn-dir $P --x-start pinn --lam $L --iters 10 --freeze-heads --tag _thetaonly >/dev/null 2>&1
  python3 -u sgn.py $OUT "$r" --pinn-dir $P --x-start pinn --lam 100 --iters 50 --tag _nocont >/dev/null 2>&1
  python3 -u sgn.py $OUT "$r" --pinn-dir $P --x-start pinn --lam $L --opt adam --adam-steps 1000 --tag _adam >/dev/null 2>&1
done
echo ABLATE_DONE
