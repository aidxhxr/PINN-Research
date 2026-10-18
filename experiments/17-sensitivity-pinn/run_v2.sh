#!/usr/bin/env bash
# SGN-PINN v2 (final basis: PINN features + 512 RFF + 24 decaying exponentials).
# Two lambda schedules per regime; the reported estimate is chosen per regime
# by the LOWEST FINAL DATA MISFIT at lam=1e3 (no use of the truth).
set -u
cd "$(dirname "$0")"
export PYTHONPATH=/home/29/aidahxr/PINN-Research/src
P=${PINN_DIR:-../10-inverse-integral-residual/runs/20260711_203325_integral}
OUT=${OUT:-runs/20261002_sgn_v2}; mkdir -p $OUT
SEED=${DATA_SEED:-42}
for r in "Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do
  python3 -u sgn.py $OUT "$r" --pinn-dir $P --data-seed $SEED --x-start pinn --n-exp 24 \
      --lam 1e-2,1e-1,1,10,100,1e3 --iters 10 --tag _S1 >/dev/null 2>&1
  python3 -u sgn.py $OUT "$r" --pinn-dir $P --data-seed $SEED --x-start pinn --n-exp 24 \
      --presolve 15 --lam 1,10,100,1e3 --iters 12 --tag _S2 >/dev/null 2>&1
done
if [ "${TRUTH:-1}" = 1 ]; then
for r in "Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do
  python3 -u sgn.py $OUT "$r" --pinn-dir $P --data-seed $SEED --x-start truth --n-exp 24 \
      --presolve 15 --lam 1e2,1e3 --iters 10 --tag _T >/dev/null 2>&1
done
fi
echo V2_DONE
