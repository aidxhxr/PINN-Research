#!/usr/bin/env bash
# 2026-09-21: the mechanistic counterpart of the hybrid attribution test.
# Classical ODE fit (all 36 parameters free, 150 noisy observations per
# condition, 4 starts) under three condition sets: the 10 base conditions,
# +2 information-matched (HYBRID_INFOCTL=1), +2 depletion (HYBRID_DEPLETION=1:
# wntKO kW=0.02 + raKO). Scored on thetaP, W and the identifiable product
# deltaP1*(1-thetaP). Tests by FIT what thetaP_design.py shows by Fisher bound.
set -u
cd "$(dirname "$0")"
ARM="$1"
OUT="runs/20260921_odefit_thetaP/$ARM"
mkdir -p "$OUT"
export PYTHONPATH="$PWD:$PWD/../../src:${PYTHONPATH:-}" OMP_NUM_THREADS=4 MPLBACKEND=Agg HYBRID_TERM=none
case "$ARM" in
  base10) ;;
  info12) export HYBRID_INFOCTL=1 ;;
  dep12)  export HYBRID_DEPLETION=1 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
echo "=== odefit $ARM start $(date -Is)" | tee "$OUT/odefit.log"
python3 -u recover_odefit.py "$OUT" 2>&1 | tee -a "$OUT/odefit.log"
echo "=== DONE $(date -Is)" | tee -a "$OUT/odefit.log"
