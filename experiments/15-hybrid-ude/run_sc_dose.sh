#!/usr/bin/env bash
# 2026-09-21: does the MONOTONE + EXACT-ANCHOR form (`sc`, the Loman & Baker
# construction plus an exact anchor) behave like `gated` under anchor visiting?
# Every depletion/dose run so far is `gated`, so the paper's reconciliation
# with Loman & Baker had no artifact behind it. Null dose k=1 and full dose
# k=0, ra arm (ra_h5) and bcat arm (bm_myc), both regimes, 3 starts.
set -u
cd "$(dirname "$0")"
OUT="${OUT:-runs/20260921_dose_sc}"
mkdir -p "$OUT"
export PYTHONPATH="$PWD:$PWD/../../src:${PYTHONPATH:-}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MPLBACKEND=Agg HYBRID_TERM=none
echo "=== sc dose start $(date -Is)" | tee -a "$OUT/sc.log"
python3 -u dose_response.py --arm ra   --edge ra_h5  --out "$OUT" --param sc --doses 1,0 --starts 3 2>&1 | tee -a "$OUT/sc.log"
python3 -u dose_response.py --arm bcat --edge bm_myc --out "$OUT" --param sc --doses 1,0 --starts 3 2>&1 | tee -a "$OUT/sc.log"
python3 -u dose_response.py --arm wnt  --edge bm_myc --out "$OUT" --param sc --doses 0 --starts 3 2>&1 | tee -a "$OUT/sc.log"
echo "=== DONE $(date -Is)" | tee -a "$OUT/sc.log"
