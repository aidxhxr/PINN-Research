#!/usr/bin/env bash
# 2026-09-21: the five-form constraint screen, re-run so the paper's claim
# "the basal penalty survives every constraint parameterisation" has artifacts
# behind all five forms (the 2026-08-01 table survives only for gated/sc).
# bm_myc only (narrowed 17:58 from bm_myc+ra_h5: 5-10 min per fit under GPU contention), 4 regimes, 3 starts, atlas budget.
set -u
cd "$(dirname "$0")"
OUT="${OUT:-runs/20260921_screen_5form}"
mkdir -p "$OUT"
export PYTHONPATH="$PWD:$PWD/../src:${PYTHONPATH:-}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MPLBACKEND=Agg HYBRID_TERM=none
echo "=== 5-form screen start $(date -Is)" | tee -a "$OUT/screen.log"
python3 -u screen_terms.py --out "$OUT" --refs runs/20260728_233450_apc_mutation_frozen/refs.pkl \
    --terms bm_myc --params gated,sc,sc_bounded,lin,lin_mono --regimes all \
    --starts 3 --stride 8 --adam 3000 --lbfgs 300 2>&1 | tee -a "$OUT/screen.log"
echo "=== DONE $(date -Is)" | tee -a "$OUT/screen.log"
