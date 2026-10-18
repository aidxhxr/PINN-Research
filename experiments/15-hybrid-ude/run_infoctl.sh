#!/usr/bin/env bash
# The information-matched control arm (2026-08-01).
#
# The depletion pair adds two experiments, so its gain could be "two more
# conditions help" rather than "the anchor is now observed". This arm adds two
# conditions of the same size and kind that move every regulator UP, never
# down -- verified to leave every state's floor bit-identical to the
# 10-condition set in all four regimes. Comparing
#
#     12(depletion)  vs  12(information)
#
# instead of 12 vs 10 is what isolates the anchor from the extra data.
#
# Runs the same 4 edges and 2 regimes as the depletion A/B so the three arms
# line up cell for cell.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE" MPLBACKEND=Agg HYBRID_TERM=none

OUT="$HERE/runs/20260801_infoctl"
mkdir -p "$OUT"
LOG="$OUT/infoctl.log"

echo "=== information-matched control arm  $(date -Is)" | tee "$LOG"
HYBRID_INFOCTL=1 python3 -u prep_refs.py "$OUT/refs_info.pkl" 2>&1 | tail -2 \
    | tee -a "$LOG"

HYBRID_INFOCTL=1 python3 -u screen_terms.py \
    --out "$OUT" --refs "$OUT/refs_info.pkl" \
    --terms bm_myc,ra_h5,bc_cyp,rc_cyp --params gated \
    --regimes 'Normal,Severe APC Loss' \
    --starts 1 --stride 8 --adam 3000 --lbfgs 300 2>&1 | tee -a "$LOG"

echo "=== comparisons ===" | tee -a "$LOG"
{
  echo "--- depletion arm vs information arm (the attribution test) ---"
  python3 -u compare_screens.py \
      "12dep=$HERE/runs/20260801_depletion_ab/screen_dep" \
      "12info=$OUT"
  echo
  echo "--- information arm vs the original 10 conditions ---"
  python3 -u compare_screens.py \
      "10cond=$HERE/runs/20260801_depletion_ab/screen_nodep" \
      "12info=$OUT"
} 2>&1 | tee "$OUT/comparison.txt" | tee -a "$LOG"

echo "INFODONE  $(date -Is)" | tee -a "$LOG"
