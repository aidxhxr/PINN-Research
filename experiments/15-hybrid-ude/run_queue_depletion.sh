#!/usr/bin/env bash
# The anchor-visiting experiment at full scale (2026-08-01).
#
# Identical hybrid, identical parameterisation, identical seeds as the 2026-07
# `bm_myc` / `ra_h5` runs -- the ONLY change is that the condition set gains two
# DEPLETION protocols (wntKO, raKO) that drive the regulator down to its f(0)=0
# anchor, so the anchor is observed rather than merely asserted.
#
# Because the data change, the depletion arm needs its own mechanistic control;
# it is the first job in the queue and every later run is scored against it.
#
#   usage: bash run_queue_depletion.sh [starts]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STARTS="${1:-3}"
LOG="$HERE/runs/queue_depletion.log"
export HYBRID_DEPLETION=1

echo "=== depletion queue  starts=$STARTS  $(date -Is)" | tee -a "$LOG"

run () {   # $1 = variant
  echo "[queue] START $1  $(date -Is)" | tee -a "$LOG"
  bash "$HERE/run_hybrid.sh" "$1" "$STARTS" 2>&1 | tee -a "$LOG"
  echo "[queue] END   $1  rc=${PIPESTATUS[0]}  $(date -Is)" | tee -a "$LOG"
  ls -1d "$HERE"/runs/*_"$1"_dep 2>/dev/null | tail -1
}

CONTROL_DIR="$(run control)"
echo "[queue] depletion control: $CONTROL_DIR" | tee -a "$LOG"

for v in bm_myc ra_h5; do
  RD="$(run "$v")"
  if [[ -n "$RD" && -n "$CONTROL_DIR" ]]; then
    {
      echo
      echo "########## $(basename "$RD")  vs  $(basename "$CONTROL_DIR") "\
           "[+2 depletion conditions] ##########"
      PYTHONPATH="$HERE" MPLBACKEND=Agg HYBRID_DEPLETION=1 \
          python3 -u "$HERE/aggregate_hybrid.py" "$RD" "$CONTROL_DIR"
    } 2>&1 | tee -a "$HERE/runs/hypothesis_test.txt" | tee -a "$LOG"
  fi
done

echo "=== depletion queue done $(date -Is)" | tee -a "$LOG"
