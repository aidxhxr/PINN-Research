#!/usr/bin/env bash
# 2026-09-22: resume the full-pipeline bcat arm after the 2026-09-21 CUDA OOM
# (another job held 86 GB of GPU memory; Early Adenoma died in both control and bm_myc, and
# run_hybrid.sh stops at the first failed regime). Runs the three missing
# regimes for the control and then for bm_myc into the EXISTING run dirs, two
# at a time, then aggregates head-to-head.
set -u
cd "$(dirname "$0")"
CTRL=runs/20260921_172342_control_depb
HYB=runs/20260921_184115_bm_myc_depb
export PYTHONPATH="$PWD:$PWD/../../src:${PYTHONPATH:-}" MPLBACKEND=Agg
export HYBRID_DEPLETION=bcat HYBRID_PARAM=gated HYBRID_CONSTRAINT=anchored HYBRID_WD=1e-8 HYBRID_FREEZE=0 HYBRID_STATE=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/queue_depletion_bcat.log
echo "=== depb RESUME $(date -Is)" | tee -a "$LOG"
run_missing () {  # $1 variant  $2 dir  $3 HYBRID_TERM
  local v="$1" d="$2"; export HYBRID_TERM="$3"
  local pids=()
  for r in "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do
    safe="${r// /_}"
    [ -f "$d/${safe}_recovered.json" ] && { echo "  skip $v/$r (done)" | tee -a "$LOG"; continue; }
    echo "  launch $v/$r $(date -Is)" | tee -a "$LOG"
    python3 -u run_hybrid.py --regime "$r" --variant "$v" --out "$d" --starts 3 --threads 14 --device auto --refs "$d/refs.pkl" > "$d/${safe}.log" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} >= 2 )); then for p in "${pids[@]}"; do wait "$p" || echo "  FAILED pid $p" | tee -a "$LOG"; done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait "$p" || echo "  FAILED pid $p" | tee -a "$LOG"; done
  echo "  $v done $(date -Is)" | tee -a "$LOG"
}
run_missing control "$CTRL" none
run_missing bm_myc  "$HYB"  bm_myc
export HYBRID_TERM=none
{ echo; echo "########## $(basename $HYB) vs $(basename $CTRL) [bcatKO+raKO] ##########"; python3 -u aggregate_hybrid.py "$HYB" "$CTRL"; } 2>&1 | tee -a runs/hypothesis_test.txt | tee -a "$LOG" | tee runs/20260921_aggregations/depfull_bcat_bm_myc.txt
python3 -u plot_learned_term.py "$HYB" 2>&1 | tail -2
echo "=== depb RESUME DONE $(date -Is)" | tee -a "$LOG"
