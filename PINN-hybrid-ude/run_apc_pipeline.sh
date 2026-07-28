#!/usr/bin/env bash
# Calibrate one cross-severity APC mutation network, then freeze it inside the
# four-regime inverse PINN. Launch this script only inside tmux.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STARTS="${1:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
CAL_DIR="$HERE/runs/${TS}_apc_calibration"
PIPELINE_LOG="$HERE/runs/${TS}_apc_pipeline.log"
mkdir -p "$CAL_DIR"

export MPLBACKEND=Agg
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

{
  echo "[pipeline] started=$(date -Is)"
  echo "[pipeline] calibration=$CAL_DIR"
  python3 -u "$HERE/calibrate_apc_mutation.py" \
    --out "$CAL_DIR" --starts 5 --adam 2000 --lbfgs 250
  CHECKPOINT="$CAL_DIR/apc_mutation.pt"
  echo "[pipeline] frozen inverse checkpoint=$CHECKPOINT"
  bash "$HERE/run_hybrid.sh" apc_mutation_frozen "$STARTS" "$CHECKPOINT"
  echo "[pipeline] finished=$(date -Is)"
} 2>&1 | tee "$PIPELINE_LOG"
