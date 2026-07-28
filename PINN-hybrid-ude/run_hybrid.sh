#!/usr/bin/env bash
# Launch one HYBRID VARIANT across all 4 regimes IN PARALLEL on the single GPU.
#   usage: bash run_hybrid.sh <variant> [starts]
#
# The hybrid knobs are exported BEFORE python starts because config.py reads
# them at import time (UNKNOWN shrinks when a term absorbs its parameters).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VARIANT="${1:?usage: run_hybrid.sh <variant> [starts] [learned-term-checkpoint]}"
STARTS="${2:-3}"
TERM_STATE="${3:-}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$HERE/runs/${TS}_${VARIANT}"
mkdir -p "$RUN_DIR"

export MPLBACKEND=Agg
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

# ---- variant -> hybrid env -------------------------------------------------
export HYBRID_CONSTRAINT="anchored"
export HYBRID_WD="1e-8"
export HYBRID_FREEZE="0"
export HYBRID_STATE=""
case "$VARIANT" in
  control)     export HYBRID_TERM="none" ;;
  ra_h5)       export HYBRID_TERM="ra_h5" ;;
  ra_h5_nc)    export HYBRID_TERM="ra_h5"; export HYBRID_CONSTRAINT="none" ;;
  ra_h5_wdlo)  export HYBRID_TERM="ra_h5"; export HYBRID_WD="0" ;;
  ra_h5_wdhi)  export HYBRID_TERM="ra_h5"; export HYBRID_WD="1e-6" ;;
  bm_myc_nc)   export HYBRID_TERM="bm_myc"; export HYBRID_CONSTRAINT="none" ;;
  bm_myc)      export HYBRID_TERM="bm_myc" ;;
  apc_mutation_frozen)
    if [[ -z "$TERM_STATE" || ! -f "$TERM_STATE" ]]; then
      echo "apc_mutation_frozen requires a calibration checkpoint" >&2
      exit 2
    fi
    export HYBRID_TERM="apc_mutation"
    export HYBRID_CONSTRAINT="anchored_monotone"
    export HYBRID_FREEZE="1"
    export HYBRID_STATE="$(cd "$(dirname "$TERM_STATE")" && pwd)/$(basename "$TERM_STATE")"
    ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

{
  echo "variant=$VARIANT"
  echo "HYBRID_TERM=$HYBRID_TERM"
  echo "HYBRID_CONSTRAINT=$HYBRID_CONSTRAINT"
  echo "HYBRID_WD=$HYBRID_WD"
  echo "HYBRID_FREEZE=$HYBRID_FREEZE"
  echo "HYBRID_STATE=$HYBRID_STATE"
  echo "starts=$STARTS"
  echo "started=$(date -Is)"
} | tee "$RUN_DIR/variant.env"

REFS="$RUN_DIR/refs.pkl"
echo "[prep] building reference trajectories -> $REFS"
python3 -u "$HERE/prep_refs.py" "$REFS" 2>&1 | tee "$RUN_DIR/prep.log"

REGIMES=("Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss")
THREADS=14   # 4 regimes x 14 ~= 56 of 64 cores

echo "[run] variant=$VARIANT starts=$STARTS  4 regimes in parallel"
pids=()
for r in "${REGIMES[@]}"; do
  safe="${r// /_}"; safe="${safe//\//_}"
  python3 -u "$HERE/run_hybrid.py" --regime "$r" --variant "$VARIANT" \
      --out "$RUN_DIR" --starts "$STARTS" --threads "$THREADS" \
      --refs "$REFS" > "$RUN_DIR/${safe}.log" 2>&1 &
  pids+=($!)
  echo "  launched $r (pid ${pids[-1]}) -> $RUN_DIR/${safe}.log"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "[done] variant=$VARIANT  (fail=$fail)"
python3 -u "$HERE/aggregate_hybrid.py" "$RUN_DIR" 2>&1 | tee "$RUN_DIR/summary.txt"
python3 -u "$HERE/plot_learned_term.py" "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/summary.txt" || true
