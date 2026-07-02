#!/usr/bin/env bash
# Launch one VARIANT across all 4 regimes IN PARALLEL (one process per regime)
# on the single GPU, splitting the 64 cores across them.
#   usage: bash run_boost.sh <variant> [starts]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VARIANT="${1:?usage: run_boost.sh <variant> [starts]}"
STARTS="${2:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$HERE/runs/${TS}_${VARIANT}"
mkdir -p "$RUN_DIR"

export MPLBACKEND=Agg
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

REFS="$RUN_DIR/refs.pkl"
echo "[prep] building reference trajectories -> $REFS"
python3 -u "$HERE/prep_refs.py" "$REFS" 2>&1 | tee "$RUN_DIR/prep.log"

REGIMES=("Normal" "Early adenoma" "Cancer-like" "Strong APC-mutant")
THREADS=14   # 4 regimes x 14 ~= 56 of 64 cores

echo "[run] variant=$VARIANT starts=$STARTS  4 regimes in parallel"
pids=()
for r in "${REGIMES[@]}"; do
  safe="${r// /_}"; safe="${safe//\//_}"
  python3 -u "$HERE/run_boost.py" --regime "$r" --variant "$VARIANT" \
      --out "$RUN_DIR" --starts "$STARTS" --threads "$THREADS" \
      --refs "$REFS" > "$RUN_DIR/${safe}.log" 2>&1 &
  pids+=($!)
  echo "  launched $r (pid ${pids[-1]}) -> $RUN_DIR/${safe}.log"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "[done] variant=$VARIANT  (fail=$fail)"
python3 -u "$HERE/aggregate.py" "$RUN_DIR" 2>&1 | tee "$RUN_DIR/summary.txt"
