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

REGIMES=("Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss")
THREADS="${PINN_THREADS:-14}"
CONCURRENCY="${PINN_CONCURRENCY:-4}"
DEVICE="${PINN_DEVICE:-auto}"
if [[ ! "$THREADS" =~ ^[1-9][0-9]*$ || ! "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "PINN_THREADS and PINN_CONCURRENCY must be positive integers" >&2
  exit 2
fi

echo "[run] variant=$VARIANT starts=$STARTS  concurrency=$CONCURRENCY device=$DEVICE"
pids=()
fail=0
for r in "${REGIMES[@]}"; do
  safe="${r// /_}"; safe="${safe//\//_}"
  python3 -u "$HERE/run_boost.py" --regime "$r" --variant "$VARIANT" \
      --out "$RUN_DIR" --starts "$STARTS" --threads "$THREADS" --device "$DEVICE" \
      --refs "$REFS" > "$RUN_DIR/${safe}.log" 2>&1 &
  pids+=("$!")
  echo "  launched $r (pid ${pids[-1]}) -> $RUN_DIR/${safe}.log"
  if (( ${#pids[@]} >= CONCURRENCY )); then
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    pids=()
    if (( fail )); then break; fi
  fi
done

for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "[done] variant=$VARIANT  (fail=$fail)"
if (( fail )); then
  echo "A regime failed; inspect the regime logs in $RUN_DIR" >&2
  exit 1
fi
python3 -u "$HERE/aggregate.py" "$RUN_DIR" 2>&1 | tee "$RUN_DIR/summary.txt"
