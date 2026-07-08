#!/usr/bin/env bash
# Launch the Bayesian FORWARD PINN for all 4 regimes IN PARALLEL. Each regime is
# an independent OS process (HMC over the ~11k network weights, small net) so
# the 4 fan out across cores; they share the one GPU but each footprint is tiny
# (one small net + an 800x7 collocation buffer).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$HERE/runs/$(date +%Y%m%d_%H%M%S)_fwdbayes"
mkdir -p "$RUN_DIR"

export MPLBACKEND=Agg
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

NOISE="${NOISE:-0.02}"
NDATA="${NDATA:-40}"
WARMUP="${WARMUP:-1000}"
DRAWS="${DRAWS:-1000}"
COLLOC="${COLLOC:-800}"
LEAPFROG="${LEAPFROG:-40}"

echo "run_dir  = $RUN_DIR"
echo "n_data=$NDATA noise=$NOISE warmup=$WARMUP draws=$DRAWS colloc=$COLLOC leapfrog=$LEAPFROG"

pids=()
for REGIME in "Normal" "Early adenoma" "Cancer-like" "Strong APC-mutant"; do
  safe="${REGIME// /_}"
  python3 -u "$HERE/bayesian_forward.py" \
      --regime "$REGIME" --out "$RUN_DIR" \
      --n-data "$NDATA" --noise-std "$NOISE" \
      --warmup "$WARMUP" --draws "$DRAWS" --colloc "$COLLOC" \
      --leapfrog "$LEAPFROG" \
      --threads 14 --seed 0 \
      > "$RUN_DIR/${safe}.log" 2>&1 &
  pids+=($!)
  echo "  launched $REGIME (pid $!)"
  sleep 2
done

echo "waiting on ${pids[*]} ..."
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

echo "=== all regimes finished (fail=$fail) — aggregating ==="
python3 -u "$HERE/aggregate_forward.py" --run "$RUN_DIR" 2>&1 | tee "$RUN_DIR/aggregate.log"
echo "DONE -> $RUN_DIR"
