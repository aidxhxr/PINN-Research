#!/usr/bin/env bash
# Launch the Tier-1 Bayesian inverse PINN for all 4 regimes IN PARALLEL,
# seeded from the pinn-boost integral run's frozen state nets. Each regime is
# an independent OS process (HMC over 36 params, tiny) so the 4 fan out across
# the cores; they share the one GPU but the per-regime footprint is small
# (10 frozen 207k nets + a 4000x7x10 collocation buffer).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SEED_RUN="${SEED_RUN:-$HERE/../PINN-inverse-pinn-boost/runs/20260711_203325_integral}"
RUN_DIR="$HERE/runs/$(date +%Y%m%d_%H%M%S)_bayes"
mkdir -p "$RUN_DIR"

export MPLBACKEND=Agg
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

WARMUP="${WARMUP:-2000}"
DRAWS="${DRAWS:-2000}"
COLLOC="${COLLOC:-4000}"
LEAP="${LEAP:-20}"          # leapfrog steps per draw (longer -> better mixing on sloppy)
TACC="${TACC:-0.8}"         # HMC target accept (lower -> bigger eps, escapes tiny-step trap)

echo "seed_run = $SEED_RUN"
echo "run_dir  = $RUN_DIR"
echo "warmup=$WARMUP draws=$DRAWS colloc=$COLLOC leapfrog=$LEAP target_accept=$TACC"

pids=()
for REGIME in "Normal" "Early Adenoma" "Advanced Adenoma" "Severe APC Loss"; do
  safe="${REGIME// /_}"
  python3 -u "$HERE/bayesian_infer.py" \
      --regime "$REGIME" --seed-run "$SEED_RUN" --out "$RUN_DIR" \
      --warmup "$WARMUP" --draws "$DRAWS" --colloc "$COLLOC" \
      --leapfrog "$LEAP" --target-accept "$TACC" \
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
python3 -u "$HERE/aggregate_marginals.py" --run "$RUN_DIR" 2>&1 | tee "$RUN_DIR/aggregate.log"
echo "DONE -> $RUN_DIR"
