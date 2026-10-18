#!/usr/bin/env bash
# 2026-07-11 config-sync regen: wait for the pinn-boost integral retrain
# (tmux `regen_boost`) to finish, then launch the Bayesian inverse PINN seeded
# from that NEW integral run (frozen state nets are keyed by the new
# regime-safe-names, so the pre-rename seed dir would not resolve).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[queue] waiting for tmux regen_boost to finish ..."
while tmux has-session -t regen_boost 2>/dev/null; do sleep 30; done
echo "[queue] regen_boost gone."

SEED=$(ls -dt "$ROOT"/experiments/10-inverse-integral-residual/runs/*_integral 2>/dev/null | head -1)
if [ -z "$SEED" ] || ! ls "$SEED"/*_params.pt >/dev/null 2>&1; then
  echo "[queue] ERROR: no usable integral seed run found ($SEED). Aborting."
  exit 1
fi
echo "[queue] SEED_RUN=$SEED"
export SEED_RUN="$SEED"
cd "$ROOT/experiments/13-bayesian-inverse" && bash run_bayes.sh
