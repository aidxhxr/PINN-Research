#!/usr/bin/env bash
# Wait for the control/ra_h5/ra_h5_nc queue to finish (GPU is single), then run
# the bm_myc pair -- the variant where the f(0)=0 anchor MEASURABLY binds
# (gate 0.22 vs ra_h5's 0.70), i.e. the only place H3 is actually testable.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "[chain] waiting for queue pid 3725796 ..."
while kill -0 3725796 2>/dev/null; do sleep 60; done
echo "[chain] queue finished at $(date -Is); starting bm_myc pair"
bash "$HERE/run_queue_hybrid.sh" 3 bm_myc bm_myc_nc
