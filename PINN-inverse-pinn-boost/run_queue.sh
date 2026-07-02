#!/usr/bin/env bash
# GPU-bound => run variants SEQUENTIALLY (each already fans its 4 regimes out
# in parallel). Waits for a prior run dir's summary.txt before starting, so it
# can be launched while `integral` is still going without thrashing the GPU.
#   usage: bash run_queue.sh <wait_for_dir> <variant1> <variant2> ...
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

WAIT_DIR="${1:?need a dir to wait on (or - for none)}"; shift
if [ "$WAIT_DIR" != "-" ]; then
  echo "[queue] waiting for $WAIT_DIR/summary.txt ..."
  until [ -f "$WAIT_DIR/summary.txt" ]; do sleep 60; done
  echo "[queue] prior run finished; starting queue"
fi

for v in "$@"; do
  echo "[queue] === running variant: $v ==="
  bash "$HERE/run_boost.sh" "$v" 2
done
echo "[queue] all variants done"
