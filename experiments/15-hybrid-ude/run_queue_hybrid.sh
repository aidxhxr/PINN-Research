#!/usr/bin/env bash
# Run hybrid VARIANTS sequentially (each variant fans its 4 regimes out in
# parallel across the single GPU, so variants must NOT overlap).
#   usage: bash run_queue_hybrid.sh [starts] <variant> [variant ...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STARTS="${1:?usage: run_queue_hybrid.sh <starts> <variant> [...]}"
shift
echo "[queue] starts=$STARTS variants: $*"
for v in "$@"; do
  echo "=== [queue] START $v  $(date -Is) ==="
  bash "$HERE/run_hybrid.sh" "$v" "$STARTS" || echo "[queue] $v FAILED"
  echo "=== [queue] END   $v  $(date -Is) ==="
done
echo "[queue] all done $(date -Is)"

# Cross-variant comparison once every variant has landed.
CTRL="$(ls -d "$HERE"/runs/*_control 2>/dev/null | tail -1)"
if [ -n "$CTRL" ]; then
  for d in "$HERE"/runs/*/; do
    d="${d%/}"
    case "$(basename "$d")" in *_control) continue ;; esac
    [ -d "$d" ] || continue
    echo; echo "########## $(basename "$d")  vs  $(basename "$CTRL") ##########"
    python3 -u "$HERE/aggregate_hybrid.py" "$d" "$CTRL"
  done | tee "$HERE/runs/hypothesis_test.txt"
fi
