#!/usr/bin/env bash
# Serial queue of full 4-regime inverse-PINN hybrid runs for the ANCHOR-ORDER
# study (2026-08-01). Each variant fans 4 regimes out over the one GPU, so the
# variants themselves must run one after another.
#
# Every run is aggregated against the SAME mechanistic control
# (runs/20260726_195830_control) that the 2026-07 gated runs were scored
# against, so the new numbers drop straight into the existing comparison table.
#
#   usage: bash run_queue_shape.sh [starts]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STARTS="${1:-3}"
CONTROL="$HERE/runs/20260726_195830_control"
LOG="$HERE/runs/queue_shape.log"

VARIANTS=(
  "bm_myc__lin"                 # headline: does the linear anchor undo the -5?
  "ra_h5__lin"                  # falsifier: r never approaches 0, so the
                                # anchor cannot bind here whatever its order
  "m_h5__lin"                   # a MODULATOR edge: new structural class
  "ra_h5+bm_myc+b_h13__lin"     # three edges at once: hybridisation dose-response
)

echo "=== anchor-order queue: ${VARIANTS[*]}  starts=$STARTS  $(date -Is)" \
    | tee -a "$LOG"

for v in "${VARIANTS[@]}"; do
  echo "[queue] START $v  $(date -Is)" | tee -a "$LOG"
  bash "$HERE/run_hybrid.sh" "$v" "$STARTS" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "[queue] END   $v  rc=$rc  $(date -Is)" | tee -a "$LOG"

  # newest run dir for this variant -> hypothesis test against the control
  RD=$(ls -1d "$HERE"/runs/*_"$v" 2>/dev/null | tail -1)
  if [[ -n "$RD" && -d "$CONTROL" ]]; then
    {
      echo
      echo "########## $(basename "$RD")  vs  $(basename "$CONTROL") ##########"
      PYTHONPATH="$HERE" MPLBACKEND=Agg python3 -u "$HERE/aggregate_hybrid.py" \
          "$RD" "$CONTROL"
    } 2>&1 | tee -a "$HERE/runs/hypothesis_test.txt" | tee -a "$LOG"
  fi
done

echo "=== queue done $(date -Is)" | tee -a "$LOG"
