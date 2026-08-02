#!/usr/bin/env bash
# Finish the edge atlas: the three regimes the 2026-08-01 run never reached.
#
# That run completed Normal (22 cells, in `screen.log`) and then died on the
# `apc_prod`/`gated` pair -- a multivariate term handed to a single-input
# parameterisation -- losing 2.8 GPU-hours of queued work. `hybrid.supports()`
# now skips invalid (term, parameterisation) cells with a printed note instead,
# so this restart is crash-safe.
#
# Runs into its OWN directory. The Normal results are not re-run; merge the two
# at analysis time (the 2026-08-01 dir has only `screen.log`, because that run
# predated the incremental `screen.json` write by 34 minutes).
set -u

cd "$(dirname "$0")"
OUT="${OUT:-runs/20260802_screen_atlas_rest}"
WAIT_LOG="${WAIT_LOG:-runs/20260802_protocol_test/protocol.log}"
REFS="${REFS:-runs/20260728_233450_apc_mutation_frozen/refs.pkl}"
REGIMES="${REGIMES:-Early Adenoma,Advanced Adenoma,Severe APC Loss}"
mkdir -p "$OUT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLBACKEND=Agg
export HYBRID_TERM=none

echo "=== atlas remainder queued  $(date -Is)" | tee -a "$OUT/atlas.log"
if [ -n "$WAIT_LOG" ]; then
    echo "    waiting for DONE in $WAIT_LOG" | tee -a "$OUT/atlas.log"
    while ! grep -q "=== DONE" "$WAIT_LOG" 2>/dev/null; do sleep 120; done
    echo "    starting  $(date -Is)" | tee -a "$OUT/atlas.log"
fi

python3 -u screen_terms.py --out "$OUT" --refs "$REFS" \
    --terms all --params gated,sc --regimes "$REGIMES" \
    --starts 2 --stride 8 --adam 3000 --lbfgs 300 \
    2>&1 | tee -a "$OUT/screen.log"

echo "=== DONE  $(date -Is)" | tee -a "$OUT/atlas.log"
