#!/usr/bin/env bash
# Anchor dose-response: three graded depletion arms, run in parallel.
#
# Fixes the weakest link in the 2026-08-01 anchor result -- it was a two-point
# A/B, which cannot show that the anchor ratio is the governing variable. Here
# the CONDITION COUNT IS FIXED at eleven for every dose, so "more data" is
# controlled by construction rather than by a separate control arm.
#
#   ra    graded retinoid restriction   -> r reaches its anchor exactly
#   wnt   graded WNT knockdown alone    -> b PLATEAUS (HOXA13 feedback)
#   bcat  knockdown of BOTH b-production arms -> the plateau breaks
#
# Predictions registered in $OUT/PREREGISTERED.md before this was run.
set -u

cd "$(dirname "$0")"
OUT="${OUT:-runs/20260802_dose_response}"
STARTS="${STARTS:-2}"
ADAM="${ADAM:-3000}"
LBFGS="${LBFGS:-300}"
REGIMES="${REGIMES:-Normal,Severe APC Loss}"
mkdir -p "$OUT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLBACKEND=Agg
# the screen chooses its term per fit via --edge; any other value here shrinks
# UNKNOWN (and INIT_GUESS with it), which both breaks control fits and makes the
# numbers incomparable with the atlas. dose_response.py refuses to start without
# it.
export HYBRID_TERM=none

echo "=== anchor dose-response  $(date -Is)" | tee -a "$OUT/dose.log"
echo "    starts=$STARTS adam=$ADAM lbfgs=$LBFGS regimes=$REGIMES" \
    | tee -a "$OUT/dose.log"

# arm -> edge scored on it. `ra` scores the RA->HOXA5 activation (basal a5,
# equation dh5); both b arms score beta-catenin->MYC (basal aM, equation dm),
# so the wnt/bcat pair differs ONLY in whether the feedback arm is knocked
# down too -- same edge, same equation, same basal parameter.
declare -A EDGE=( [ra]=ra_h5 [wnt]=bm_myc [bcat]=bm_myc )

pids=()
for arm in ra wnt bcat; do
    python3 -u dose_response.py \
        --arm "$arm" --edge "${EDGE[$arm]}" --out "$OUT" \
        --regimes "$REGIMES" --starts "$STARTS" \
        --adam "$ADAM" --lbfgs "$LBFGS" \
        >> "$OUT/dose_$arm.log" 2>&1 &
    pids+=($!)
    echo "    launched arm=$arm edge=${EDGE[$arm]} pid=${pids[-1]}" \
        | tee -a "$OUT/dose.log"
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

echo "=== arms finished (rc=$rc)  $(date -Is)" | tee -a "$OUT/dose.log"
python3 -u aggregate_dose.py "$OUT" 2>&1 | tee -a "$OUT/dose.log"
echo "=== DONE  $(date -Is)" | tee -a "$OUT/dose.log"
