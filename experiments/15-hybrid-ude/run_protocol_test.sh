#!/usr/bin/env bash
# Prospective test of the training-free reachability table (anchor_reach.py).
#
# Waits for the dose-response sweep to finish so the two do not fight for the
# GPU, then runs the two worst production edges in the atlas under three
# protocols each: none / a LARGER perturbation that misses their anchor / the
# protocol the table prescribes. The middle arm is the control that makes this
# a test rather than a demonstration.
#
# Predictions registered in $OUT/PREREGISTERED.md before this was run.
set -u

cd "$(dirname "$0")"
OUT="${OUT:-runs/20260802_protocol_test}"
WAIT_LOG="${WAIT_LOG:-runs/20260802_dose_response/dose.log}"
STARTS="${STARTS:-2}"
ADAM="${ADAM:-3000}"
LBFGS="${LBFGS:-300}"
REGIMES="${REGIMES:-Normal,Severe APC Loss}"
mkdir -p "$OUT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLBACKEND=Agg
export HYBRID_TERM=none

echo "=== protocol test queued  $(date -Is)" | tee -a "$OUT/protocol.log"
if [ -n "$WAIT_LOG" ]; then
    echo "    waiting for DONE in $WAIT_LOG" | tee -a "$OUT/protocol.log"
    while ! grep -q "=== DONE" "$WAIT_LOG" 2>/dev/null; do sleep 60; done
    echo "    dose sweep finished, starting  $(date -Is)" \
        | tee -a "$OUT/protocol.log"
fi

# edge -> the three protocols, in order: baseline, near-miss, prescribed.
# m_h13  : bcatKO is a DOUBLE genetic knockout that still leaves m at 0.0786
# h13_b  : mycKO is a TRIPLE knockout that still leaves h13 at 0.1191
run_edge() {
    python3 -u protocol_test.py --edge "$1" --protocols "$2" --out "$OUT" \
        --regimes "$REGIMES" --starts "$STARTS" \
        --adam "$ADAM" --lbfgs "$LBFGS" >> "$OUT/protocol_$1.log" 2>&1
}

run_edge m_h13 none,bcatKO,mycKO &
p1=$!
run_edge h13_b none,mycKO,hox13KO &
p2=$!
echo "    launched m_h13 pid=$p1  h13_b pid=$p2" | tee -a "$OUT/protocol.log"

rc=0
wait $p1 || rc=1
wait $p2 || rc=1
echo "=== edges finished (rc=$rc)  $(date -Is)" | tee -a "$OUT/protocol.log"
python3 -u aggregate_protocol.py "$OUT" 2>&1 | tee -a "$OUT/protocol.log"
echo "=== DONE  $(date -Is)" | tee -a "$OUT/protocol.log"
