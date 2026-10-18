#!/usr/bin/env bash
# 2026-09-21: restart-noise replicates for the headline one-siRNA contrast.
# wnt vs bcat arm of bm_myc at k=0 only (the headline contrast), both regimes,
# 5 starts, three DISJOINT seed sets. Spread across seed bases = the noise of a
# whole best-of-5 selection, which is the error bar the paper needs.
set -u
cd "$(dirname "$0")"
OUT="${OUT:-runs/20260921_dose_seeds}"
mkdir -p "$OUT"
export PYTHONPATH="$PWD:$PWD/../../src:${PYTHONPATH:-}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MPLBACKEND=Agg HYBRID_TERM=none
echo "=== dose seeds start $(date -Is)" | tee -a "$OUT/seeds.log"
for base in 100 300 500; do
  for arm in wnt bcat; do
    echo "--- seed_base=$base arm=$arm $(date -Is)" | tee -a "$OUT/seeds.log"
    python3 -u dose_response.py --arm "$arm" --edge bm_myc --out "$OUT/seed$base" \
        --doses 0 --starts 5 --seed-base "$base" 2>&1 | tee -a "$OUT/seeds.log"
  done
done
echo "=== DONE $(date -Is)" | tee -a "$OUT/seeds.log"
