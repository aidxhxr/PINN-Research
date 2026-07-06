#!/bin/bash
# Real profile-likelihood identifiability (Raue-style, pip `identifiability` v0.5.0)
# on all 4 regimes, in parallel. Output CSV/JSON/PNG per regime into this run dir.
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"   # excite folder
export PYTHONPATH="$HERE"
OUT="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
declare -a REGS=("Normal" "Early adenoma" "Cancer-like" "Strong APC-mutant")
for r in "${REGS[@]}"; do
  safe="${r// /_}"; safe="${safe//\//_}"
  echo "launching regime: $r -> $OUT/${safe}_profile_ci.csv"
  python3 -u profile_identifiability.py "$OUT" "$r" > "$OUT/${safe}.log" 2>&1 &
done
wait
echo "ALL REGIMES DONE" | tee "$OUT/DONE"
