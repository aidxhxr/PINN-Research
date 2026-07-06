#!/bin/bash
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$HERE"
OUT="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
declare -a REGS=("Normal" "Early adenoma" "Cancer-like" "Strong APC-mutant")
for r in "${REGS[@]}"; do
  safe="${r// /_}"; safe="${safe//\//_}"
  echo "launching regime: $r"
  python3 -u profile_identifiability.py "$OUT" "$r" > "$OUT/${safe}.log" 2>&1 &
done
wait
echo "ALL REGIMES DONE" | tee "$OUT/DONE"
