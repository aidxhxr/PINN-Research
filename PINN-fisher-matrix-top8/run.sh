#!/usr/bin/env bash
# Reduced (8 most-identifiable params) Fisher-matrix analysis. Fresh timestamped
# run dir, all 4 regimes, FIM/corr .npy + figures + CSV verdicts.
set -euo pipefail
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
OUT="runs/${TS}_fisher_top8"
mkdir -p "$OUT"
echo "run dir: $OUT"
python3 -u fisher_matrix.py "$OUT" "${1:-all}" 2>&1 | tee "$OUT/fisher.log"
