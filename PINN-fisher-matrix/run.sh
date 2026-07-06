#!/usr/bin/env bash
# Fisher-matrix identifiability analysis. Makes a fresh timestamped run dir,
# runs all 4 regimes, writes FIM/corr .npy, per-regime figures, CSV verdicts.
set -euo pipefail
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
OUT="runs/${TS}_fisher"
mkdir -p "$OUT"
echo "run dir: $OUT"
python3 -u fisher_matrix.py "$OUT" "${1:-all}" 2>&1 | tee "$OUT/fisher.log"
