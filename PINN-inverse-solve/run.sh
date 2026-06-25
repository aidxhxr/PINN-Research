#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$HERE/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

export MPLBACKEND=Agg
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

python3 -u "$HERE/main.py" 2>&1 | tee train.log
