#!/usr/bin/env bash
# Build paper.pdf and report anything that would leave the text block.
# Usage: bash build.sh [--bib]
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--bib" ]]; then
    pdflatex -interaction=nonstopmode paper.tex >/dev/null
    bibtex paper >/dev/null
fi

pdflatex -interaction=nonstopmode paper.tex >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null

echo "--- $(grep -o 'Output written on paper.pdf ([0-9]* pages' paper.log | tail -1))"

echo "--- overfull/underfull boxes:"
if grep -q "Overfull\|Underfull" paper.log; then
    grep -n "Overfull\|Underfull" paper.log
else
    echo "    none"
fi

echo "--- undefined references / citations:"
if grep -q "Warning: Reference\|Warning: Citation\|multiply defined" paper.log; then
    grep -n "Warning: Reference\|Warning: Citation\|multiply defined" paper.log
else
    echo "    none"
fi
