#!/usr/bin/env bash
# Build the deck. Runs lualatex twice (TikZ remember-picture needs two passes).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p build
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error \
           -output-directory=build pinn_presentation.tex > "build/pass${pass}.log" 2>&1 \
    || { echo "pdflatex failed on pass ${pass}; last errors:"; \
         grep -A4 -m5 '^!' "build/pass${pass}.log"; exit 1; }
done

cp -f build/pinn_presentation.pdf .
echo "Built pinn_presentation.pdf"
