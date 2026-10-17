#!/usr/bin/env bash
# Three-pass build of the final manuscript; page renders in preview/.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
grep -E "Warning: (Citation|Reference)|undefined" main.log | sort -u | head -30 || true
python3 -c "import fitz" 2>/dev/null || true
echo "pages: $(mutool info main.pdf | grep Pages)"
