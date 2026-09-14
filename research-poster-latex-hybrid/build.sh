#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p builds
POSTER_BUILD_DIR="$(mktemp -d "builds/$(date -u +%Y%m%d_%H%M%S)_XXXXXX")"
exec > >(tee "$POSTER_BUILD_DIR/build.log") 2>&1

python3 figures/make_figures.py
for pass in 1 2; do
  if ! lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="$POSTER_BUILD_DIR" poster.tex > "$POSTER_BUILD_DIR/latex-pass-$pass.txt"; then
    tail -45 "$POSTER_BUILD_DIR/latex-pass-$pass.txt"
    exit 1
  fi
done
python3 verify.py "$POSTER_BUILD_DIR"
cp "$POSTER_BUILD_DIR/poster.pdf" poster.pdf
cp "$POSTER_BUILD_DIR/preview.png" preview.png
cp "$POSTER_BUILD_DIR/preview-detail.png" preview-detail.png
cp "$POSTER_BUILD_DIR/verification.json" verification.json
python3 package.py
echo "Finished: $POSTER_BUILD_DIR (poster.pdf and previews updated)"
