#!/usr/bin/env bash
# Build poster-v1: figures -> typst -> size assert -> squint render.
set -euo pipefail
cd "$(dirname "$0")"

TYPST="${TYPST:-$HOME/.local/bin/typst}"

echo "==> figures (all write .pdf + .svg + .png into assets/)"
cd figures
for f in fig_forward_fit fig_inv_recovery fig_bayes_marginals fig_bayes_miscal \
         fig_posterior fig_fim fig_support fig_dose_compact; do
  python3 "$f.py" >/dev/null 2>&1 && echo "    $f" || { echo "    FAILED $f"; exit 1; }
done
cd ..

echo "==> typeset"
"$TYPST" compile --font-path fonts poster-v1.typ poster-v1.pdf

echo "==> checks"
"$TYPST" compile --font-path fonts --format png --ppi 34 poster-v1.typ preview.png
"$TYPST" compile --font-path fonts --format png --ppi 60 poster-v1.typ preview_hi.png
python3 - <<'PY'
from PIL import Image
im = Image.open('preview.png')
w, h = im.size[0] / 34, im.size[1] / 34
assert abs(w - 48) < 0.05 and abs(h - 36) < 0.05, f"page is {w}x{h} in, expected 48x36"
print(f"    page size OK: {w:.0f} x {h:.0f} in")
PY

if command -v mutool >/dev/null; then
  words=$(mutool draw -F txt -o - poster-v1.pdf 2>/dev/null | wc -w)
  echo "    ~$words visible words (target: under 800)"
  test "$words" -lt 800 || { echo "    FAILED word budget"; exit 1; }
fi

echo "==> done: poster-v1.pdf + preview.png"
