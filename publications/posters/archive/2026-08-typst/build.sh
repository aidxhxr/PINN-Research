#!/usr/bin/env bash
# Build the advanced poster: figures -> typst -> size assert -> squint render.
set -euo pipefail
cd "$(dirname "$0")"

TYPST="${TYPST:-$HOME/.local/bin/typst}"

echo "==> figures"
cd figures
for f in fig_regimes fig_support fig_fim fig_posterior fig_param_fail \
         fig_attribution fig_dose fig_prospective; do
  python3 "$f.py" >/dev/null 2>&1 && echo "    $f" || { echo "    FAILED $f"; exit 1; }
done
# the recolored TikZ network schematic (only if the .tex changed)
if [ schematic_poster.tex -nt ../assets/schematic_poster.svg ]; then
  pdflatex -interaction=nonstopmode schematic_poster.tex >/dev/null
  mutool draw -F svg -o ../assets/schematic_poster.svg schematic_poster.pdf
fi
cd ..

echo "==> typeset"
"$TYPST" compile --font-path fonts poster.typ poster.pdf

echo "==> checks"
python3 - <<'PY'
from PIL import Image
import subprocess, sys
subprocess.run([__import__('os').path.expanduser('~/.local/bin/typst'), 'compile',
                '--font-path', 'fonts', '--format', 'png', '--ppi', '34',
                'poster.typ', 'preview.png'], check=True)
im = Image.open('preview.png')
w, h = im.size[0] / 34, im.size[1] / 34
assert abs(w - 48) < 0.05 and abs(h - 36) < 0.05, f"page is {w}x{h} in, expected 48x36"
print(f"    page size OK: {w:.0f} x {h:.0f} in")
PY

# word budget, measured on the rendered PDF (visible copy, not source)
if command -v mutool >/dev/null; then
  words=$(mutool draw -F txt -o - poster.pdf 2>/dev/null | wc -w)
  echo "    ~$words visible words (target: under ~1000)"
fi

echo "==> done: poster.pdf + preview.png"
