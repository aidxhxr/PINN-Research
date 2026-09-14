"""Publication-artifact checks; does not test or run the research model."""
from pathlib import Path
import json
import re
import sys
import fitz
import hashlib

out = Path(sys.argv[1])
schematic = Path(__file__).resolve().parent / "assets/model-original.pdf"
expected_schematic = "6836f765e455581fdfdf8bff7f8b65423f9fa1c051da3ed7afed3d80576ab086"
assert hashlib.sha256(schematic.read_bytes()).hexdigest() == expected_schematic, "Original model schematic was altered"
pdf = fitz.open(out / "poster.pdf")
assert len(pdf) == 1, f"Expected one page, got {len(pdf)}"
page = pdf[0]
assert abs(page.rect.width - 48*72) < .1, page.rect
assert abs(page.rect.height - 36*72) < .1, page.rect
log = (out/"poster.log").read_text()
problems = [line for line in log.splitlines() if
            re.search(r"Overfull|Missing character|LaTeX Font Warning", line)]
assert not problems, "\n".join(problems)
spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b
         for line in b["lines"] for s in line["spans"] if s["text"].strip()]
outside = [s["text"] for s in spans if not page.rect.contains(fitz.Rect(s["bbox"]))]
assert not outside, f"Text outside the page: {outside}"
unsafe = [s["text"] for s in spans if not fitz.Rect(70,70,48*72-70,36*72-70).contains(fitz.Rect(s["bbox"]))]
assert not unsafe, f"Text within one inch of trim edge: {unsafe}"
fonts = [f for f in page.get_fonts(full=True)]
assert all(f[1] != "n/a" for f in fonts), f"Unembedded fonts: {fonts}"
for name, dpi in [("preview.png",40), ("preview-detail.png",80)]:
    page.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),alpha=False).save(out/name)
report = {"pages": len(pdf), "size_inches": [page.rect.width/72, page.rect.height/72],
          "visible_words": len(page.get_text().split()), "fonts_embedded": True,
          "overfull_boxes": len(problems), "text_outside_page": outside,
          "text_inside_trim_margin": unsafe,
          "raster_images_on_page": len(page.get_images()),
          "original_model_schematic_unchanged": True,
          "column_heights": re.findall(r"POSTER-HEIGHT[^\n]+",log)}
(out/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
