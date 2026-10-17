"""Trace a two-tone (garnet on white) raster logo into a filled vector PDF.
Usage (from this folder): python3 -I figures/vectorize_logo.py assets/swarthmore-logo-source.jpg assets/swarthmore-logo.pdf /tmp/logo-check.png
Source JPG: the official high-resolution file from swarthmore.edu/communications-office/college-logo (only format offered).
Method: darkness mask -> 4x bicubic upsample -> light blur -> iso-contour at 0.5
via matplotlib contourf (handles nested holes) -> PDF with exact raster aspect."""
import sys
import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out_pdf, out_png = sys.argv[1:4]
im = Image.open(src).convert("RGB")
w, h = im.size
arr = np.asarray(im).astype(float)
# dominant ink colour (official file): measured (179,30,59)
ink = np.array([179, 30, 59.0]); white = np.array([255, 255, 255.0])
# projection onto the white->ink axis gives a clean 0..1 coverage, immune to JPEG chroma noise
d = white - ink
cov = ((white - arr) @ d) / (d @ d)
cov = np.clip(cov, 0, 1)
mask = Image.fromarray((cov * 255).astype(np.uint8))
scale = 4
big = mask.resize((w * scale, h * scale), Image.BICUBIC).filter(ImageFilter.GaussianBlur(1.2))
m = np.asarray(big).astype(float) / 255.0

fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, frameon=False)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, m.shape[1]); ax.set_ylim(m.shape[0], 0); ax.set_aspect("equal")
ax.contourf(m, levels=[0.5, 1.5], colors=["#B31E3B"], antialiased=True)
fig.savefig(out_pdf, format="pdf", transparent=True)
fig.savefig(out_png, format="png", dpi=300, transparent=False, facecolor="white")
print("wrote", out_pdf, "and", out_png, "from", w, "x", h, "raster; ink coverage", round(float(cov.mean()), 4))
