"""Bundle the compiled poster PDF and editable LaTeX project at the ZIP root."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
paths = [root/name for name in ["poster.pdf", "poster.tex", "README.md", "SOURCES.md", "build.sh",
                               "verify.py", "package.py", "PRESERVED_VERSION.md"]]
assets = ["pinn-architecture.tex", "model-original.pdf", "forward.pdf", "forward-window.tex", "posterior.pdf", "learned-myc.pdf", "tables.tex"]
paths += [root/"assets"/name for name in assets]
paths += list((root/"figures").glob("*.py"))
paths += list((root/"data").glob("*"))
with zipfile.ZipFile(root/"poster-latex-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(paths):
        archive.write(path, path.relative_to(root))
with zipfile.ZipFile(root/"poster-latex-source.zip") as archive:
    assert archive.testzip() is None
    for required in ["poster.pdf", "poster.tex"]+["assets/"+name for name in assets]:
        assert required in archive.namelist(), required
    assert archive.read("poster.pdf") == (root/"poster.pdf").read_bytes()
print("Packaged poster-latex-source.zip")
