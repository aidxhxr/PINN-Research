# Installation

Use Python 3.11 or 3.12. CI checks both versions on Linux with CPU PyTorch.
The shared package requires Python 3.11 or newer. GPU training also needs a
PyTorch build compatible with the machine's driver.

## Create an environment

From a fresh checkout:

```bash
git clone --depth 1 https://github.com/aidxhxr/PINN-Research.git
cd PINN-Research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
wnt-pinn --help
```

The base installation includes NumPy and SciPy. It supports the shared
numerical model and repository management commands. Commands that inspect
saved results, run legacy experiments, or build publications require a
repository checkout; those research inputs are not bundled into the wheel.

Choose the extra dependencies for your task:

| Extra | Install command | Purpose |
|---|---|---|
| `ml` | `python -m pip install -e '.[ml]'` | PyTorch networks and training |
| `publications` | `python -m pip install -e '.[publications]'` | Matplotlib figures and PyMuPDF verification |
| `dev` | `python -m pip install -e '.[dev]'` | Pytest, Ruff and package builds |

For development and all CPU checks, install CPU PyTorch first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[ml,dev,publications]'
make check
make reproduce
```

The test suite collects only `tests/`. It includes the existing hybrid-term
checks through a separate CPU process, so imports from historical experiment
folders cannot change the settings of other tests. Without the `ml` extra,
tests that need PyTorch are skipped; CI installs it and runs those checks.
The reproduction command reads saved numerical inputs and recomputes reported
results without retraining.

For GPU use, install PyTorch with the command from its
[official installation selector](https://pytorch.org/get-started/locally/),
then install `.[ml,dev,publications]`. Record the resolved Torch, CUDA and
driver versions with every run. Numerical agreement across machines does
not imply identical training trajectories.

`requirements.txt` is a convenience entry point for the base editable
installation. `requirements-lock.txt` preserves the historical VM environment,
including its CUDA packages. It is evidence about that environment, not a
portable lock for a fresh CPU machine. New runs record their own resolved
environment in their manifests.

## Build the posters

Install the `publications` extra and LuaLaTeX. On Debian or Ubuntu, the required
TeX packages and fonts can be installed with:

```bash
sudo apt-get install texlive-luatex texlive-latex-extra \
  texlive-fonts-recommended fonts-texgyre
make posters
```

Both poster builds read their saved data, regenerate figures, run LuaLaTeX,
verify page dimensions and layout, and package the editable sources. The
fonts are TeX Gyre Heros and TeX Gyre Pagella. No training or posterior
sampling is part of the build.

New build outputs are in timestamped `build/publications/` directories,
with each PDF under `<publication-id>/source/`. The reviewed PDFs remain at
`research-poster-latex/poster.pdf` and `research-poster-latex-hybrid/poster.pdf`. The existing VM preview links remain
`http://localhost:8003/first/poster.pdf` and
`http://localhost:8003/second/poster.pdf` while that server and the SSH tunnel
are running. The manually triggered **Build posters** GitHub workflow provides
the same build and verification steps in a fresh Linux environment.

## Check a distribution

```bash
make build
python -m venv /tmp/wnt-pinn-wheel-check
/tmp/wnt-pinn-wheel-check/bin/python -m pip install dist/*.whl
cd /tmp
/tmp/wnt-pinn-wheel-check/bin/wnt-pinn --help
```

The package workflow performs this check outside the checkout with only base
dependencies. This catches imports that accidentally rely on an editable
installation or on an undeclared PyTorch dependency.

For experiment protocols and session recording, follow
[`AGENTS.md`](../AGENTS.md) and the tracked experiment documentation.
