"""Single figure style for the final paper. Every make_fig*.py imports this.

Palette (the advisor's standing convention, SA-big colours):
  * regimes:  Normal #1f77b4, Early Adenoma #2ca02c, Advanced Adenoma #ff7f0e,
              Severe APC Loss #d40000 (one fixed colour per regime, everywhere);
  * verdicts: identifiable blue, weak orange, non-identifiable red;
  * estimators (ordered from weakest to strongest) on one blue ramp, with the
    Cramer-Rao bound always drawn in ink (black) -- so "darker = closer to the
    bound" reads the same in every figure;
  * hybrid arms: mechanistic control grey, hybrid at base design orange,
    hybrid with a designed perturbation blue, failure/rejection red.

Type: TeX Gyre Heros (Helvetica/Arial metric clone, as PLOS asks), 7-9 pt,
panel letters bold 10 pt upper-case. Widths: 1 column 3.5 in, 1.5 col 5.2 in,
full 7.0 in.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = HERE

# ---- palette ----------------------------------------------------------------
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#cc3333"
DEEPRED = "#d40000"
GREY = "#7f7f7f"
LIGHT = "#bbbbbb"
EDGE = "#444444"
INK = "#1a1a1a"
INK2 = "#555555"
GRID = "#e9e9e9"

REGIMES = ["Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss"]
REG_SHORT = {"Normal": "Normal", "Early Adenoma": "Early",
             "Advanced Adenoma": "Advanced", "Severe APC Loss": "Severe"}
REG_COLOR = {"Normal": BLUE, "Early Adenoma": GREEN,
             "Advanced Adenoma": ORANGE, "Severe APC Loss": DEEPRED}
VERDICT_COLOR = {"identifiable": BLUE, "weak": ORANGE, "non-identifiable": RED}

# estimator ramp (light -> dark = weaker -> stronger); bound = ink
EST_COLOR = {
    "PINN (autodiff)": "#c6dbef",
    "PINN (integral)": "#6baed6",
    "SGN-PINN": "#2171b5",
    "MLE": "#08306b",
    "CRLB": INK,
}
ARM_COLOR = {"control": GREY, "hybrid": ORANGE, "designed": BLUE, "fail": RED}

STATE_LABEL = {"b": r"$\beta$-catenin", "apc": "APC", "h5": "HOXA5",
               "h13": "HOXA13", "m": "MYC", "r": "RA", "c": "CYP26A1"}
STATES = ["b", "apc", "h5", "h13", "m", "r", "c"]

COL1, COL15, COL2 = 3.5, 5.2, 7.0


def _register_fonts():
    from matplotlib import font_manager
    d = "/usr/share/texmf/fonts/opentype/public/tex-gyre/"
    for f in ("texgyreheros-regular.otf", "texgyreheros-bold.otf",
              "texgyreheros-italic.otf", "texgyreheros-bolditalic.otf"):
        if os.path.exists(d + f):
            font_manager.fontManager.addfont(d + f)


def setup():
    _register_fonts()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "TeX Gyre Heros",
        "mathtext.it": "TeX Gyre Heros:italic",
        "mathtext.bf": "TeX Gyre Heros:bold",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, name, audit=None):
    """Write <name>.pdf (vector, used by LaTeX), <name>.png (preview) and,
    when given, <name>_audit.json listing every plotted number + its source."""
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    if audit is not None:
        with open(os.path.join(OUT, name + "_audit.json"), "w") as fh:
            json.dump(audit, fh, indent=1, default=float)
    plt.close(fig)


def regime_legend(ax, **kw):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=REG_COLOR[r], lw=1.6, label=REG_SHORT[r]) for r in REGIMES]
    return ax.legend(handles=h, **kw)


_GREEK = ["epsilon", "lambda", "kappa", "theta", "delta", "alpha", "beta",
          "gamma", "sigma", "omega", "tau", "rho", "eta", "eps", "mu", "phi"]
_GREEK_TEX = {"eps": r"\varepsilon", "epsilon": r"\varepsilon"}


def param_tex(k):
    """'kappaB13' -> '$\\kappa_{B13}$', 'aM' -> '$a_{M}$', 'W' -> '$W$'."""
    for g in _GREEK:
        if k.startswith(g):
            sub = k[len(g):]
            g_tex = _GREEK_TEX.get(g, "\\" + g)
            return f"${g_tex}_{{{sub}}}$" if sub else f"${g_tex}$"
    if len(k) > 1 and k[0] in "a":
        return f"$a_{{{k[1:]}}}$"
    return f"${k}$"
