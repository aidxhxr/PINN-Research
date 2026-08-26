"""Shared poster figure style — professor's SA palette, poster typography.

Figures are authored at their final printed size (column width ~10.4 in) so
they are placed in the poster at scale 1:1. SVG output embeds text as paths
(svg.fonttype='path'), so the figures are self-contained vector art.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- palette (professor's standing request: SA-big colors) ----------------
BLUE = "#1f77b4"    # primary series / IDENT
ORANGE = "#ff7f0e"  # secondary series / WEAK
GREY = "#7f7f7f"    # neutral / reference / de-emphasis
RED = "#cc3333"     # alert / NON-IDENT / failure  (reserved, status only)
INK = "#152A3A"     # text
INK2 = "#5A6B78"    # secondary text
SURFACE = "#FFFFFF"
GRID = "#E8E6E1"    # hairline grid, one step off surface

REGIME_COLORS = {  # severity order, same 4 colors (SA-big convention)
    "Normal": BLUE,
    "Early Adenoma": ORANGE,
    "Advanced Adenoma": GREY,
    "Severe APC Loss": RED,
}

_HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(os.path.dirname(_HERE), "fonts")
ASSETS = os.path.join(os.path.dirname(_HERE), "assets")


def setup():
    for f in os.listdir(FONTS):
        if f.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(FONTS, f))
    plt.rcParams.update({
        "font.family": "Inter",
        "font.size": 15,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.labelsize": 17,
        "axes.titlesize": 19,
        "axes.titlecolor": INK,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.frameon": False,
        "legend.fontsize": 15,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "svg.fonttype": "path",
        "figure.constrained_layout.use": True,
    })


def clean(ax, x=True, y=True):
    """Recessive chrome: no top/right spines, hairline grid on one axis."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y" if y and not x else ("x" if x and not y else "both"),
            visible=True)
    ax.tick_params(length=0)


def save(fig, name):
    os.makedirs(ASSETS, exist_ok=True)
    fig.savefig(os.path.join(ASSETS, name + ".svg"))
    fig.savefig(os.path.join(ASSETS, name + ".png"), dpi=110)
    print("wrote", name, f"{fig.get_size_inches()[0]:.1f}x{fig.get_size_inches()[1]:.1f} in")
