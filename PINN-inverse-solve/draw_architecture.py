"""
Draw the architecture of the inverse PINN in PINN-inverse-solve/.

Stylistically mirrors the forward-PINN reference figure (colored nodes,
dashed NN / DE sub-boxes, a red optimisation loop), but adapted for the
*inverse* problem:

  * the network maps  t -> 7 states  x_hat = [b, apc, h5, h13, m, r, c];
  * a block of UNKNOWN biological parameters  theta = (W, thetaP, ...)  are
    free trainable scalars that feed into the ODE residual;
  * the loss combines a DATA term (sparse, noisy observations), the DE/physics
    residual, and an initial-condition residual;
  * the backward pass updates BOTH the network weights AND theta jointly.

Output: inverse_pinn_architecture.png (300 dpi) next to this script.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

# ----- palette (matched to the reference figure) ------------------------------
ORANGE_F = "#FBE2C4"   # orange node fill
ORANGE_E = "#E8912A"   # orange node edge
BLUE_F = "#C9CEEC"     # hidden / derivative node fill
BLUE_E = "#5B6BC0"     # hidden / derivative node edge
GREEN = "#2E8B3D"      # NN box
BLUE = "#2746C9"       # DE box
RED = "#E23B2E"        # PINN loop
PURPLE = "#7A3FB0"     # parameters box
GREEN_DONE = "#A9D6A0"

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")


def node(x, y, label, fill, edge, r=0.34, fs=12, weight="normal"):
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge,
                         linewidth=1.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            zorder=4, fontweight=weight)


def arrow(p0, p1, color="black", lw=1.2, ls="-", rad=0.0,
          mut=12, zorder=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mut, color=color,
        lw=lw, linestyle=ls, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def dbox(x, y, w, h, color, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor="none", edgecolor=color, linewidth=lw,
                 linestyle=(0, (6, 4)), zorder=1))


# ===== outer PINN box =========================================================
dbox(0.25, 0.3, 14.5, 8.4, RED, lw=2.2)
ax.text(2.0, 1.05, "PINN (inverse)", color=RED, fontsize=22,
        fontweight="bold")

# ===== NN sub-box =============================================================
dbox(0.6, 3.3, 7.7, 5.0, GREEN, lw=1.8)
ax.text(1.2, 7.95, "NN", color=GREEN, fontsize=20, fontweight="bold")
ax.text(4.0, 7.95, "Hidden Layers", color="black", fontsize=15)

# input
ax.text(1.55, 6.6, "Input", fontsize=13)
node(1.55, 5.8, r"$t$", ORANGE_F, ORANGE_E, r=0.40, fs=14)

# hidden layers: two visible columns, each 4 nodes, dots between
hx1, hx2 = 3.2, 5.2
hy = [7.1, 6.3, 5.3, 4.5]
for y in hy:
    node(hx1, y, r"$\sigma$", BLUE_F, BLUE_E)
    node(hx2, y, r"$\sigma$", BLUE_F, BLUE_E)
# vertical dots within each column
ax.text(hx1, 5.85, r"$\vdots$", ha="center", va="center", fontsize=16)
ax.text(hx2, 5.85, r"$\vdots$", ha="center", va="center", fontsize=16)
# horizontal dots between columns
ax.text(4.2, 6.7, r"$\cdots$", ha="center", va="center", fontsize=18)
ax.text(4.2, 4.9, r"$\cdots$", ha="center", va="center", fontsize=18)

# output
ax.text(7.05, 6.6, "Output", fontsize=13)
node(7.05, 5.8, r"$\hat{x}$", ORANGE_F, ORANGE_E, r=0.42, fs=15)
ax.text(7.05, 5.0, r"$[b,apc,h_5,h_{13},m,r,c]$", ha="center",
        va="center", fontsize=9.5, color="#444")

# input -> hidden col1
for y in hy:
    arrow((1.9, 5.8), (hx1 - 0.34, y), lw=1.0)
# col1 -> col2 (sampled)
for y1 in hy:
    for y2 in hy:
        arrow((hx1 + 0.34, y1), (hx2 - 0.34, y2), lw=0.5,
              color="#888", mut=7)
# col2 -> output
for y in hy:
    arrow((hx2 + 0.34, y), (7.05 - 0.42, 5.8), lw=1.0)

# ===== DE sub-box =============================================================
dbox(8.7, 3.3, 5.4, 5.0, BLUE, lw=1.8)
ax.text(13.1, 7.95, "DE", color=BLUE, fontsize=20, fontweight="bold")

# derivative nodes
dnx = 10.0
node(dnx, 7.0, r"$\frac{\partial \hat{x}_1}{\partial t}$", BLUE_F, BLUE_E,
     r=0.40, fs=12)
ax.text(dnx, 5.8, r"$\vdots$", ha="center", va="center", fontsize=16)
node(dnx, 4.6, r"$\frac{\partial \hat{x}_n}{\partial t}$", BLUE_F, BLUE_E,
     r=0.40, fs=12)

# DE residual node
node(12.6, 5.8, r"$DE$", ORANGE_F, ORANGE_E, r=0.42, fs=14)

# output -> derivative nodes & DE
arrow((7.47, 5.8), (dnx - 0.40, 7.0), lw=1.1)
arrow((7.47, 5.8), (dnx - 0.40, 4.6), lw=1.1)
arrow((7.47, 5.8), (12.6 - 0.42, 5.8), lw=1.1)
# derivative nodes -> DE
arrow((dnx + 0.40, 7.0), (12.6 - 0.42, 5.8), lw=1.1)
arrow((dnx + 0.40, 4.6), (12.6 - 0.42, 5.8), lw=1.1)

# ===== UNKNOWN PARAMETERS theta (inverse-specific) ============================
pbox_x, pbox_y, pbox_w, pbox_h = 8.7, 1.5, 5.4, 1.45
dbox(pbox_x, pbox_y, pbox_w, pbox_h, PURPLE, lw=1.8)
ax.text(pbox_x + 0.15, pbox_y + pbox_h - 0.32,
        "Unknown parameters", color=PURPLE, fontsize=12, fontweight="bold")
node(10.0, pbox_y + 0.55, r"$\theta$", "#E7D7F2", PURPLE, r=0.36, fs=15)
ax.text(11.9, pbox_y + 0.62, r"$W,\ \theta_P,\ \eta_{13},\ \dots$",
        ha="center", va="center", fontsize=11, color="#333")
ax.text(11.9, pbox_y + 0.22, "(trainable scalars)",
        ha="center", va="center", fontsize=8.5, color="#666")
# theta feeds the ODE residual
arrow((10.4, pbox_y + 0.55), (12.6, 5.8 - 0.42), color=PURPLE,
      lw=1.4, rad=-0.18)

# ===== DATA block (inverse-specific) =========================================
dbx, dby = 0.6, 1.45
dbox(dbx, dby, 2.4, 1.55, ORANGE_E, lw=1.6)
ax.text(dbx + 0.2, dby + 1.25, "Data", color=ORANGE_E, fontsize=13,
        fontweight="bold")
# little scattered observation dots
import numpy as np
rng = np.random.default_rng(3)
sx = dbx + 0.35 + 1.7 * np.linspace(0, 1, 9)
sy = dby + 0.55 + 0.45 * rng.random(9)
ax.scatter(sx, sy, s=18, color=ORANGE_E, zorder=4)
ax.text(dbx + 1.2, dby + 0.18, r"sparse, noisy $y_{obs}$",
        ha="center", va="center", fontsize=9, color="#555")

# ===== LOSS + convergence =====================================================
node(7.4, 1.85, "Loss", ORANGE_F, ORANGE_E, r=0.52, fs=14)

# NN (data) residual: x_hat -> Loss  (down the middle, red)
arrow((7.05, 5.8 - 0.42), (7.4, 1.85 + 0.52), color=RED, lw=2.4, mut=16)
ax.text(7.62, 3.7, "Data Residual", color=RED, rotation=90,
        ha="center", va="center", fontsize=11)

# DE residual: DE -> Loss (red)
arrow((12.6, 5.8 - 0.42), (7.4 + 0.52, 1.95), color=RED, lw=2.4,
      mut=16, rad=0.12)
ax.text(11.2, 3.55, "DE Residual", color=RED, rotation=0,
        ha="center", va="center", fontsize=11)

# data block -> Loss
arrow((dbx + 2.4, dby + 0.7), (7.4 - 0.52, 1.85), color=ORANGE_E,
      lw=1.6, rad=-0.05)

# IC residual annotation into the loss
ax.text(7.4, 0.95, r"$\mathcal{L}=\lambda_d L_d+\lambda_p L_p+\lambda_{ic}L_{ic}$",
        ha="center", va="center", fontsize=10.5, color="#333")

# convergence diamond
dia_x, dia_y = 10.2, 1.85
ax.add_patch(plt.Polygon([[dia_x - 0.85, dia_y], [dia_x, dia_y + 0.6],
                          [dia_x + 0.85, dia_y], [dia_x, dia_y - 0.6]],
                         closed=True, facecolor=ORANGE_F,
                         edgecolor=ORANGE_E, lw=1.6, zorder=3))
ax.text(dia_x, dia_y, r"$<\epsilon\,?$", ha="center", va="center",
        fontsize=12, zorder=4)
arrow((7.92, 1.85), (dia_x - 0.85, dia_y), color=RED, lw=2.4, mut=16)

# YES -> Done
ax.add_patch(FancyBboxPatch((12.9, 1.45), 1.4, 0.8,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             facecolor=GREEN_DONE, edgecolor="#3a7d33", lw=1.6, zorder=3))
ax.text(13.6, 1.85, "Done", ha="center", va="center", fontsize=13,
        zorder=4)
arrow((dia_x + 0.85, dia_y), (12.9, 1.85), color=RED, lw=2.4, mut=16)
ax.text(12.05, 2.1, "YES", ha="center", va="center", fontsize=11,
        color=RED)

# NO -> update BOTH the network weights AND theta (inverse-specific)
ax.text(dia_x, 0.95, "NO", ha="center", va="center", fontsize=11,
        color=RED)
# loop back up the left edge to the NN box (update weights)
arrow((dia_x, dia_y - 0.6), (dia_x, 0.62), color=RED, lw=2.4, mut=14)
arrow((dia_x, 0.62), (1.0, 0.62), color=RED, lw=2.4, mut=14)
arrow((1.0, 0.62), (1.0, 3.3), color=RED, lw=2.4, mut=16)
ax.text(4.3, 0.45, "update network weights", color=RED, fontsize=10,
        ha="center")
# branch of the loop up to theta (update the unknown parameters)
arrow((9.0, 0.62), (9.0, pbox_y), color=PURPLE, lw=2.0, mut=14,
      ls=(0, (4, 3)))
ax.text(9.0, pbox_y - 0.12, "update " + r"$\theta$", color=PURPLE,
        fontsize=10, ha="center", va="top")

ax.set_title("Inverse PINN architecture  —  PINN-inverse-solve/",
             fontsize=15, fontweight="bold", pad=12)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "inverse_pinn_architecture.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved:", out)
