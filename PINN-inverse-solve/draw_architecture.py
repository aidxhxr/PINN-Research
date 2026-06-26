"""
Draw the architecture of the inverse PINN in PINN-inverse-solve/.

Visual language follows the forward-PINN reference figure (colored nodes,
dashed sub-boxes, a red optimisation loop), adapted for the *inverse* problem.

Top-to-bottom flow:

    t --NN--> x_hat = [b, apc, h5, h13, m, r, c]
    x_hat --autodiff--> dx_hat/dt           )
    x_hat, theta -----> f(x_hat, theta, t)   )--> r_p = dx_hat/dt - f  (physics)
    x_hat, observations ---------------------> data residual
    x_hat(0), x0 -----------------------------> initial-condition residual

    L = lam_d L_data + lam_ic L_ic + lam_p L_phys   ->  < eps ?  -> Done
        on NO: update BOTH the network weights AND theta, then repeat

Output: inverse_pinn_architecture.png (300 dpi) next to this script.
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

# ----- palette (matched to the reference figure) ------------------------------
ORANGE_F = "#FBE2C4"   # orange node fill
ORANGE_E = "#E8912A"   # orange node edge
BLUE_F = "#C9CEEC"     # hidden / derivative node fill
BLUE_E = "#5B6BC0"     # hidden / derivative node edge
GREEN = "#2E8B3D"      # NN box
BLUE = "#2746C9"       # physics box
RED = "#E23B2E"        # PINN optimisation loop
PURPLE = "#7A3FB0"     # unknown-parameter highlights
GREEN_DONE = "#A9D6A0"

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")


# ----- drawing helpers --------------------------------------------------------
def node(x, y, label, fill, edge, r=0.34, fs=12):
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge,
                        linewidth=1.6, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=5)


def arrow(p0, p1, color="black", lw=1.2, ls="-", rad=0.0, mut=13, z=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mut, color=color, lw=lw,
        linestyle=ls, zorder=z, shrinkA=2, shrinkB=2,
        connectionstyle=f"arc3,rad={rad}"))


def seg(p0, p1, color, lw, ls="-", z=2):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=ls,
            zorder=z, solid_capstyle="round")


def dbox(x, y, w, h, color, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor="none", edgecolor=color, linewidth=lw,
                 linestyle=(0, (6, 4)), zorder=1))


def term(cx, cy, w, h, title, expr, edge):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor="#FFF6EC", edgecolor=edge, linewidth=1.8, zorder=3))
    ax.text(cx, cy + 0.21, title, ha="center", va="center", fontsize=12.5,
            fontweight="bold", zorder=4)
    ax.text(cx, cy - 0.2, expr, ha="center", va="center", fontsize=10.5,
            zorder=4)


# ===== outer PINN box =========================================================
dbox(0.3, 0.3, 15.4, 11.4, RED, lw=2.2)
ax.text(1.7, 2.3, "PINN\n(inverse)", color=RED, fontsize=20,
        fontweight="bold", ha="center", va="center")

# =============================================================================
# 1) NN block  (t -> hidden -> x_hat)
# =============================================================================
dbox(0.7, 7.6, 6.6, 3.7, GREEN, lw=1.8)
ax.text(1.15, 10.95, "NN", color=GREEN, fontsize=19, fontweight="bold")
ax.text(3.7, 10.95, "Hidden Layers", fontsize=14)

ax.text(1.5, 9.95, "Input", fontsize=12)
node(1.5, 9.3, r"$t$", ORANGE_F, ORANGE_E, r=0.36, fs=14)

hx1, hx2 = 2.8, 4.4
hy = [10.2, 9.5, 8.6, 7.9]
for y in hy:
    node(hx1, y, r"$\sigma$", BLUE_F, BLUE_E)
    node(hx2, y, r"$\sigma$", BLUE_F, BLUE_E)
ax.text(hx1, 9.05, r"$\vdots$", ha="center", va="center", fontsize=15)
ax.text(hx2, 9.05, r"$\vdots$", ha="center", va="center", fontsize=15)
ax.text(3.6, 9.85, r"$\cdots$", ha="center", va="center", fontsize=16)
ax.text(3.6, 8.25, r"$\cdots$", ha="center", va="center", fontsize=16)

ax.text(6.4, 9.95, "Output", fontsize=12, ha="center")
node(6.4, 9.3, r"$\hat{x}$", ORANGE_F, ORANGE_E, r=0.40, fs=15)
ax.text(6.4, 8.55, r"$[b,apc,h_5,h_{13},m,r,c]$", ha="center", va="center",
        fontsize=8.5, color="#555")

for y in hy:
    arrow((1.86, 9.3), (hx1 - 0.34, y), lw=1.0)
for y1 in hy:
    for y2 in hy:
        arrow((hx1 + 0.34, y1), (hx2 - 0.34, y2), lw=0.45, color="#9aa", mut=6)
for y in hy:
    arrow((hx2 + 0.34, y), (6.4 - 0.40, 9.3), lw=1.0)

# =============================================================================
# 2) Physics block  (autodiff + ODE RHS using theta -> physics residual r_p)
# =============================================================================
dbox(8.6, 7.6, 6.8, 3.7, BLUE, lw=1.8)
ax.text(8.95, 10.95, "Physics  —  ODE residual", color=BLUE, fontsize=15,
        fontweight="bold")

node(9.9, 10.0, r"$\dfrac{\partial \hat{x}}{\partial t}$", BLUE_F, BLUE_E,
     r=0.52, fs=12)
node(9.9, 8.5, r"$f(\hat{x},\theta,t)$", BLUE_F, BLUE_E, r=0.60, fs=10.5)
node(12.5, 9.25, r"$r_p$", ORANGE_F, ORANGE_E, r=0.55, fs=14)
ax.text(12.5, 8.42, r"$=\dfrac{\partial \hat{x}}{\partial t}-f$", ha="center",
        va="center", fontsize=9.5, color="#444")

# unknown parameters theta live here, feeding the ODE right-hand side f
node(14.3, 10.1, r"$\theta$", "#E7D7F2", PURPLE, r=0.40, fs=15)
ax.text(14.3, 9.45, r"$W,\theta_P,\eta_{13},\dots$", ha="center",
        va="center", fontsize=8.5, color=PURPLE)
ax.text(14.3, 9.13, "(unknown,\ntrainable)", ha="center", va="center",
        fontsize=7.5, color=PURPLE)

# x_hat -> autodiff & f ; theta -> f ; both -> r_p
arrow((6.8, 9.45), (9.9 - 0.52, 10.0), lw=1.2)
ax.text(7.9, 10.05, "autodiff", fontsize=8.5, color="#555", ha="center")
arrow((6.8, 9.15), (9.9 - 0.60, 8.5), lw=1.2)
ax.text(7.95, 8.75, r"eval $f$", fontsize=8.5, color="#555", ha="center")
arrow((14.0, 9.78), (10.45, 8.6), color=PURPLE, lw=1.4, rad=0.12)
arrow((10.42, 10.0), (12.0, 9.45), lw=1.2)
arrow((10.5, 8.6), (12.1, 9.0), lw=1.2)

# =============================================================================
# 3) Data block (sparse, noisy observations)
# =============================================================================
dbox(0.7, 5.3, 2.3, 1.45, ORANGE_E, lw=1.6)
ax.text(0.9, 6.5, "Data", color=ORANGE_E, fontsize=12.5, fontweight="bold")
rng = np.random.default_rng(3)
sx = 0.95 + 1.6 * np.linspace(0, 1, 8)
sy = 5.7 + 0.45 * rng.random(8)
ax.scatter(sx, sy, s=16, color=ORANGE_E, zorder=4)
ax.text(1.85, 5.45, r"sparse, noisy $y_{obs}$", ha="center", va="center",
        fontsize=8.5, color="#555")

# =============================================================================
# 4) Loss terms  (data + initial-condition + physics)
# =============================================================================
Y_TERM = 4.2
term(2.8, Y_TERM, 2.7, 1.0, r"$L_{data}$", r"$\|\hat{x}(t_i)-y_i\|^2$",
     ORANGE_E)
term(6.0, Y_TERM, 2.5, 1.0, r"$L_{ic}$", r"$\|\hat{x}(0)-x_0\|^2$", ORANGE_E)
term(10.5, Y_TERM, 2.7, 1.0, r"$L_{phys}$", r"$\|r_p\|^2$", ORANGE_E)

# x_hat fans out (down) to the data + IC terms via a junction
J = (6.0, 6.9)
arrow((6.4, 8.9), J, lw=1.4)
ax.text(6.55, 7.7, r"$\hat{x}(t)$", fontsize=10, color="#444")
arrow(J, (6.0, Y_TERM + 0.5), lw=1.4)                 # -> L_ic
arrow(J, (3.6, Y_TERM + 0.5), lw=1.4, rad=0.05)       # -> L_data
# observations -> data term
arrow((3.0, 5.95), (2.5, Y_TERM + 0.5), color=ORANGE_E, lw=1.4)
# physics residual -> physics term
arrow((12.5, 8.7), (10.95, Y_TERM + 0.5), lw=1.4)
ax.text(11.95, 6.6, "physics\nresidual", fontsize=8.5, color="#555",
        ha="center", va="center")

# =============================================================================
# 5) Loss node + convergence + Done
# =============================================================================
node(7.0, 2.0, "Loss", ORANGE_F, ORANGE_E, r=0.55, fs=13)
arrow((2.8, Y_TERM - 0.5), (6.55, 2.25), color=RED, lw=2.0, rad=0.04)
arrow((6.0, Y_TERM - 0.5), (6.85, 2.55), color=RED, lw=2.0)
arrow((10.5, Y_TERM - 0.5), (7.45, 2.2), color=RED, lw=2.0, rad=-0.04)
ax.text(7.0, 0.95,
        r"$\mathcal{L}=\lambda_d L_{data}+\lambda_{ic}L_{ic}"
        r"+\lambda_p L_{phys}$",
        ha="center", va="center", fontsize=11, color="#222")

# convergence diamond
dx, dy = 9.7, 2.0
ax.add_patch(plt.Polygon([[dx - 0.85, dy], [dx, dy + 0.55],
                          [dx + 0.85, dy], [dx, dy - 0.55]], closed=True,
                         facecolor=ORANGE_F, edgecolor=ORANGE_E, lw=1.6,
                         zorder=3))
ax.text(dx, dy, r"$<\epsilon\,?$", ha="center", va="center", fontsize=12,
        zorder=4)
arrow((7.55, 2.0), (dx - 0.85, dy), color=RED, lw=2.2)

# Done
ax.add_patch(FancyBboxPatch((11.9, 1.6), 1.5, 0.8,
             boxstyle="round,pad=0.02,rounding_size=0.08",
             facecolor=GREEN_DONE, edgecolor="#3a7d33", lw=1.6, zorder=3))
ax.text(12.65, 2.0, "Done", ha="center", va="center", fontsize=13, zorder=4)
arrow((dx + 0.85, dy), (11.9, 2.0), color=RED, lw=2.2)
ax.text(11.25, 2.28, "YES", color=RED, fontsize=11, ha="center")

# =============================================================================
# 6) Backward loop : NO -> update BOTH network weights AND theta
# =============================================================================
ax.text(dx, 1.15, "NO", color=RED, fontsize=11, ha="center")
seg((dx, dy - 0.55), (dx, 0.65), RED, 2.2)        # down from diamond
seg((dx, 0.65), (0.55, 0.65), RED, 2.2)           # along the bottom
seg((0.55, 0.65), (0.55, 8.1), RED, 2.2)          # up the left edge
arrow((0.55, 8.1), (1.1, 8.1), color=RED, lw=2.2)  # into NN box
ax.text(4.7, 0.45, "update network weights", color=RED, fontsize=10.5,
        ha="center")

# branch of the same loop up to theta (clean river between NN and Physics)
seg((8.2, 0.65), (8.2, 8.0), PURPLE, 2.0, ls=(0, (4, 3)))
arrow((8.2, 8.0), (8.6, 8.0), color=PURPLE, lw=2.0, ls=(0, (4, 3)))
ax.text(8.35, 1.25, r"update $\theta$", color=PURPLE, fontsize=10.5,
        ha="left")

ax.set_title("Inverse PINN architecture  —  PINN-inverse-solve/",
             fontsize=16, fontweight="bold", pad=10)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "inverse_pinn_architecture.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved:", out)
