"""The deltaP1-thetaP posterior valley (HMC samples, Normal regime).

The degeneracy is HYPERBOLIC, not linear: samples ride the curve
deltaP1*(1-thetaP) = const. So the honest evidence is the spread of the
product versus the spread of the parameter alone --
rel. sd 0.19 vs 0.91, i.e. the product is ~4.7x better constrained.
A *linear* correlation (0.32) understates a curved valley; in log space it is
-0.99.

Posterior GEOMETRY only -- this run fails its own calibration gates, so no
marginal widths are quoted anywhere on the poster.
Data: PINN-bayesian/runs/20260713_204442_bayes/Normal_posterior_samples.npz
"""
import numpy as np
import style
from style import BLUE, RED, INK, INK2, SURFACE
import matplotlib.pyplot as plt

style.setup()
d = np.load("/home/29/aidahxr/PINN-Research/PINN-bayesian/runs/"
            "20260713_204442_bayes/Normal_posterior_samples.npz")
names = [str(n) for n in d["names"]]
th = d["values"][:, names.index("thetaP")]
dp = d["values"][:, names.index("deltaP1")]
prod = dp * (1 - th)
rel_prod = prod.std() / prod.mean()
rel_dp = dp.std() / dp.mean()

fig, ax = plt.subplots(figsize=(10.4, 6.2))
ax.scatter(th, dp, s=16, color=BLUE, alpha=0.22, lw=0, zorder=3)

# the iso-product curve the samples ride
c = float(np.mean(prod))
xs = np.linspace(0.30, 0.9993, 400)
ys = c / (1 - xs)
m = ys < dp.max() * 1.08
ax.plot(xs[m], ys[m], color=INK, lw=2.4, ls="--", zorder=4)

ax.annotate(r"$\delta_{P1}(1-\theta_P)$ = const",
            (0.86, c / (1 - 0.86)), xytext=(0.36, 8.6),
            fontsize=23, color=INK, fontweight=600,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.6))

ax.scatter([1.0], [3.5], marker="X", s=320, color=RED, zorder=6,
           edgecolors="white", linewidths=2.0, clip_on=False)
ax.annotate("truth: the product\nvanishes here", (1.0, 3.5),
            xytext=(0.60, 1.2), fontsize=21, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))

ax.text(0.02, 0.97,
        "product      rel. sd  0.19\n"
        r"$\delta_{P1}$ alone   rel. sd  0.91",
        transform=ax.transAxes, fontsize=21, color=INK, va="top",
        family="IBM Plex Mono", linespacing=1.5)

ax.set_xlabel(r"$\theta_P$   (APC functionality)")
ax.set_ylabel(r"$\delta_{P1}$")
ax.set_xlim(0.28, 1.02)
ax.set_ylim(0, min(dp.max() * 1.05, 16))
style.clean(ax)
style.save(fig, "fig_posterior")
print("rel_prod %.3f rel_dp %.3f ratio %.1f" % (rel_prod, rel_dp, rel_dp / rel_prod))
