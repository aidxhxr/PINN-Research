"""Fisher structure and information spectrum from the saved full matrices."""
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def make_fisher(data_dir, out_dir):
    data = np.load(data_dir / "fisher_matrices.npz")
    names, regimes = data["names"].tolist(), data["regimes"].tolist()
    matrices = data["matrices"]
    # Display the complete 36 x 36 matrix, grouped by the equation containing
    # each parameter; the original matrix order is preserved in the NPZ.
    groups = [
        (r"$b$", ["W", "eta13", "kappa13", "lambdaP", "lambda5", "kappa5"]),
        (r"$p$", ["thetaP", "epsP", "rho5", "rhoB", "rho13", "deltaP1"]),
        (r"$h_5$", ["eps5", "a5", "etaR", "kappaR", "etaM", "kappaM"]),
        (r"$h_{13}$", ["eps13", "a13", "etaB13", "kappaB13", "etaM13", "kappaM13"]),
        (r"$m$", ["epsM", "aM", "etaBM", "kappaBM"]),
        (r"$r$", ["epsR", "lambdaC"]),
        (r"$c$", ["epsC", "aC", "etaRC", "kappaRC", "etaBC", "kappaBC"]),
    ]
    ordered = [name for _, group in groups for name in group]
    assert len(ordered) == len(set(ordered)) == 36 and set(ordered) == set(names)
    order = [names.index(name) for name in ordered]
    matrix = matrices[regimes.index("Severe APC Loss")][np.ix_(order, order)]
    diagonal = np.sqrt(np.diag(matrix))
    assert np.all(diagonal > 0)
    normalized = matrix / np.outer(diagonal, diagonal)
    assert np.max(np.abs(normalized)) < 1 + 1e-8
    normalized = np.clip(normalized, -1, 1)

    width, height = 14.6, 5.7
    fig = plt.figure(figsize=(width, height), facecolor="white")
    heat = fig.add_axes([.65/width, 1.05/height, 4.05/width, 4.05/height])
    cmap = LinearSegmentedColormap.from_list("sensitivity_overlap", ["#1f77b4", "#ffffff", "#78253b"])
    mesh = heat.pcolormesh(np.arange(37), np.arange(37), normalized,
                          cmap=cmap, vmin=-1, vmax=1, shading="flat",
                          edgecolors="none", rasterized=False)
    heat.set(xlim=(0, 36), ylim=(36, 0), aspect="equal")
    edges = np.r_[0, np.cumsum([len(group) for _, group in groups])]
    centers = (edges[:-1] + edges[1:]) / 2
    labels = [label for label, _ in groups]
    heat.set_xticks(centers, labels, fontsize=25)
    heat.set_yticks(centers, labels, fontsize=25)
    heat.tick_params(length=0, pad=7)
    for edge in edges[1:-1]:
        heat.axvline(edge, color="white", lw=1.0)
        heat.axhline(edge, color="white", lw=1.0)
    heat.set_xlabel("Parameters grouped by ODE", fontsize=23, labelpad=8)
    fig.text(.65/width, 5.40/height, "Fisher matrix · Severe", fontsize=26, ha="left", va="center")
    cax = fig.add_axes([5.05/width, 1.05/height, .20/width, 4.05/height])
    cbar = fig.colorbar(mesh, cax=cax, ticks=[-1, 0, 1])
    cbar.solids.set_rasterized(False)
    cbar.ax.tick_params(labelsize=23)
    cbar.set_label(r"Normalized entry $C_{ij}$", fontsize=24, labelpad=10)

    spectrum = fig.add_axes([8.03/width, 1.05/height, 6.13/width, 4.05/height])
    raw_eigenvalues, null_counts = {}, {}
    colors = ["#1f77b4", "#ff7f0e", "#7f7f7f", "#cc3333"]
    short = ["Normal", "Early", "Advanced", "Severe"]
    floor = 1e-4
    for regime, matrix, color, label in zip(regimes, matrices, colors, short):
        values = np.linalg.eigvalsh(matrix)[::-1]
        raw_eigenvalues[regime] = values.tolist()
        null_counts[regime] = int(np.count_nonzero(values < 1))
        assert null_counts[regime] == 1
        # Numerically tiny/negative modes of a PSD matrix are not interpreted
        # as accurate magnitudes: all are explicitly shown at a display floor.
        spectrum.semilogy(np.arange(1, 37), np.maximum(values, floor),
                          color=color, marker="o", markersize=4.2,
                          lw=2.4, label=label)
    spectrum.set(xlim=(.4, 36.9), ylim=(2e-5, 2e10),
                 xticks=[1, 12, 24, 36], yticks=[1e-4, 1, 1e4, 1e8])
    spectrum.tick_params(labelsize=23)
    spectrum.set_xlabel("Eigenvalue rank", fontsize=24, labelpad=8)
    spectrum.set_ylabel(r"Information eigenvalue $\lambda$", fontsize=24, labelpad=7)
    spectrum.axhline(1, color="#596570", lw=1.2, ls=(0, (4, 3)))
    spectrum.text(2, 2.2, r"$\lambda=1$", color="#596570", fontsize=23)
    spectrum.grid(False, which="both")
    spectrum.legend(loc="upper right", ncol=2, fontsize=21, frameon=False,
                    handlelength=1.5, columnspacing=.75, handletextpad=.4)
    fig.text(8.03/width, 5.40/height, "Eigenvalues · four regimes", fontsize=26, ha="left", va="center")
    fig.savefig(out_dir / "fisher.pdf")
    fig.savefig(out_dir / "fisher.png", dpi=140)
    plt.close(fig)
    (out_dir / "fisher_audit.json").write_text(json.dumps({
        "matrix_regime": "Severe APC Loss", "matrix_size": [36, 36],
        "displayed_parameter_order": ordered,
        "normalization": "C_ij = I_ij / sqrt(I_ii * I_jj)",
        "group_sizes": [len(group) for _, group in groups],
        "eigenvalues_raw_descending": raw_eigenvalues,
        "lambda_below_1": null_counts, "display_floor": floor,
        "at_true_parameters": True, "sigma": float(data["sigma"]),
    }, indent=2) + "\n")
