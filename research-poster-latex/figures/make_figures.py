"""Compact poster figures and native LaTeX tables from saved results.

Forward curves use saved predictions and Radau references; the schematic is reused.
No fitting, simulation, or resampling is performed by this script.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from forward_plot import make_forward
from fisher_plot import make_fisher
from normal_inference_plots import make_normal_inference

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "assets"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREY, RED = "#1f77b4", "#ff7f0e", "#7f7f7f", "#cc3333"
INK, LIGHT = "#152A3A", "#DEDDE0"
for path in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyreheros-*.otf"):
    font_manager.fontManager.addfont(path)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "DejaVu Sans"],
    "font.size": 24, "axes.labelsize": 24, "xtick.labelsize": 22,
    "ytick.labelsize": 22, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GREY,
    "axes.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "stixsans", "axes.spines.top": False,
    "axes.spines.right": False, "savefig.facecolor": "white",
})

def read(name):
    return json.loads((DATA / name).read_text())

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=110)
    plt.close(fig)

def forward():
    make_forward(DATA, OUT, [("m", r"MYC $m(\tau)$"), ("p", r"APC $p(\tau)$")], height=5.7)

def fisher():
    make_fisher(DATA, OUT)

def normal_inference():
    make_normal_inference(DATA, OUT)

def posterior():
    d = np.load(DATA / "normal_posterior.npz")
    names = list(map(str, d["names"]))
    th, dp = (d["values"][:, names.index(k)] for k in ("thetaP", "deltaP1"))
    product = np.mean(dp * (1 - th))
    fig, ax = plt.subplots(figsize=(7.65, 4.0))
    fig.subplots_adjust(left=.15, right=.965, bottom=.23, top=.95)
    ax.scatter(th, dp, color=BLUE, alpha=.24, s=7, linewidths=0)
    xs = np.linspace(.28, .9999, 1200)
    ax.plot(xs, product / (1-xs), color=INK, lw=1.5, ls=(0, (4, 3)))
    ax.scatter([1], [3.5], marker="x", color=RED, s=110, linewidths=2.2, zorder=5)
    ax.annotate("Truth", xy=(1, 3.5), xytext=(.63, 7), fontsize=22,
                color=RED, arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.1})
    ax.set(xlim=(.28, 1.025), ylim=(0, 16), xlabel=r"APC functionality, $\theta_P$",
           ylabel=r"$\delta_{P1}$", xticks=[.3, .5, .7, .9, 1.0], yticks=[0, 5, 10, 15])
    ax.grid(axis="y", color=LIGHT, lw=.6)
    ax.set_axisbelow(True)
    save(fig, "posterior")

def anchor():
    u = np.linspace(0, 1, 800)
    f1 = .85*u/(.18+u)
    z = np.minimum(u/.30, 1)
    f2 = f1 - .24*z*z*(3-2*z)
    fig, ax = plt.subplots(figsize=(14.6, 3.65))
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.79)
    ax.axvspan(.30, 1, color=BLUE, alpha=.07, lw=0)
    for ys, color, ls in [(f1, BLUE, "-"), (f2, GREY, (0, (5, 3)))]:
        ax.plot(u, ys, color=color, lw=2.6, ls=ls)
    ax.scatter([0], [0], color=INK, s=70, zorder=5)
    ax.text(.14, 1.12, "Unobserved", ha="center", transform=ax.get_xaxis_transform(),
            color=GREY, fontsize=23)
    ax.text(.66, 1.12, "Regulator range covered by data", ha="center",
            transform=ax.get_xaxis_transform(), color=BLUE, fontsize=23)
    ax.annotate("", (.65, np.interp(.65,u,f1)), (.65, np.interp(.65,u,f2)),
                arrowprops={"arrowstyle": "<->", "lw": 1.4, "color": INK})
    ax.text(.69, .53, r"offset $c$", va="center", fontsize=23)
    ax.text(.87, .74, r"$f(u)$", color=BLUE, fontsize=27)
    ax.text(.84, .27, r"$\widetilde f(u)$", color=GREY, fontsize=27)
    ax.text(.045, .025, r"$f(0)=0$", fontsize=23)
    ax.set(xlim=(-.025, 1.025), ylim=(-.03, .82), xlabel=r"Regulator, $u$",
           ylabel="Production", xticks=[0, .3, 1], xticklabels=["0", r"$u_{\min}$", ""], yticks=[])
    save(fig, "anchor")

def tables():
    tex = ["% Generated from saved results by figures/make_figures.py."]
    def macro(name, lines):
        tex.append("\\newcommand{\\"+name+"}{%\n"+"\n".join(lines)+"\n}")
    short = {"Normal":"Normal", "Early Adenoma":"Early",
             "Advanced Adenoma":"Advanced", "Severe APC Loss":"Severe"}
    rows = []
    for r in read("fim.json"):
        assert r["n_ident"]+r["n_weak"]+r["n_nonident"] == 36
        rows.append(f"{short[r['regime']]} & {r['n_ident']} & {r['n_weak']} & {r['n_nonident']} "+r"\\")
    macro("FisherCompactRows", rows)
    reduced = read("fisher_top8.json")
    summaries = reduced["summaries"]
    assert len(reduced["parameters"]) == 8
    assert [r["regime"] for r in summaries] == list(short)
    assert all(r["K"] == r["n_ident"] == 8 and r["n_null_directions"] == 0
               and r["n_obs"] == 14000 and r["sigma"] == .002 for r in summaries)
    full_null = {r["n_null_directions"] for r in read("fim.json")}
    reduced_null = {r["n_null_directions"] for r in summaries}
    assert len(full_null) == len(reduced_null) == 1
    macro("FullFisherNullCount", [str(next(iter(full_null)))])
    macro("ReducedFisherNullCount", [str(next(iter(reduced_null)))])
    macro("ReducedFisherRows", [
        "Locally identifiable & "+" & ".join(f"{r['n_ident']}/{r['K']}" for r in summaries)+r" \\",
        "Eigenvalues below 1 & "+" & ".join(str(r['n_null_directions']) for r in summaries)+r" \\",
        r"$\operatorname{cond}(\mathcal I)$ & "+" & ".join(f"{r['cond_number']:.1f}" for r in summaries)+r" \\",
    ])
    comparison = np.load(DATA/"normal_treatment_comparison.npz")
    state_names = comparison["states"].tolist()
    for i, condition in enumerate(["Untreated", "Treated"]):
        for state, name in [("m", "MYC"), ("p", "APC")]:
            error = 100*comparison["relative_l2"][i,state_names.index(state)]
            macro(condition+name+"Error", [f"{error:.2f}"])
    macro("ForwardRows", [
        f"Dense supervised & 100 & {100*read('forward_dense.json')['grand_mean_rel_l2']:.2f}"+r"\% \\",
        f"Sparse PINN & 40 & {100*read('forward_sparse.json')['grand_mean_rel_l2']:.2f}"+r"\% \\",
    ])
    inv = read("inverse_recovery.json")
    counts, rows = {key:[] for key in inv}, []
    for regime in short:
        for method in inv:
            d = inv[method][regime]
            assert len(d["true"]) == 36
            n = sum(abs(d["recovered"][k]-v)/abs(v)<.1 for k,v in d["true"].items())
            assert n == d["under10"]
            counts[method].append(n)
        a, b = counts["autodiff"][-1], counts["integral"][-1]
        rows.append(f"{regime} & {a} & {b} & +{b-a} "+r"\\")
    assert counts["autodiff"] == [16,9,6,6]
    assert counts["integral"] == [17,16,10,7]
    macro("InverseRows", rows)
    macro("AutodiffTotal", [str(sum(counts["autodiff"]))])
    macro("IntegralTotal", [str(sum(counts["integral"]))])
    rows, audit = [], []
    for name, label in [("dose_wnt.json", "WNT depletion only"),
                        ("dose_bcat.json", "WNT + HOXA13 feedback suppression")]:
        vals = [100*next(r for r in read(name) if r["regime"]==regime and r["dose"]==0)["hybrid_basal_err"]
                for regime in ("Normal","Severe APC Loss")]
        rows.append(label+" & "+" & ".join(f"{x:.3f}" for x in vals)+r" \\")
        audit.append(f"{label}: {vals}")
    macro("DoseRows", rows)
    rows = []
    for protocol, label in [("none","Baseline"), ("bcatKO",r"$\beta$-catenin depletion"),
                            ("mycKO","MYC depletion")]:
        vals = [100*next(r for r in read("protocol_m_h13.json")
                        if r["regime"]==regime and r["protocol"]==protocol)["hybrid_basal_err"]
                for regime in ("Normal","Severe APC Loss")]
        rows.append(label+" & "+" & ".join(f"{x:.3f}" for x in vals)+r" \\")
        audit.append(f"{label}: {vals}")
    macro("ProtocolRows", rows)
    (OUT/"tables.tex").write_text("\n\n".join(tex)+"\n")
    (OUT/"numerical_audit.txt").write_text("\n".join(audit)+"\n")

if __name__ == "__main__":
    for make in [normal_inference, fisher, tables]:
        make()
        print(f"generated {make.__name__}", flush=True)
