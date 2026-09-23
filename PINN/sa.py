# Converted from sa.ipynb (repo path: PINN/sa.ipynb) on 2026-09-23.
# Cells appear in notebook order; code is verbatim. Lines that were IPython-only
# ("!shell", "%magic") are kept but commented with "[ipython-only, skipped]".
# Run from this directory:  cd PINN && python3 sa.py

# %% [cell 1]
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from SALib.sample.morris import sample
from SALib.analyze.morris import analyze


def hill(x, K, n=1):
    x = max(x, 0.0)
    return x**n / (K**n + x**n)


def ra_input(tau, p):
    # background + periodic dietary oscillation + smooth ATRA pulse
    dietary = p["AR"] * (1.0 + np.cos(2.0 * np.pi * tau / p["TR"] - p["phi"]))
    treatment = 0.5 * p["DR"] * (
        np.tanh(p["q"] * (tau - p["tau1"])) - np.tanh(p["q"] * (tau - p["tau2"]))
    )
    return p["mu0"] + dietary + treatment


def model(tau, y, p):
    b, apc, h5, h13, m, r, c = np.maximum(y, 0.0)
    # im capping at 0.0 because biologically they can't be negative
    # also the ODE solver was overshooting
    deltaP = 1.0 + p["deltaP1"] * (1.0 - p["thetaP"])
    muR = ra_input(tau, p)

    # 7 ODE system
    db = (
        p["W"]
        + p["eta13"] * hill(h13, p["kappa13"], p["nH"])
        - b
        - p["lambdaP"] * apc * b
        - p["lambda5"] * h5 * b / (p["kappa5"] + b)
    )
    dapc = (1.0 / p["epsP"]) * (
        (1.0 + p["rho5"] * h5) / (1.0 + p["rhoB"] * b + p["rho13"] * h13)
        - deltaP * apc
    )
    dh5 = (1.0 / p["eps5"]) * (
        p["a5"] + p["etaR"] * hill(r, p["kappaR"], 1)
        - h5 - p["etaM"] * m * h5 / (p["kappaM"] + m)
    )
    dh13 = (1.0 / p["eps13"]) * (
        p["a13"]
        + p["etaB13"] * hill(b, p["kappaB13"], p["nB"])
        + p["etaM13"] * hill(m, p["kappaM13"], p["nM"])
        - h13
    )
    dm = (1.0 / p["epsM"]) * (
        p["aM"] + p["etaBM"] * hill(b, p["kappaBM"], p["nB"]) - m
    )
    dr = (1.0 / p["epsR"]) * (muR - r - p["lambdaC"] * c * r)
    dc = (1.0 / p["epsC"]) * (
        p["aC"]
        + p["etaRC"] * hill(r, p["kappaRC"], 1)
        + p["etaBC"] * hill(b, p["kappaBC"], p["nB"])
        - c
    )
    return [db, dapc, dh5, dh13, dm, dr, dc]


# all the constants. Normal regime (W, thetaP fixed as regime knobs)
BASELINE = {
    "W": 0.80, "thetaP": 1.00,
    "nB": 2, "nM": 2, "nH": 2,          # integer Hill coefficients
    "eta13": 0.75, "kappa13": 0.55, "lambdaP": 1.60, "lambda5": 1.30, "kappa5": 0.50,
    "epsP": 1.00, "rho5": 1.10, "rhoB": 1.10, "rho13": 1.30, "deltaP1": 3.50,
    "eps5": 1.20, "a5": 0.15, "etaR": 2.50, "kappaR": 0.40, "etaM": 2.50, "kappaM": 0.50,
    "eps13": 1.00, "a13": 0.18, "etaB13": 0.95, "kappaB13": 0.50,
    "etaM13": 0.55, "kappaM13": 0.50,
    "epsM": 0.60, "aM": 0.18, "etaBM": 1.35, "kappaBM": 0.50,
    "epsR": 0.40, "lambdaC": 0.85,
    "epsC": 0.80, "aC": 0.08, "etaRC": 1.50, "kappaRC": 0.50,
    "etaBC": 1.50, "kappaBC": 0.50,
    "mu0": 0.35, "AR": 0.04, "TR": 24.0, "phi": 0.0,
    "DR": 1.50, "q": 0.30, "tau1": 40.0, "tau2": 80.0,
    "alpha13": 1.00, "alpha5": 1.00,
}

# skip integers, regime knobs, and timing params we HAVE
VARY = [
    "eta13", "kappa13", "lambdaP", "lambda5", "kappa5",
    "epsP", "rho5", "rhoB", "rho13", "deltaP1",
    "eps5", "a5", "etaR", "kappaR", "etaM", "kappaM",
    "eps13", "a13", "etaB13", "kappaB13", "etaM13", "kappaM13",
    "epsM", "aM", "etaBM", "kappaBM",
    "epsR", "lambdaC",
    "epsC", "aC", "etaRC", "kappaRC", "etaBC", "kappaBC",
    "mu0", "AR", "DR",
    "alpha13", "alpha5",
]

# SALib wants this exact format
# i took the 30% variation for bounds that is later used for sampling; 
problem = {
    "num_vars": len(VARY),
    "names": VARY,
    "bounds": [[BASELINE[k] * 0.7, BASELINE[k] * 1.3] for k in VARY],  
}

Y0 = [0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40]
TAU_SPAN = (0.0, 150.0)

OUTPUT_NAMES = ["b-catenin", "APC", "HOXA5", "HOXA13", "MYC", "RA", "CYP26A1", "Stemness"]


def run_model(params, n_eval=500):
    tau_eval = np.linspace(*TAU_SPAN, n_eval)
    # "try" is good for catching exceptions
    # because LSODA wasn failing for some
    # we can change it later
    # idk what works the best for 7-element ODE
    try:
        sol = solve_ivp(
            lambda t, y: model(t, y, params),
            TAU_SPAN, Y0, t_eval=tau_eval,
            method="LSODA", rtol=1e-6, atol=1e-8,
        )
        return sol if sol.success else None
    except Exception:
        return None


def extract_outputs(sol, params):
    # average over tau in [100, 150]; avoids transient and RA phase noise at final point
    # 100 is a guess. We can change later
    if sol is None:
        # if overshoots/undershoots, we just fill with nan
        return np.full(len(OUTPUT_NAMES), np.nan)
    mask = sol.t >= 100.0
    # mask = sol.t is a mask or an array of indices that have a time >= 100
    means = sol.y[:, mask].mean(axis=1)
    # we only need to average across the first axis
    b, apc, h5, h13, m, r, c = means
    S = (
        sol.y[0, mask] * (1.0 + params["alpha13"] * sol.y[3, mask])
        / (1.0 + sol.y[1, mask] + params["alpha5"] * sol.y[2, mask])
    ).mean()
    # take the average output once the solution settled and is not dependent on how it started (y_0)
    return np.array([b, apc, h5, h13, m, r, c, S])


def main():
    N = 20  # number of steps for morris
    # morris calls it trajectories

    X = sample(problem, N, num_levels=4, seed=42)
    n_rows = X.shape[0]
    print(f"running {n_rows} solves ({N} trajectories x {len(VARY)+1} points each)...")

    Y = np.full((n_rows, len(OUTPUT_NAMES)), np.nan)
    for i in range(n_rows):
        if i % 100 == 0:
            print(f"  {i}/{n_rows}")
        p = BASELINE.copy()
        for k, key in enumerate(VARY):
            p[key] = X[i, k]
        Y[i] = extract_outputs(run_model(p), p)

    # count the nan. not doing anything now, but maybe later we can remove?
    n_failed = np.isnan(Y).any(axis=1).sum()
    if n_failed:
        print(f"  warning: {n_failed} runs returned NaN, filling with column mean")

    # BELOW IS JUST PLOTTING AND SHIT

    
    for o, name in enumerate(OUTPUT_NAMES):
        y_col = Y[:, o].copy()
        # nan impute before passing to SALib
        nan_mask = np.isnan(y_col)
        if nan_mask.any():
            y_col[nan_mask] = np.nanmean(y_col)

        Si = analyze(problem, X, y_col, num_levels=4, seed=42, print_to_console=False)

        mu_star = Si["mu_star"]
        sigma   = Si["sigma"]
        order   = np.argsort(-mu_star)
        # sort by the descending order of ARGUMENTS


        # the logs. 
        # i trunced the rank, parameter and mu_star
        # as per docs, if  n it's a super bad sign; idk the reason behind. refer to docs.
        
        

        
        print(f"\n── {name} ──")
        print(f"  {'Rank':<5} {'Parameter':<15} {'mu_star':>10} {'sigma':>10}")
        print(f"  {'-'*45}")
        for rank, idx in enumerate(order, 1):
            flag = " *" if sigma[idx] > mu_star[idx] / 2 else ""
            print(f"  {rank:<5} {VARY[idx]:<15} {mu_star[idx]:10.4f} {sigma[idx]:10.4f}{flag}")
        print("  (* nonlinear or interaction: sigma > mu*/2)")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.scatter(mu_star, sigma, s=55, color="steelblue", zorder=3)
        for idx in order[:10]:
            ax.annotate(VARY[idx], (mu_star[idx], sigma[idx]),
                        textcoords="offset points", xytext=(4, 3), fontsize=7.5)
        lim = max(np.nanmax(mu_star), np.nanmax(sigma)) * 1.15
        ax.plot([0, lim], [0, lim / 2], "k--", lw=0.8, label=r"$\sigma = \mu^*/2$")
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$\mu^*$"); ax.set_ylabel(r"$\sigma$")
        ax.set_title(f"{name} — Morris screening")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        top15 = order[:15][::-1]
        colors = ["#d62728" if sigma[i] > mu_star[i] / 2 else "steelblue" for i in top15]
        ax2.barh([VARY[i] for i in top15], mu_star[top15], color=colors, edgecolor="white")
        ax2.set_xlabel(r"$\mu^*$")
        ax2.set_title(f"{name} — top 15")
        ax2.grid(True, axis="x", alpha=0.3)

        from matplotlib.patches import Patch
        ax2.legend(handles=[
            Patch(facecolor="steelblue", label="linear"),
        ], fontsize=8, loc="lower right")

        fig.tight_layout()
        fname = f"morris_{name.lower().replace('-', '').replace(' ', '_')}.png"
        fig.savefig(fname, dpi=150)
        print(f"  saved {fname}")
        plt.show()
        plt.close(fig)

    print("\ndone.")


if __name__ == "__main__":
    main()


# %% [cell 2]


# %% [cell 3]
