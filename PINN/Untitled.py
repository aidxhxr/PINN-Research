# Converted from Untitled.ipynb (repo path: PINN/Untitled.ipynb) on 2026-09-23.
# Cells appear in notebook order; code is verbatim. Lines that were IPython-only
# ("!shell", "%magic") are kept but commented with "[ipython-only, skipped]".
# A block marked "[conversion shim - not notebook code]" restores kernel state the
# notebook relied on but never stored; no notebook cell is modified.
# Run from this directory:  cd PINN && python3 Untitled.py

# %% [cell 1]
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# Hill function
# ============================================================
def hill(x, K, n=1):
    x = max(x, 0.0)
    return x**n / (K**n + x**n)


# ============================================================
# Periodic dietary RA input + ATRA treatment
# ============================================================
def ra_input(tau, p):
    dietary = p["AR"] * (
        1.0 + np.cos((2.0 * np.pi * tau / p["TR"]) - p["phi"])
    )

    treatment = 0.5 * p["DR"] * (
        np.tanh(p["q"] * (tau - p["tau1"]))
        - np.tanh(p["q"] * (tau - p["tau2"]))
    )

    return p["mu0"] + dietary + treatment


# ============================================================
# Final nondimensional 7-variable model
# y = [b, p, h5, h13, m, r, c]
# ============================================================
def model(tau, y, p):
    b, apc, h5, h13, m, r, c = np.maximum(y, 0.0)

    W = p["W"]
    thetaP = p["thetaP"]

    deltaP = 1.0 + p["deltaP1"] * (1.0 - thetaP)
    muR = ra_input(tau, p)

    db = (
        W
        + p["eta13"] * hill(h13, p["kappa13"], p["nH"])
        - b
        - p["lambdaP"] * apc * b
        - p["lambda5"] * h5 * b / (p["kappa5"] + b)
    )

    dapc = (1.0 / p["epsP"]) * (
        (1.0 + p["rho5"] * h5)
        / (1.0 + p["rhoB"] * b + p["rho13"] * h13)
        - deltaP * apc
    )

    dh5 = (1.0 / p["eps5"]) * (
        p["a5"]
        + p["etaR"] * hill(r, p["kappaR"], 1)
        - h5
        - p["etaM"] * m * h5 / (p["kappaM"] + m)
    )

    dh13 = (1.0 / p["eps13"]) * (
        p["a13"]
        + p["etaB13"] * hill(b, p["kappaB13"], p["nB"])
        + p["etaM13"] * hill(m, p["kappaM13"], p["nM"])
        - h13
    )

    dm = (1.0 / p["epsM"]) * (
        p["aM"]
        + p["etaBM"] * hill(b, p["kappaBM"], p["nB"])
        - m
    )

    dr = (1.0 / p["epsR"]) * (
        muR
        - r
        - p["lambdaC"] * c * r
    )

    dc = (1.0 / p["epsC"]) * (
        p["aC"]
        + p["etaRC"] * hill(r, p["kappaRC"], 1)
        + p["etaBC"] * hill(b, p["kappaBC"], p["nB"])
        - c
    )

    return [db, dapc, dh5, dh13, dm, dr, dc]


# ============================================================
# Stemness index
# ============================================================
def stemness(sol, p):
    b = sol.y[0]
    apc = sol.y[1]
    h5 = sol.y[2]
    h13 = sol.y[3]

    return b * (1.0 + p["alpha13"] * h13) / (
        1.0 + apc + p["alpha5"] * h5
    )


# ============================================================
# Baseline parameters
# ============================================================
params = {
    # Biological regime
    "W": 0.80,
    "thetaP": 1.00,

    # Hill coefficients
    "nB": 2,
    "nM": 2,
    "nH": 2,

    # beta-catenin equation
    "eta13": 0.75,
    "kappa13": 0.55,
    "lambdaP": 1.60,
    "lambda5": 1.30,
    "kappa5": 0.50,

    # APC equation
    "epsP": 1.00,
    "rho5": 1.10,
    "rhoB": 1.10,
    "rho13": 1.30,
    "deltaP1": 3.50,

    # HOXA5 equation
    "eps5": 1.20,
    "a5": 0.15,
    "etaR": 2.50,
    "kappaR": 0.40,
    "etaM": 2.50,
    "kappaM": 0.50,

    # HOXA13 equation
    "eps13": 1.00,
    "a13": 0.18,
    "etaB13": 0.95,
    "kappaB13": 0.50,
    "etaM13": 0.55,
    "kappaM13": 0.50,

    # MYC equation
    "epsM": 0.60,
    "aM": 0.18,
    "etaBM": 1.35,
    "kappaBM": 0.50,

    # RA equation
    "epsR": 0.40,
    "lambdaC": 0.85,

    # CYP26A1 equation
    "epsC": 0.80,
    "aC": 0.08,
    "etaRC": 1.50,
    "kappaRC": 0.50,
    "etaBC": 1.50,
    "kappaBC": 0.50,

    # RA input: background + dietary periodic input + treatment
    "mu0": 0.35,
    "AR": 0.04,
    # "AR": 0.08,
    "TR": 24.0,
    "phi": 0.0,

    # ATRA treatment window
    "DR": 1.50,
    "q": 0.30,
    "tau1": 40.0,
    "tau2": 80.0,

    # Stemness
    "alpha13": 1.00,
    "alpha5": 1.00,
}


# ============================================================
# Biological regimes
# ============================================================
regimes = {
    "Normal": {
        "W": 0.80,
        "thetaP": 1.00,
    },
    "Early adenoma": {
        "W": 1.00,
        "thetaP": 0.75,
    },
    "Cancer-like": {
        "W": 1.50,
        "thetaP": 0.50,
    },
    "Strong APC-mutant": {
        "W": 2.00,
        "thetaP": 0.25,
    },
}


# ============================================================
# Initial conditions
# ============================================================
y0 = [
    0.20,  # b: beta-catenin
    1.00,  # p: APC
    0.80,  # h5: HOXA5
    0.30,  # h13: HOXA13
    0.30,  # m: MYC
    0.60,  # r: RA
    0.40,  # c: CYP26A1
]


# ============================================================
# Time domain
# ============================================================
tau_span = (0.0, 150.0)
tau_eval = np.linspace(tau_span[0], tau_span[1], 3000)


# ============================================================
# Solve all regimes
# ============================================================
solutions = {}

for name, setting in regimes.items():
    pcopy = params.copy()
    pcopy.update(setting)

    sol = solve_ivp(
        fun=lambda tau, y: model(tau, y, pcopy),
        t_span=tau_span,
        y0=y0,
        t_eval=tau_eval,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Simulation failed for {name}: {sol.message}")

    solutions[name] = (sol, pcopy)


# ============================================================
# Plot model variables
# ============================================================
variables = [
    r"$\beta$-catenin",
    "APC",
    "HOXA5",
    "HOXA13",
    "MYC",
    "RA",
    "CYP26A1",
]

for i, var in enumerate(variables):
    plt.figure(figsize=(7.2, 4.2))

    for name, (sol, pcopy) in solutions.items():
        plt.plot(sol.t, sol.y[i], linewidth=2.0, label=name)

    plt.axvspan(
        params["tau1"],
        params["tau2"],
        alpha=0.15,
        label="ATRA treatment",
    )

    plt.xlabel(r"Nondimensional time $\tau$")
    plt.ylabel(var)
    plt.title(var)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()


# ============================================================
# Plot RA input
# ============================================================
plt.figure(figsize=(7.2, 4.2))
mu_vals = np.array([ra_input(t, params) for t in tau_eval])

plt.plot(tau_eval, mu_vals, linewidth=2.0)
plt.axvspan(params["tau1"], params["tau2"], alpha=0.15, label="ATRA treatment")
plt.xlabel(r"Nondimensional time $\tau$")
plt.ylabel(r"$\mu_R(\tau)$")
plt.title("Periodic dietary RA input plus ATRA treatment")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Plot stemness
# ============================================================
plt.figure(figsize=(7.2, 4.2))

for name, (sol, pcopy) in solutions.items():
    S = stemness(sol, pcopy)
    plt.plot(sol.t, S, linewidth=2.0, label=name)

plt.axvspan(params["tau1"], params["tau2"], alpha=0.15, label="ATRA treatment")
plt.xlabel(r"Nondimensional time $\tau$")
plt.ylabel("Stemness index")
plt.title("Stemness index across biological regimes")
plt.legend(frameon=True)
plt.tight_layout()
plt.show()


# ============================================================
# Print final and treatment-window summary
# ============================================================
print("\nSummary values")
print("-" * 90)

for name, (sol, pcopy) in solutions.items():
    S = stemness(sol, pcopy)

    final = sol.y[:, -1]
    final_S = S[-1]

    treatment_mask = (sol.t >= params["tau1"]) & (sol.t <= params["tau2"])
    min_S_treatment = np.min(S[treatment_mask])
    mean_S_treatment = np.mean(S[treatment_mask])

    print(f"\n{name}")
    print(f"Final beta-catenin:      {final[0]:.4f}")
    print(f"Final APC:              {final[1]:.4f}")
    print(f"Final HOXA5:            {final[2]:.4f}")
    print(f"Final HOXA13:           {final[3]:.4f}")
    print(f"Final MYC:              {final[4]:.4f}")
    print(f"Final RA:               {final[5]:.4f}")
    print(f"Final CYP26A1:          {final[6]:.4f}")
    print(f"Final stemness:         {final_S:.4f}")
    print(f"Minimum S during ATRA:  {min_S_treatment:.4f}")
    print(f"Mean S during ATRA:     {mean_S_treatment:.4f}")

# %% [cell 2]
# [ipython-only, skipped] !pip3 install SALib

# %% [cell 3]
# [ipython-only, skipped] !pip3 install numpy pandas matplotlib

# %% [cell 4]
from SALib.test_functions import Ishigami
from SALib.sample.morris import sample
from SALib.analyze import morris

problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359]]
}


# %% [cell 5]
problem['num_vars']

# %% [conversion shim - not notebook code]
# Cells 6-7 read `param_values` (execution counts 28, 29) before cell 8 defines
# it (execution count 30): the kernel held an earlier definition that was later
# overwritten. The same call as cell 8 is made here so the file runs
# top-to-bottom. No notebook cell is modified.
param_values = sample(problem, 1024)

# %% [cell 6]
type(param_values)

# %% [cell 7]
param_values.shape

# %% [cell 8]
param_values = sample(problem, 1024)
Y = Ishigami.evaluate(param_values)

Si = morris.analyze(problem, param_values, Y)

# %% [cell 9]
Si

# %% [cell 10]
#def ra_input(tau, p):
 #   dietary = p["AR"] * (
#        1.0 + np.cos((2.0 * np.pi * tau / p["TR"]) - p["phi"])
 #   )
#
 #   treatment = 0.5 * p["DR"] * (
  #      np.tanh(p["q"] * (tau - p["tau1"]))
   #     - np.tanh(p["q"] * (tau - p["tau2"]))
    #)

    #return p["mu0"] + dietary + treatment
def morris_ra():
    problem = {
        'num_vars': 9,
        'names': ['tau', 'AR', 'TR', 'phi', 'DR', 'q', 'tau1', 'tau2', 'mu0'],
        'bounds': [[0, 150],
                   [0.036, 0.044], # guessed standard deviation; mean is 0.04; assuming 10% stdev
                   [22.6, 26.4], # guessed standard deviation as well
                   [0, 2*3.141592], # pi value for phi
                   [1.35, 1.65], # 
                   [0.27, 0.33],
                   [36, 44],
                   [72, 88],
                   [0.315, 0.385],
                  ],          
        }
    X = sample(problem, 1024)
    Y = Ishigami.evaluate(X)

    Si = morris.analyze(problem, X, Y)
    print(Si)

# %% [cell 11]
#def ra_input(tau, p):
 #   dietary = p["AR"] * (
#        1.0 + np.cos((2.0 * np.pi * tau / p["TR"]) - p["phi"])
 #   )
#
 #   treatment = 0.5 * p["DR"] * (
  #      np.tanh(p["q"] * (tau - p["tau1"]))
   #     - np.tanh(p["q"] * (tau - p["tau2"]))
    #)

    #return p["mu0"] + dietary + treatment
def morris_ra():
    problem = {
        'num_vars': 9,
        'names': ['tau', 'AR', 'TR', 'phi', 'DR', 'q', 'tau1', 'tau2', 'mu0'],
        'bounds': [[0, 150],
                   [0.036, 0.044], # guessed standard deviation; mean is 0.04; assuming 10% stdev
                   [22.6, 26.4], # guessed standard deviation as well
                   [0, 2*3.141592], # pi value for phi
                   [1.35, 1.65], # 
                   [0.27, 0.33],
                   [36, 44],
                   [72, 88],
                   [0.315, 0.385],
                  ],          
        }
    X = sample(problem, 1024)
    Y = Ishigami.evaluate(X)

    Si = morris.analyze(problem, X, Y)
    print(Si)


# %% [cell 12]
morris_ra()

# %% [cell 13]
