import numpy as np
import torch

torch.set_num_threads(48)
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = dict(
    W=0.80, thetaP=1.00,
    nB=2, nM=2, nH=2,
    eta13=0.75, kappa13=0.55, lambdaP=1.60, lambda5=1.30, kappa5=0.50,
    epsP=1.00, rho5=1.10, rhoB=1.10, rho13=1.30, deltaP1=3.50,
    eps5=1.20, a5=0.15, etaR=2.50, kappaR=0.40, etaM=2.50, kappaM=0.50,
    eps13=1.00, a13=0.18, etaB13=0.95, kappaB13=0.50,
    etaM13=0.55, kappaM13=0.50,
    epsM=0.60, aM=0.18, etaBM=1.35, kappaBM=0.50,
    epsR=0.40, lambdaC=0.85,
    epsC=0.80, aC=0.08, etaRC=1.50, kappaRC=0.50,
    etaBC=1.50, kappaBC=0.50,
    mu0=0.35, AR=0.04, TR=24.0, phi=0.0,
    DR=1.50, q=0.30, tau1=40.0, tau2=80.0,
    alpha13=1.00, alpha5=1.00,
)

REGIMES = {
    "Normal":            dict(W=0.80, thetaP=1.00),
    "Early adenoma":     dict(W=1.00, thetaP=0.75),
    "Cancer-like":       dict(W=1.50, thetaP=0.50),
    "Strong APC-mutant": dict(W=2.00, thetaP=0.25),
}

Y0 = np.array([0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40])
VAR_NAMES  = ["b", "apc", "h5", "h13", "m", "r", "c"]
VAR_LABELS = [r"$\beta$-catenin", "APC", "HOXA5", "HOXA13",
              "MYC", "RA", "CYP26A1"]

# ----------------------------------------------------------------------
# INVERSE PROBLEM SETUP — recover ALL identifiable biological parameters
# ----------------------------------------------------------------------
# Forward PINN: parameters known -> solve for the trajectory.
# Inverse PINN: trajectory (sparse, noisy observations) known -> recover
# the unknown ODE parameters jointly with the state network.
#
# This is the deliberately over-parameterised "recover everything"
# baseline: every CONTINUOUS parameter that appears in the physics
# residual is treated as unknown. A downstream sensitivity analysis uses
# the per-parameter recovery error to prune this set.
#
# Held FIXED (not biological unknowns here):
#   nB, nM, nH                  integer Hill exponents (structural)
#   mu0, AR, TR, phi, DR, q,    RA-forcing PROTOCOL (the experimentally
#   tau1, tau2                  set circadian + ATRA schedule)
#   alpha13, alpha5            do not appear in the residual at all
#                              (zero gradient -> meaningless to recover)
FIXED = {
    "nB", "nM", "nH",
    "mu0", "AR", "TR", "phi", "DR", "q", "tau1", "tau2",
    "alpha13", "alpha5",
}

# Derived: everything else, kept in BASELINE order (W, thetaP lead). -> 36
UNKNOWN = [k for k in BASELINE if k not in FIXED]

# Valid ranges for the constrained re-parameterisation:
#   thetaP in (0,1)  -> sigmoid;   every other unknown > 0 -> softplus.
PARAM_RANGE = {k: ((0.0, 1.0) if k == "thetaP" else (0.0, None))
               for k in UNKNOWN}

# Deliberately wrong starting guess: a uniform +50% offset (1.5x baseline)
# for every unknown, so the convergence plots show a clear init->truth gap.
# W, thetaP keep their hand-picked wrong values.
INIT_GUESS = {k: 1.5 * BASELINE[k] for k in UNKNOWN}
INIT_GUESS["W"] = 1.20
INIT_GUESS["thetaP"] = 0.60
