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
# INVERSE PROBLEM SETUP
# ----------------------------------------------------------------------
# Forward PINN: parameters known -> solve for the trajectory.
# Inverse PINN: trajectory (sparse, noisy observations) known -> recover
# the unknown ODE parameters. We recover the two regime-defining,
# biologically meaningful parameters:
#     W       WNT drive on beta-catenin               (true: 0.80 .. 2.00)
#     thetaP  APC functionality (1=healthy, 0=lost)   (true: 1.00 .. 0.25)
# Their per-regime TRUE values live in REGIMES above. The network sees
# only the trajectory data + the ODE structure, never these values.
UNKNOWN = ["W", "thetaP"]

# Deliberately wrong starting guess (same for every regime) so the
# convergence plots show the estimates travelling to the true values.
INIT_GUESS = dict(W=1.20, thetaP=0.60)

# Valid ranges for the constrained re-parameterisation (softplus / sigmoid).
PARAM_RANGE = dict(W=(0.0, None), thetaP=(0.0, 1.0))
