import numpy as np
import torch

torch.set_num_threads(48)
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared 7-ODE WNT-RA-HOX baseline (identical to the forward-hybrid folder;
# this module is the BAYESIAN version of that honest sparse-data forward solve).
# For the FORWARD problem the ODE parameters are KNOWN — only the RA-forcing
# control protocol is used (single condition per regime), and the unknown we put
# a posterior on is the NEURAL NETWORK, not the parameters.
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
    DR=1.50, q=0.30, tau1=40.0, tau2=88.0,
    alpha13=1.00, alpha5=1.00,
)

REGIMES = {
    "Normal":            dict(W=0.80, thetaP=1.00),
    "Early Adenoma":     dict(W=1.00, thetaP=0.75),
    "Advanced Adenoma":       dict(W=1.50, thetaP=0.50),
    "Severe APC Loss": dict(W=2.00, thetaP=0.25),
}

Y0 = np.array([0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40])
VAR_NAMES  = ["b", "p", r"$h_5$", r"$h_{13}$", "m", "r", "c"]
VAR_LABELS = [r"$\beta$-catenin", "APC", "HOXA5", "HOXA13",
              "MYC", "RA", "CYP26A1"]
