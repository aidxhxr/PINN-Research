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
    DR=1.50, q=0.30, tau1=40.0, tau2=88.0,
    alpha13=1.00, alpha5=1.00,
    # exogenous WNT / MYC experimental input channels (excite variant).
    # DW=DM=0 by default -> identical to the base model; the conditions below
    # switch them on. Experimenter-set (known) protocol knobs, like the ATRA
    # pulse — NOT recovered.
    DW=0.0, qW=0.30, tauW1=40.0, tauW2=80.0,
    DM=0.0, qM=0.30, tauM1=40.0, tauM2=80.0,
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
# MULTIPLE EXPERIMENTAL CONDITIONS  (the key identifiability lever)
# ----------------------------------------------------------------------
# A single trajectory leaves many parameter combinations indistinguishable
# (Engl et al. 2009; Peifer & Timmer 2007): the inverse problem is
# ill-posed and the optimiser settles on a wrong-but-compatible parameter
# set. The cure both papers point to is *more information* — observe the
# same system (same biological parameters) under several distinct
# experimental protocols (`nexp` in Peifer; 16 conditions in Engl's
# benchmark). Conditions that excite otherwise-dormant dynamics (e.g. the
# APC/RA sub-system) restore identifiability.
#
# Here the experimenter-set knob is the RA forcing protocol in `_ra_input`
# (`AR, DR, tau1, tau2, mu0, ...`). Each condition overrides a few of those
# *known* protocol parameters; the BIOLOGICAL unknowns (UNKNOWN below) are
# shared across all conditions and are what we recover.
CONDITIONS = [
    # control: the original circadian + single ATRA pulse over [40, 80]
    {"name": "ctrl",       "forcing": {}},
    # no ATRA treatment: pure circadian drive — exposes the untreated
    # baseline of the RA / APC / CYP26A1 sub-dynamics
    {"name": "noATRA",     "forcing": {"DR": 0.0}},
    # strong early ATRA pulse: a large, earlier dose that hits the system
    # while it is still transient — excites RA, CYP26A1 and (via dP) the
    # thetaP-controlled APC decay that is flat under the control protocol
    {"name": "earlyATRA",  "forcing": {"DR": 2.50, "tau1": 20.0, "tau2": 55.0}},
    # late ATRA pulse: hits the (near-)settled system — probes the relaxation
    # timescales / how fast the RA-CYP26 loop re-equilibrates after settling
    {"name": "lateATRA",   "forcing": {"DR": 2.00, "tau1": 90.0, "tau2": 130.0}},
    # low ATRA dose over the control window: a small perturbation that keeps
    # the RA sub-dynamics in their (more identifiable) near-linear regime
    {"name": "lowATRA",    "forcing": {"DR": 0.60, "tau1": 40.0, "tau2": 80.0}},
    # strong circadian, no ATRA: large-amplitude oscillatory drive that
    # excites the RA / MYC / HOX modes the single ATRA pulse barely touches
    {"name": "strongCirc", "forcing": {"AR": 0.12, "DR": 0.0}},

    # ------------------------------------------------------------------
    # EXCITE variant: perturb the WNT/HOX/MYC subsystem directly. The 6
    # conditions above only vary the RA forcing, which reaches the rest of
    # the network through a single door (etaR*hill(r,kappaR)); the profile
    # likelihood showed the b/h5/h13/m/apc parameters stay weakly identifiable
    # as a result. These add two new experimenter-set input channels (a known
    # exogenous WNT pulse into db, a known MYC pulse into dm) that (a) excite
    # the dormant sub-network and (b) sweep b and m ACROSS the Hill kappa
    # scales, which is what separates the correlated eta/kappa pairs.
    # ------------------------------------------------------------------
    # exogenous WNT pulse: drives b up over [30,70] -> excites lambdaP, lambda5,
    # kappa5, etaB13/kappaB13, etaBM/kappaBM, etaBC/kappaBC, rhoB and sweeps b
    # through the kappaB* scales
    {"name": "wntPulse",  "forcing": {"DW": 1.00, "tauW1": 30.0, "tauW2": 70.0}},
    # WNT inhibition step: lowers b from t=50 on -> de-saturates the high-WNT
    # regimes (Cancer-like / Strong APC-mutant) where b pins near saturation and
    # thetaP/W lose all constraint; tauW2 past the horizon => sustained step
    {"name": "wntInhib",  "forcing": {"DW": -0.70, "tauW1": 50.0, "tauW2": 200.0}},
    # exogenous MYC pulse: drives m up over [30,70] -> directly excites the
    # worst-recovered params etaM/kappaM (h5 eq) and etaM13/kappaM13 (h13 eq),
    # and sweeps m across kappaM / kappaM13
    {"name": "mycPulse",  "forcing": {"DM": 1.20, "tauM1": 30.0, "tauM2": 70.0}},
    # combined WNT+ATRA: a WNT step under the control ATRA pulse probes the
    # b<->RA cross-talk (apc/dP coupling) that neither single perturbation isolates
    {"name": "wntRAcombo","forcing": {"DW": 0.80, "tauW1": 25.0, "tauW2": 200.0,
                                      "DR": 2.00, "tau1": 40.0, "tau2": 80.0}},
]
# NOTE (excite variant): earlier variants found that adding MORE RA-only
# conditions helped only marginally (3/6/8 cond -> 5/5/7 of 36 for the PINN),
# and the profile likelihood (runs/..._ident) showed why — all those conditions
# perturb the RA input, which reaches the WNT/HOX/MYC sub-network through a
# single door (etaR*hill(r,kappaR)), so those parameters stay weakly
# identifiable no matter how many RA protocols you stack. The 4 conditions
# added above change the KIND of information, not just the amount: they drive
# the WNT and MYC nodes directly, exciting the dormant sub-network and sweeping
# b and m across their Hill kappa scales (which is what separates the
# correlated eta/kappa pairs). This is the recover_odefit.py path — the
# accuracy tool; the PINN in main.py remains derivative-limited regardless.

# ----------------------------------------------------------------------
# INVERSE PROBLEM SETUP — recover ALL identifiable biological parameters
# ----------------------------------------------------------------------
# Held FIXED (not biological unknowns here):
#   nB, nM, nH                  integer Hill exponents (structural)
#   mu0, AR, TR, phi, DR, q,    RA-forcing PROTOCOL (experimentally set;
#   tau1, tau2                  varied per condition above)
#   alpha13, alpha5            do not appear in the residual at all
FIXED = {
    "nB", "nM", "nH",
    "mu0", "AR", "TR", "phi", "DR", "q", "tau1", "tau2",
    "alpha13", "alpha5",
    # excite variant: the WNT/MYC input-channel protocol knobs are known
    "DW", "qW", "tauW1", "tauW2", "DM", "qM", "tauM1", "tauM2",
}

# Derived: everything else, kept in BASELINE order (W, thetaP lead). -> 36
UNKNOWN = [k for k in BASELINE if k not in FIXED]

# Valid ranges for the constrained re-parameterisation:
#   thetaP in (0,1)  -> sigmoid;   every other unknown > 0 -> log-positive.
PARAM_RANGE = {k: ((0.0, 1.0) if k == "thetaP" else (0.0, None))
               for k in UNKNOWN}

# Nominal scale for the log/relative parametrisation in model.py: each
# positive unknown is recovered as nominal * exp(s) so the optimiser sees
# an order-1, scale-invariant variable for every parameter regardless of
# whether its true value is 0.08 or 3.5.
NOMINAL = {k: BASELINE[k] for k in UNKNOWN}

# Deliberately wrong starting guess: a uniform +50% offset (1.5x baseline)
# for every unknown, so the convergence plots show a clear init->truth gap.
# W, thetaP keep their hand-picked wrong values.
INIT_GUESS = {k: 1.5 * BASELINE[k] for k in UNKNOWN}
INIT_GUESS["W"] = 1.20
INIT_GUESS["thetaP"] = 0.60
