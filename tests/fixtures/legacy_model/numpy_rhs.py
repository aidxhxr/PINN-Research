import numpy as np


def _hill(x, K, n=1):
    x = max(x, 0.0)
    return x**n / (K**n + x**n)


def _ra_input(tau, p):
    dietary   = p["AR"] * (1.0 + np.cos(2*np.pi * tau / p["TR"] - p["phi"]))
    treatment = 0.5 * p["DR"] * (
        np.tanh(p["q"]*(tau - p["tau1"])) - np.tanh(p["q"]*(tau - p["tau2"])))
    return p["mu0"] + dietary + treatment


def _pulse(tau, D, q, t1, t2):
    """Smooth on/off box (same tanh shape as the ATRA pulse). D=0 -> no input,
    so a condition that leaves a channel unset is numerically identical to the
    original model. t2 beyond the horizon gives a sustained STEP (rising edge
    only). These are the extra experimenter-set input channels of the excite
    variant — known forcing, not recovered."""
    if D == 0.0:
        return 0.0
    return 0.5 * D * (np.tanh(q*(tau - t1)) - np.tanh(q*(tau - t2)))


def _wnt_input(tau, p):
    # exogenous WNT perturbation (WNT agonist / GSK3 inhibitor when DW>0, or a
    # tankyrase-driven inhibition when DW<0). Adds to the constant drive W.
    return _pulse(tau, p["DW"], p["qW"], p["tauW1"], p["tauW2"])


def _myc_input(tau, p):
    # exogenous MYC perturbation (e.g. an inducible MYC-ER pulse). Extra
    # production term in the m equation.
    return _pulse(tau, p["DM"], p["qM"], p["tauM1"], p["tauM2"])


def _ode_rhs(tau, y, p):
    b, apc, h5, h13, m, r, c = np.maximum(y, 0.0)
    dP  = 1.0 + p["deltaP1"] * (1.0 - p["thetaP"])
    muR = _ra_input(tau, p)

    db   = (p.get("kW", 1.0)*p["W"] + _wnt_input(tau, p)
            + p.get("k13", 1.0)*p["eta13"]*_hill(h13, p["kappa13"], p["nH"])
            - b - p["lambdaP"]*apc*b
            - p["lambda5"]*h5*b / (p["kappa5"] + b))
    dapc = (1/p["epsP"]) * (
            (1 + p["rho5"]*h5) / (1 + p["rhoB"]*b + p["rho13"]*h13) - dP*apc)
    dh5  = (1/p["eps5"]) * (
            p.get("ka5", 1.0)*p["a5"] + p["etaR"]*_hill(r, p["kappaR"], 1)
            - h5 - p["etaM"]*m*h5 / (p["kappaM"] + m))
    dh13 = (1/p["eps13"]) * (
            p.get("ka13", 1.0)*p["a13"]
            + p["etaB13"]*_hill(b, p["kappaB13"], p["nB"])
            + p["etaM13"]*_hill(m, p["kappaM13"], p["nM"]) - h13)
    dm   = (1/p["epsM"]) * (
            p.get("kaM", 1.0)*p["aM"] + _myc_input(tau, p)
            + p["etaBM"]*_hill(b, p["kappaBM"], p["nB"]) - m)
    dr   = (1/p["epsR"]) * (muR - r - p["lambdaC"]*c*r)
    dc   = (1/p["epsC"]) * (
            p.get("kaC", 1.0)*p["aC"] + p["etaRC"]*_hill(r, p["kappaRC"], 1)
            + p["etaBC"]*_hill(b, p["kappaBC"], p["nB"]) - c)
    return [db, dapc, dh5, dh13, dm, dr, dc]
