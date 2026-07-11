import numpy as np


def _hill(x, K, n=1):
    x = max(x, 0.0)
    return x**n / (K**n + x**n)


def _ra_input(tau, p):
    dietary   = p["AR"] * (1.0 + np.cos(2*np.pi * tau / p["TR"] - p["phi"]))
    treatment = 0.5 * p["DR"] * (
        np.tanh(p["q"]*(tau - p["tau1"])) - np.tanh(p["q"]*(tau - p["tau2"])))
    return p["mu0"] + dietary + treatment


def _ode_rhs(tau, y, p):
    b, apc, h5, h13, m, r, c = np.maximum(y, 0.0)
    dP  = 1.0 + p["deltaP1"] * (1.0 - p["thetaP"])
    muR = _ra_input(tau, p)

    db   = (p["W"] + p["eta13"]*_hill(h13, p["kappa13"], p["nH"])
            - b - p["lambdaP"]*apc*b
            - p["lambda5"]*h5*b / (p["kappa5"] + b))
    dapc = (1/p["epsP"]) * (
            (1 + p["rho5"]*h5) / (1 + p["rhoB"]*b + p["rho13"]*h13) - dP*apc)
    dh5  = (1/p["eps5"]) * (
            p["a5"] + p["etaR"]*_hill(r, p["kappaR"], 1)
            - h5 - p["etaM"]*m*h5 / (p["kappaM"] + m))
    dh13 = (1/p["eps13"]) * (
            p["a13"] + p["etaB13"]*_hill(b, p["kappaB13"], p["nB"])
            + p["etaM13"]*_hill(m, p["kappaM13"], p["nM"]) - h13)
    dm   = (1/p["epsM"]) * (
            p["aM"] + p["etaBM"]*_hill(b, p["kappaBM"], p["nB"]) - m)
    dr   = (1/p["epsR"]) * (muR - r - p["lambdaC"]*c*r)
    dc   = (1/p["epsC"]) * (
            p["aC"] + p["etaRC"]*_hill(r, p["kappaRC"], 1)
            + p["etaBC"]*_hill(b, p["kappaBC"], p["nB"]) - c)
    return [db, dapc, dh5, dh13, dm, dr, dc]
