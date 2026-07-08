import numpy as np
import torch


def _hill_t(x, K, n=1):
    return x**n / (K**n + x**n)


def _ra_t(t, p):
    dietary   = p["AR"] * (1.0 + torch.cos(2*np.pi * t / p["TR"] - p["phi"]))
    treatment = 0.5 * p["DR"] * (
        torch.tanh(p["q"]*(t - p["tau1"])) - torch.tanh(p["q"]*(t - p["tau2"])))
    return p["mu0"] + dietary + treatment


def physics_residual(t, z, dz, p):
    b, apc, h5, h13, m, r, c = (z[:, i:i+1] for i in range(7))
    dP  = 1.0 + p["deltaP1"] * (1.0 - p["thetaP"])
    muR = _ra_t(t, p)

    f0 = (p["W"] + p["eta13"]*_hill_t(h13, p["kappa13"], p["nH"])
          - b - p["lambdaP"]*apc*b
          - p["lambda5"]*h5*b / (p["kappa5"] + b))
    f1 = (1/p["epsP"]) * (
          (1 + p["rho5"]*h5) / (1 + p["rhoB"]*b + p["rho13"]*h13) - dP*apc)
    f2 = (1/p["eps5"]) * (
          p["a5"] + p["etaR"]*_hill_t(r, p["kappaR"], 1)
          - h5 - p["etaM"]*m*h5 / (p["kappaM"] + m))
    f3 = (1/p["eps13"]) * (
          p["a13"] + p["etaB13"]*_hill_t(b, p["kappaB13"], p["nB"])
          + p["etaM13"]*_hill_t(m, p["kappaM13"], p["nM"]) - h13)
    f4 = (1/p["epsM"]) * (
          p["aM"] + p["etaBM"]*_hill_t(b, p["kappaBM"], p["nB"]) - m)
    f5 = (1/p["epsR"]) * (muR - r - p["lambdaC"]*c*r)
    f6 = (1/p["epsC"]) * (
          p["aC"] + p["etaRC"]*_hill_t(r, p["kappaRC"], 1)
          + p["etaBC"]*_hill_t(b, p["kappaBC"], p["nB"]) - c)

    return dz - torch.cat([f0, f1, f2, f3, f4, f5, f6], dim=1)
