import numpy as np
import torch


def _hill_t(x, K, n=1):
    return x**n / (K**n + x**n)


def _ra_t(t, p):
    dietary   = p["AR"] * (1.0 + torch.cos(2*np.pi * t / p["TR"] - p["phi"]))
    treatment = 0.5 * p["DR"] * (
        torch.tanh(p["q"]*(t - p["tau1"])) - torch.tanh(p["q"]*(t - p["tau2"])))
    return p["mu0"] + dietary + treatment


def _pulse_t(t, D, q, t1, t2):
    # torch mirror of odes._pulse; D=0 short-circuits so unset channels are exact
    # zeros (and the constant folds out of autodiff).
    if D == 0.0:
        return 0.0
    return 0.5 * D * (torch.tanh(q*(t - t1)) - torch.tanh(q*(t - t2)))


def physics_rhs(t, z, p, nn_terms=None):
    """The ODE right-hand side f(t, z, theta): the 7-vector [db, dapc, ...].

    Split out from physics_residual so the derivative-free INTEGRAL residual
    (trapezoidal multiple-shooting) can use f directly without an autodiff
    time-derivative of the state net — the biased dz/dt is the PINN's proven
    accuracy ceiling, so the integral formulation avoids it entirely.

    HYBRID/UDE: `nn_terms` maps a term name (config.HYBRID_TERMS) to a small
    MechanisticNN that REPLACES the closed-form Hill expression for that term.
    Empty dict => the pure mechanistic model, bit-for-bit as before. Because the
    integral residual needs no autodiff time-derivative, the learned term drops
    straight in and trains by ordinary backprop.
    """
    nn_terms = nn_terms or {}
    b, apc, h5, h13, m, r, c = (z[:, i:i+1] for i in range(7))
    muR = _ra_t(t, p)
    wnt = _pulse_t(t, p["DW"], p["qW"], p["tauW1"], p["tauW2"])
    myc = _pulse_t(t, p["DM"], p["qM"], p["tauM1"], p["tauM2"])

    # --- every hybridisable relationship: learned net or closed form ---------
    # Three structural classes, and the class is the point of the comparison:
    #   production  f(u) enters an equation ADDITIVELY (can trade a constant
    #               with that equation's basal-production parameter)
    #   modulator   f(u) MULTIPLIES another state (can only trade a scale)
    #   ratio       f(h5, b, h13), multivariate and not a Hill activation
    def _term(name, closed_form, x):
        net = nn_terms.get(name)
        return net(x) if net is not None else closed_form()

    # production / activation edges
    ra_h5  = _term("ra_h5",
                   lambda: p["etaR"]*_hill_t(r, p["kappaR"], 1), r)
    b_h13  = _term("b_h13",
                   lambda: p["etaB13"]*_hill_t(b, p["kappaB13"], p["nB"]), b)
    bm_myc = _term("bm_myc",
                   lambda: p["etaBM"]*_hill_t(b, p["kappaBM"], p["nB"]), b)
    bc_cyp = _term("bc_cyp",
                   lambda: p["etaBC"]*_hill_t(b, p["kappaBC"], p["nB"]), b)
    rc_cyp = _term("rc_cyp",
                   lambda: p["etaRC"]*_hill_t(r, p["kappaRC"], 1), r)
    m_h13  = _term("m_h13",
                   lambda: p["etaM13"]*_hill_t(m, p["kappaM13"], p["nM"]), m)
    h13_b  = _term("h13_b",
                   lambda: p["eta13"]*_hill_t(h13, p["kappa13"], p["nH"]), h13)

    # modulator edges: the learned factor multiplies a second state
    m_h5   = _term("m_h5",  lambda: p["etaM"]*m/(p["kappaM"] + m), m)
    h5_b   = _term("h5_b",  lambda: p["lambda5"]*b/(p["kappa5"] + b), b)
    apc_b  = _term("apc_b", lambda: p["lambdaP"]*apc, apc)
    c_ra   = _term("c_ra",  lambda: p["lambdaC"]*c, c)

    # multivariate APC production ratio
    apc_prod = _term(
        "apc_prod",
        lambda: (1 + p["rho5"]*h5) / (1 + p["rhoB"]*b + p["rho13"]*h13),
        torch.cat([h5, b, h13], dim=1))

    apc_loss = (1.0 - p["thetaP"]) + torch.zeros_like(apc)
    apc_mutation = _term(
        "apc_mutation",
        lambda: p["deltaP1"] * apc_loss,
        apc_loss,
    )
    dP = 1.0 + apc_mutation

    f0 = (p.get("kW", 1.0)*p["W"] + wnt + p.get("k13", 1.0)*h13_b
          - b - apc_b*b
          - h5_b*h5)
    f1 = (1/p["epsP"]) * (apc_prod - dP*apc)
    f2 = (1/p["eps5"]) * (
          p.get("ka5", 1.0)*p["a5"] + ra_h5
          - h5 - m_h5*h5)
    f3 = (1/p["eps13"]) * (
          p.get("ka13", 1.0)*p["a13"] + b_h13
          + m_h13 - h13)
    f4 = (1/p["epsM"]) * (
          p.get("kaM", 1.0)*p["aM"] + myc + bm_myc - m)
    f5 = (1/p["epsR"]) * (muR - r - c_ra*r)
    f6 = (1/p["epsC"]) * (
          p.get("kaC", 1.0)*p["aC"] + rc_cyp
          + bc_cyp - c)

    return torch.cat([f0, f1, f2, f3, f4, f5, f6], dim=1)


def physics_residual(t, z, dz, p, nn_terms=None):
    return dz - physics_rhs(t, z, p, nn_terms)
