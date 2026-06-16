"""
SciPy/BDF reference solver for the 27-state Wnt/Retinoid(RAS)/HOX model, ported from MATLAB ode15s.
Covers model='const' (modelsys_2025.m) and model='dyn' (modelsys_2025_k16dynamic.m).
"""

import time as _time
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Driver parameter vector p  (from normasysl_call.m, lines 2-28)
# Index comments mirror the .m exactly (including its typos in comments).
# ---------------------------------------------------------------------------
P_VEC = np.array([
    0.5,                # p1  kp1
    0.1,                # p2  km1
    0.5,                # p3  k2
    (2.13e11) * (1e-9), # p4  kp3
    1.32e1,             # p5  km3
    (6.00e9) * (1e-0),  # p6  kp4
    3.60e1,             # p7  km5 (comment), used as rkm4
    (1.2e11) * (1e-9),  # p8  kp5
    4.00e1,             # p9  km5
    1.01e1,             # p10 kp6
    1.01e1,             # p11 km6
    1.02e1,             # p12 k7
    2.70e1,             # p13 k8
    (3.60e9) * (1e-7),  # p14 kp9
    3.00e1,             # p15 km9
    1.00e-1,            # p16 k10
    3.85e-2,            # p17 k14
    1.73e-3,            # p18 k15
    2.00e-1,            # p19 v16
    2.00e-1,            # p20 v17
    2.00e-2,            # p21 v18
    2.50e-2,            # p22 v19
    3.85e-2,            # p23 k20
    1.73e-1,            # p24 k21
    8.35e-3,            # p25 k22
    1.00e-2,            # p26 k23
    1.00e-1,            # p27 additional parameter (MCsynth)
], dtype=float)


def build_params(model, gamma1=1.0):
    """Return parameter dict for model in {'const', 'dyn'}."""
    if model not in ("const", "dyn"):
        raise ValueError("model must be 'const' or 'dyn'")

    # ---- driver scalars (normasysl_call.m) ----
    retef = 0.0015
    wntef = 100.0
    CCret = 1000.0
    CCwnt = 1000.0
    funcpercent = 1.0
    k8log = False
    k17log = False

    P = {}
    P["model"] = model
    P["retef"] = retef
    P["wntef"] = wntef

    # ---- time discretization ----
    if model == "const":
        P["t_0"], P["t_N"], P["N_t"] = 0.0, 1500.0, 800
        P["W_amp"] = 1.0
    else:
        P["t_0"], P["t_N"], P["N_t"] = 0.0, 30000.0, 5000
        P["W_amp"] = 3.0

    # ---- Wnt dimensional parameters ----
    DSH0 = 100.0
    TCF0 = 15.0
    APC0 = 100.0      # noqa: F841 (kept for traceability; unused like in .m)
    GSK0 = 50.0

    # K8 / K17 logic (k8log,k17log false -> base values)
    K8 = (1 + (1 - funcpercent)) * 120.0 if k8log else 120.0
    K17 = (1 + (1 - funcpercent)) * 1200.0 if k17log else 1200.0

    K7 = 50.0
    K20 = 1.0
    K21 = 1.0
    Km = 98.0

    k1 = 0.182
    k2 = 1.82e-2
    k3 = 5e-2
    k4 = 0.267
    k5 = 0.133
    k6 = 9.09e-2
    k_6 = 0.909
    k9 = 206.0
    k10 = 206.0
    k11 = 0.417
    v12 = 0.423
    k13 = 2.57e-4
    v14 = 8.22e-5
    k15 = 0.33

    # APC regulating function parameters
    Parameters = [212.8453, 39.9102, 34.1111]
    k19 = 1.0 / K17
    v18 = Parameters[0] * k19
    Kt = Parameters[1]
    Kb = Parameters[2]
    eta = 1.0
    if funcpercent < 1:
        v18 = v18 * (eta * funcpercent)

    # K16 handling differs between models
    if model == "const":
        K16 = 30.0
    else:
        # dynamic K16 sub-parameters
        P["K16_max"] = 200.0 * gamma1
        P["K16_0"] = 100.0
        P["Cs"] = 0.005
        P["nc"] = 2.0
        K16 = None  # not a fixed scalar in dyn

    # ---- non-dimensionalization ----
    w = CCwnt
    P["w"] = w
    gsk0 = GSK0 / w
    tcf0 = TCF0 / w
    dsh0 = DSH0 / w

    K7n = w / K7
    K8n = w / K8
    K17n = w / K17
    K20n = w / K20
    Ktn = (TCF0 * w) / Kt
    Kbn = w / Kb
    if model == "const":
        K16n = K16 * w  # only defined in const model

    k1n = k1 / k5
    k2n = k2 / k5
    k3n = k3 * w / k5
    k4n = k4 / k5
    k6n = (k6 * K21 * w ** 2) / (k5 * K7)
    k_6n = k_6 / k5
    k9n = (k9 * w) / (k5 * K8)
    k10n = k10 / k5
    k11n = k11 / k5
    v12n = v12 / (w * k5)
    k13n = k13 / k5
    v14n = v14 / (k5 * w)
    k15n = k15 * w / k5
    v18n = v18 / (w * k5)
    k19n = k19 / k5

    # ---- RAS parameters from p ----
    p = P_VEC
    P["rkp1"] = p[0]
    P["rkm1"] = p[1]
    P["rk2"] = p[2]
    P["rkp3"] = p[3]
    P["rkm3"] = p[4]
    P["rkp4"] = p[5]
    P["rkm4"] = p[6]
    P["rkp5"] = p[7]
    P["rkm5"] = p[8]
    P["rkp6"] = p[9]
    P["rkm6"] = p[10]
    P["rk7"] = p[11]
    P["rk8"] = p[12]
    P["rkp9"] = p[13]
    P["rkm9"] = p[14]
    P["rk10"] = p[15]
    P["rk11"] = 0.0
    P["rk12"] = 0.0
    P["rk13"] = 0.0
    P["rk14"] = p[16]
    P["rk15"] = p[17]
    P["rv16"] = p[18]
    P["rv17"] = p[19]
    P["rv18"] = p[20]
    P["rv19"] = p[21]
    P["rk20"] = p[22]
    P["rk21"] = p[23]
    P["rk22"] = p[24]
    P["rk23"] = p[25]
    P["MCsynth"] = p[26]

    # ---- HOX parameters ----
    k31 = 8.25 * 16.8182e2
    k32 = 20.85 * 1.25e1
    # k33 below is a function; the scalar k33 in .m is overwritten by the lambda
    k34 = 0.25 * 10.0
    k36 = 0.13
    kp37 = 0.45 * 1.0e-1
    km37 = 0.1 * 1.65
    v31 = 0.15 * 3.333e1
    k38 = 0.083
    v32 = 0.20 * 3.333e1
    k39 = 0.15

    a2 = 7.735
    a3 = 50.5
    a4 = 1.5      # noqa: F841 (unused, kept for traceability)
    Kc = 1.0
    a1 = 55.5     # noqa: F841 (unused)
    Kd = 0.01
    k33_max = 1.0
    n = 1.0
    # New constants
    k = 1500.0    # noqa: F841 (used only via knn which is unused)
    a5 = 1.0      # noqa: F841
    a6 = 1.0      # noqa: F841
    n1 = 1.0      # noqa: F841
    n2 = 1.0      # noqa: F841
    n3 = 1.0
    phi = 0.00139

    # k33(Nr) = k33_max*(w*Nr)^n / (Kd^n + (w*Nr)^n)
    # nondim HOX params (scaled by Bcat / k5)
    k31n = k31 / (w * k5)
    k32n = k32 / k5
    # k33n(Nr) = k33(Nr)/k5
    k36n = k36 / k5
    kp37n = kp37 * w / k5
    km37n = km37 / k5
    v31n = v31 / (w * k5)
    k38n = k38 / k5
    v32n = v32 / (w * k5)
    k39n = k39 / k5

    # Linked parameters
    kp40 = 2.5e-6
    km40 = 1.0e-3
    k41 = 0.1475e-4
    kp40n = (kp40 * w) / k5
    km40n = km40 / k5
    k41n = (k41 * w) / k5

    # ---- both pathways ----
    r = CCret
    a = 1.0 / k5

    # gamma: const uses 0.025*gamma1 ; dyn uses 0.025 (gamma1 enters via K16_max)
    if model == "const":
        gamma = 0.025 * gamma1
    else:
        gamma = 0.025

    Pmin = 0.002

    # stash everything needed by the RHS
    P.update(dict(
        TCF0=TCF0, GSK0=GSK0, K7=K7, K8=K8, K17=K17, K20=K20, K21=K21, Km=Km,
        k5=k5, gsk0=gsk0, tcf0=tcf0, dsh0=dsh0,
        K7n=K7n, K8n=K8n, K17n=K17n, K20n=K20n, Ktn=Ktn, Kbn=Kbn,
        k1n=k1n, k2n=k2n, k3n=k3n, k4n=k4n, k6n=k6n, k_6n=k_6n, k9n=k9n,
        k10n=k10n, k11n=k11n, v12n=v12n, k13n=k13n, v14n=v14n, k15n=k15n,
        v18n=v18n, k19n=k19n,
        k31n=k31n, k32n=k32n, k34=k34, k36n=k36n, kp37n=kp37n, km37n=km37n,
        v31n=v31n, k38n=k38n, v32n=v32n, k39n=k39n,
        k33_max=k33_max, Kd=Kd, n=n, n3=n3, a2=a2, a3=a3, Kc=Kc, phi=phi,
        kp40n=kp40n, km40n=km40n, k41n=k41n,
        r=r, a=a, gamma=gamma, Pmin=Pmin, gamma1=gamma1,
    ))
    if model == "const":
        P["K16"] = K16
        P["K16n"] = K16n
    return P


# ---------------------------------------------------------------------------
# Forcing functions
# ---------------------------------------------------------------------------
def forcing_W(t, P):
    """Wnt signal: amp on [3000,25000], 0 otherwise."""
    on = (t >= 3000.0) and (t <= 25000.0)
    return P["W_amp"] if on else 0.0


def forcing_gtild(t, P):
    """RA periodic source: (a/r)*1e2*(1+cos((pi/6)*a*t)).  (C phase=0)"""
    a = P["a"]
    r = P["r"]
    A = 1e2
    B = np.pi / 6.0
    return (a / r) * A * (1.0 + np.cos(B * a * t))


def forcing_treat(t, P):
    """Treatment: dose_strength=0 -> identically 0 in both models."""
    return 0.0


# ---------------------------------------------------------------------------
# Linking helper functions
# ---------------------------------------------------------------------------
def _K16_dyn(Ci, P):
    """Dynamic K16 = K16_0 + K16_max*(w*Ci)^nc / (Cs^nc + (w*Ci)^nc)."""
    w = P["w"]
    nc = P["nc"]
    Cs = P["Cs"]
    wci = (w * Ci) ** nc
    return P["K16_0"] + (P["K16_max"] * wci) / (Cs ** nc + wci)


def bctf_scalar(Ba, Ci, P):
    """Beta-catenin/TCF complex at a single state point."""
    w = P["w"]
    TCF0 = P["TCF0"]
    if P["model"] == "const":
        K16 = P["K16"]
    else:
        K16 = _K16_dyn(Ci, P)
    return TCF0 * Ba / (K16 + w * Ba)


def bcat_tcf(U, P):
    """Vectorized derived BcatTcf column from a [N,27] solution array."""
    Ba = U[:, 19]
    Ci = U[:, 26]
    w = P["w"]
    TCF0 = P["TCF0"]
    if P["model"] == "const":
        K16 = P["K16"]
    else:
        K16 = _K16_dyn(Ci, P)
    return TCF0 * Ba / (K16 + w * Ba)


def _cyp_synth(R, Ba, Ci, P):
    """CYP synthesis rate."""
    r = P["r"]
    bt = bctf_scalar(Ba, Ci, P)
    rR2 = (r * R) ** 2
    return (P["rv18"] + P["wntef"] * bt + P["MCsynth"] * rR2) / (1.0 + rR2)


def _APCdeg(Rx, Pv, P):
    """APC degradation rate, piecewise on P vs 0.75*Pmin."""
    k19n = P["k19n"]
    retef = P["retef"]
    r = P["r"]
    thresh = 0.75 * P["Pmin"]
    rR2 = r * Rx ** 2
    if Pv >= thresh:
        return k19n / (1.0 + retef * rR2)
    else:
        return k19n * (1.0 + retef * rR2)


def _k33(Nr, P):
    """k33(Nr) = k33_max*(w*Nr)^n / (Kd^n + (w*Nr)^n)."""
    w = P["w"]
    n = P["n"]
    wNr = (w * Nr) ** n
    return (P["k33_max"] * wNr) / (P["Kd"] ** n + wNr)


def _k35n(Ca, P):
    """k35n(Ca) = (k34 + a3*(w*Ca)^n3) / (1 + (phi*w*Ca)^n3) / k5."""
    w = P["w"]
    n3 = P["n3"]
    return (P["k34"] + P["a3"] * (w * Ca) ** n3) / \
           (1.0 + (P["phi"] * w * Ca) ** n3) / P["k5"]


def _v30n(bt, P):
    """v30n(bctf) = (Kc + a2*w*bctf) / (1 + w*bctf) / (w*k5)."""
    w = P["w"]
    return (P["Kc"] + P["a2"] * w * bt) / (1.0 + w * bt) / (w * P["k5"])


# ---------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------
def rhs(t, U, P):
    """Full 27-state derivative vector. U is a length-27 array."""
    w = P["w"]
    a = P["a"]
    r = P["r"]

    # unpack RAS (py 0-12)
    Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn, Bc = U[0:13]
    # unpack WNT (py 13-20)
    V, Di, Db, Bp, Da, Pv, Ba, X = U[13:21]
    # unpack HOX (py 21-26)
    H5, M, Mi, Ca, H13, Ci = U[21:27]

    K7n = P["K7n"]; K8n = P["K8n"]; K17n = P["K17n"]; K20n = P["K20n"]
    K8 = P["K8"]; K21 = P["K21"]; Km = P["Km"]
    gsk0 = P["gsk0"]; tcf0 = P["tcf0"]; dsh0 = P["dsh0"]
    k1n = P["k1n"]; k2n = P["k2n"]; k3n = P["k3n"]; k4n = P["k4n"]
    k6n = P["k6n"]; k_6n = P["k_6n"]; k9n = P["k9n"]; k10n = P["k10n"]
    k11n = P["k11n"]; v12n = P["v12n"]; k13n = P["k13n"]; v14n = P["v14n"]
    k15n = P["k15n"]; v18n = P["v18n"]
    Ktn = P["Ktn"]; Kbn = P["Kbn"]; gamma = P["gamma"]
    kp40n = P["kp40n"]; km40n = P["km40n"]; k41n = P["k41n"]

    Wt = forcing_W(t, P)

    # ---------------- WNT explicit derivatives ----------------
    dV = k1n * (dsh0 - V) * Wt - k2n * V
    dDi = (-(k3n * V + k4n + k_6n) * Di + Da
           + (k6n * Pv * X * (gsk0 - (1 + K8n * Ba) * Da - Di - Db)) / (K21 + w * X))
    dDb = k9n * Da * Ba - k10n * Db
    dBp = k10n * Db - k11n * Bp

    # ---------------- WNT implicit 4x4 block ----------------
    # Matrix entries (An..Nn)
    An = -K8n * (K8 + w * Ba) * X / (K21 + w * X)
    Bn = K7n * X
    Cn = K20n * X - w * K8n * Da * X / (K21 + w * X)
    Dn_ = (1 + K7n * Pv + K20n * Ba
           + K21 * w * (gsk0 - (1 + K8n * Ba) * Da - Di - Db) / ((K21 + w * X) ** 2))
    En = 1 + K8n * Ba
    Fn = K8n * Da
    Gn = K8n * Ba
    Hn = K17n * Ba
    if P["model"] == "const":
        K16 = P["K16"]
        In = 1 + K8n * Da + P["K16n"] * tcf0 / ((K16 + w * Ba) ** 2) + K17n * Pv + K20n * X
    else:
        K16d = _K16_dyn(Ci, P)
        In = 1 + K8n * Da + (K16d * tcf0) / ((K16d + w * Ba) ** 2) + K17n * Pv + K20n * X
    Jn = K20n * Ba
    Kn = 1 + K8n * Ba
    Ln = 1 + K7n * X + K17n * Ba
    Mn = K8n * Da + K17n * Pv
    Nn = K7n * Pv

    Matrixn = np.array([
        [An, Bn, Cn, Dn_],
        [En, 0.0, Fn, 0.0],
        [Gn, Hn, In, Jn],
        [Kn, Ln, Mn, Nn],
    ], dtype=float)

    # Vector entries (RHS1n..RHS4n).  Note RHS1n and RHS4n use dDi+dDb.
    RHS1 = (v14n + (k3n * V + k_6n) * Di
            - k6n * Pv * X * (gsk0 - (1 + K8n * Ba) * Da - Di - Db) / (K21 + w * X)
            - k15n * Pv * X / (Km + w * Pv)
            + w * X / (K21 + w * X) * (dDi + dDb))
    RHS2 = k4n * Di - (1 + k9n * Ba) * Da + k10n * Db
    RHS3 = (v12n - (k13n + k9n * Da + kp40n * H13 + k41n * H5) * Ba
            + km40n * Ci)
    APCdeg_val = _APCdeg(R, Pv, P)
    if P["model"] == "const":
        K16 = P["K16"]
        synth_term = v18n / (1 + Ktn * Ba / (K16 + w * Ba) + Kbn * Ba + gamma * w * H13)
    else:
        K16d = _K16_dyn(Ci, P)
        synth_term = v18n / (1 + Ktn * Ba / (K16d + w * Ba) + Kbn * Ba + gamma * w * H13)
    RHS4 = synth_term - APCdeg_val * Pv - (dDi + dDb)

    Vectorn = np.array([RHS1, RHS2, RHS3, RHS4], dtype=float)

    # Solve Matrixn @ [dDa,dP,dBa,dX] = Vectorn
    sol = np.linalg.solve(Matrixn, Vectorn)
    dDa, dPv, dBa, dX = sol

    # ---------------- RAS derivatives ----------------
    rkp1 = P["rkp1"]; rkm1 = P["rkm1"]; rk2 = P["rk2"]
    rkp3 = P["rkp3"]; rkm3 = P["rkm3"]; rkp4 = P["rkp4"]; rkm4 = P["rkm4"]
    rkp5 = P["rkp5"]; rkm5 = P["rkm5"]; rkp6 = P["rkp6"]; rkm6 = P["rkm6"]
    rk7 = P["rk7"]; rk8 = P["rk8"]; rkp9 = P["rkp9"]; rkm9 = P["rkm9"]
    rk10 = P["rk10"]; rk14 = P["rk14"]; rk15 = P["rk15"]
    rv16 = P["rv16"]; rv17 = P["rv17"]; rv19 = P["rv19"]
    rk20 = P["rk20"]; rk21 = P["rk21"]; rk22 = P["rk22"]; rk23 = P["rk23"]

    gt = forcing_gtild(t, P)
    tr = forcing_treat(t, P)

    dRo = a * (rkm1 * Ra - rkp1 * Ro) + gt
    dRa = a * (rkp1 * Ro - (rkm1 + rk2 * r * A) * Ra)
    dA = (a / r) * rv19 - a * (rk23 * A)
    dR = tr + a * (rk2 * r * Ra * A - rkp3 * r * R * B + (rkm3 + rk14) * Br
                   - rkp4 * r * R * N + (rkm4 + rk15) * Nr - rkp5 * r * R * C
                   + rkm5 * Cr)
    dB = (a / r) * rv16 + a * (-(rk20 + rkp3 * r * R) * B + rkm3 * Br + rk10 * Dn)
    dBr = a * (rkp3 * r * R * B - rkm3 * Br - rkp6 * r * Br * C + rkm6 * Dc
               - rkp9 * r * Br * N + rkm9 * Dn - rk14 * Br)
    dN = (a / r) * rv17 + a * (-(rk21 + rkp4 * r * R) * N + rkm4 * Nr
                               - rkp9 * r * Br * N + rkm9 * Dn)
    dNr = a * (rkp4 * r * R * N - (rkm4 + rk15) * Nr + rk10 * Dn)
    # dCdtn: const uses cyp_synthFunc(R,Ba); dyn uses cyp_synthFunc(R,Ba,Ci)
    dC = (a / r) * _cyp_synth(R, Ba, Ci, P) + a * (
        -(rk22 + rkp5 * r * R) * C + (rkm5 + rk8) * Cr - rkp6 * r * Br * C + rkm6 * Dc)
    dCr = a * (rkp5 * r * R * C - (rkm5 + rk8) * Cr)
    dDc = a * (rkp6 * r * Br * C - (rkm6 + rk7) * Dc)
    dDn = a * (rkp9 * r * Br * N - (rkm9 + rk10) * Dn)
    # dBcdtn: const = a*rk7*Bc ; dyn = a*rk7*Dc  (DIFFERENT)
    if P["model"] == "const":
        dBc = a * rk7 * Bc
    else:
        dBc = a * rk7 * Dc

    # ---------------- HOX derivatives ----------------
    k31n = P["k31n"]; k32n = P["k32n"]
    k36n = P["k36n"]; kp37n = P["kp37n"]; km37n = P["km37n"]
    v31n = P["v31n"]; k38n = P["k38n"]; v32n = P["v32n"]; k39n = P["k39n"]

    dH5 = k31n + k32n * Mi + _k33(Nr, P) * Nr - _k35n(Ca, P) * H5
    dM = _v30n(bctf_scalar(Ba, Ci, P), P) - k36n * M - kp37n * M * Mi + km37n * Ca
    dMi = v31n - k38n * Mi - kp37n * M * Mi + km37n * Ca
    dCa = kp37n * M * Mi - km37n * Ca
    dH13 = v32n - k39n * H13 - kp40n * H13 * Ba + km40n * Ci
    dCi = kp40n * H13 * Ba - km40n * Ci

    return np.array([
        dRo, dRa, dA, dR, dB, dBr, dN, dNr, dC, dCr, dDc, dDn, dBc,   # RAS
        dV, dDi, dDb, dBp, dDa, dPv, dBa, dX,                          # WNT
        dH5, dM, dMi, dCa, dH13, dCi,                                  # HOX
    ], dtype=float)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
def initial_state(P):
    """U0 in order [R_0n(13); W_0n(8); H_0n(6)]."""
    w = P["w"]
    model = P["model"]

    # WNT IC (Katie), scaled by 1/w
    V_0, Di_0, Db_0, Bp_0 = 0.0, 4.83e-3, 2.02e-3, 1.0
    Da_0, P_0, Ba_0, X_0 = 9.66e-3, 18.116, 25.1, 4.93e-4
    W_0n = (1.0 / w) * np.array([V_0, Di_0, Db_0, Bp_0, Da_0, P_0, Ba_0, X_0])

    # RAS IC, scaled by 1/1000
    R_0n = (1.0 / 1000.0) * np.array([10, 10, 10, 100, 1, 0, 1, 0, 0.01,
                                      0, 0, 0, 0], dtype=float)

    # HOX IC differs between models (M_0, Mi_0, Ca_0)
    H5_0 = 0.00146
    H13_0 = 0.04439
    Ci_0 = 0.0012
    if model == "const":
        M_0, Mi_0, Ca_0 = 0.055, 0.058, 0.8984
    else:
        M_0, Mi_0, Ca_0 = 0.04999, 0.0601, 0.84028
    H_0n = np.array([H5_0, M_0, Mi_0, Ca_0, H13_0, Ci_0])

    return np.concatenate([R_0n, W_0n, H_0n])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def solve_reference(model, gamma1=1.0, method="BDF", rtol=1e-8, atol=1e-10):
    """Integrate with a stiff BDF solver and return (t, U [N,27], P)."""
    P = build_params(model, gamma1=gamma1)
    U0 = initial_state(P)
    t_eval = np.linspace(P["t_0"], P["t_N"], P["N_t"])

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, P),
        t_span=(P["t_0"], P["t_N"]),
        y0=U0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"[{model}] integration failed: {sol.message}")
    return sol.t, sol.y.T, P


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = "/Users/amirkhanaidarkhan/Documents/Lessons/raw"

    for model, fname in (("const", "reference_const.npz"),
                         ("dyn", "reference_dyn.npz")):
        print("=" * 70)
        print(f"Solving model = '{model}'")
        t0 = _time.time()
        t, U, P = solve_reference(model)
        wall = _time.time() - t0
        BcatTcf = bcat_tcf(U, P)

        finite = np.all(np.isfinite(U)) and np.all(np.isfinite(BcatTcf))
        print(f"  wall-clock: {wall:.2f} s")
        print(f"  t shape: {t.shape}   U shape: {U.shape}")
        print(f"  all finite (no NaN/Inf): {finite}")
        print(f"  t range: [{t[0]:.1f}, {t[-1]:.1f}]")
        # sample final-state values
        Ba_final = U[-1, 19]
        P_final = U[-1, 18]
        Ci_final = U[-1, 26]
        H13_final = U[-1, 25]   # iH13 = 25 (24 is Ca)
        print(f"  FINAL state samples:")
        print(f"    Ba (py19)   = {Ba_final:.8e}")
        print(f"    P  (py18)   = {P_final:.8e}")
        print(f"    Ci (py26)   = {Ci_final:.8e}")
        print(f"    H13(py25)   = {H13_final:.8e}")
        print(f"    BcatTcf     = {BcatTcf[-1]:.8e}")
        print(f"  full final 27-vector:")
        with np.printoptions(precision=6, suppress=False, linewidth=120):
            print("   ", U[-1])

        np.savez(f"{out_dir}/{fname}", t=t, U=U, BcatTcf=BcatTcf)
        print(f"  saved -> {out_dir}/{fname}")

    print("=" * 70)
    print("DONE")
