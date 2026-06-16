"""
Physics for the 27-state coupled Wnt/Retinoid(RAS)/HOX model, ported from MATLAB.
Covers model='const' (modelsys_2025.m, fixed K16=30) and model='dyn' (modelsys_2025_k16dynamic.m).
"""

import numpy as np
import torch

# global state indices (python, 0-based)
iRo, iRa, iA, iR, iB, iBr, iN, iNr, iC, iCr, iDc, iDn, iBc = range(0, 13)
iV, iDi, iDb, iBp, iDa, iP, iBa, iX = range(13, 21)
iH5, iM, iMi, iCa, iH13, iCi = range(21, 27)


# ----------------------------------------------------------------------------
# parameter vector p  (from normasysl_call.m, lines 2-28)  -- IDENTICAL both models
# ----------------------------------------------------------------------------
def _p_vector():
    return [
        0.5,                  # p1  rkp1
        0.1,                  # p2  rkm1
        0.5,                  # p3  rk2
        (2.13e11) * (1e-9),   # p4  rkp3
        1.32e1,               # p5  rkm3
        (6.00e9) * (1e-0),    # p6  rkp4
        3.60e1,               # p7  rkm4
        (1.2e11) * (1e-9),    # p8  rkp5
        4.00e1,               # p9  rkm5
        1.01e1,               # p10 rkp6
        1.01e1,               # p11 rkm6
        1.02e1,               # p12 rk7
        2.70e1,               # p13 rk8
        (3.60e9) * (1e-7),    # p14 rkp9
        3.00e1,               # p15 rkm9
        1.00e-1,              # p16 rk10
        3.85e-2,              # p17 rk14
        1.73e-3,              # p18 rk15
        2.00e-1,              # p19 rv16
        2.00e-1,              # p20 rv17
        2.00e-2,              # p21 rv18
        2.50e-2,              # p22 rv19
        3.85e-2,              # p23 rk20
        1.73e-1,              # p24 rk21
        8.35e-3,              # p25 rk22
        1.00e-2,              # p26 rk23
        1.00e-1,              # p27 MCsynth
    ]


def build_params(model, gamma1=1.0, funcpercent=1.0, retef=0.0015, wntef=100.0,
                 CCret=1000.0, CCwnt=1000.0, k8log=False, k17log=False):
    """Return parameter dict P for model in {'const', 'dyn'}."""
    assert model in ("const", "dyn"), "model must be 'const' or 'dyn'"
    P = {}
    P['model'] = model

    # ---- time discretization (differs) -------------------------------------
    if model == "const":
        P['T'] = 1500.0          # modelsys_2025.m line 19 (t_N)
        P['N_t'] = 800           # line 17
        P['Wamp'] = 1.0          # line 257: amplitude 1
    else:
        P['T'] = 30000.0         # k16dynamic line 18
        P['N_t'] = 5000          # line 16
        P['Wamp'] = 3.0          # line 261: amplitude 3

    # ---- WNT dimensional parameters (identical lines 27-30 / 23-26) --------
    DSH0 = 100.0
    TCF0 = 15.0
    APC0 = 100.0   # noqa: F841 (only used via commented P_0 alternative)
    GSK0 = 50.0
    P['TCF0'] = TCF0

    # K8, K17 with k8log/k17log switches (lines 35-45 / 31-41)
    if k8log:
        K8 = (1 + (1 - funcpercent)) * 120.0
    else:
        K8 = 120.0
    if k17log:
        K17 = (1 + (1 - funcpercent)) * 1200.0
    else:
        K17 = 1200.0
    P['K8'] = K8

    # dynamic-K16 extra params (k16dynamic lines 43-46). Stored for both models;
    # const ignores them and uses fixed K16=30.
    P['K16'] = 30.0                 # const fixed value (modelsys_2025 line 48)
    P['K16_max'] = 200.0 * gamma1   # k16dynamic line 43
    P['K16_0'] = 100.0              # line 44
    P['Cs'] = 0.005                 # line 45
    P['nc'] = 2.0                   # line 46

    K7 = 50.0
    K20 = 1.0
    K21 = 1.0
    Km = 98.0
    P['K21'] = K21
    P['Km'] = Km

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
    P['k5'] = k5

    # APC regulating function params (lines 69-78)
    Parameters = [212.8453, 39.9102, 34.1111]
    k19 = 1.0 / K17
    v18 = Parameters[0] * k19
    Kt = Parameters[1]
    Kb = Parameters[2]
    eta = 1.0
    if funcpercent < 1:
        v18 = v18 * (eta * funcpercent)

    # ---- non-dimensionalization (lines 81-116) -----------------------------
    w = CCwnt
    P['w'] = w
    P['gsk0'] = GSK0 / w
    P['tcf0'] = TCF0 / w
    P['dsh0'] = DSH0 / w

    P['K7n'] = w / K7
    P['K8n'] = w / K8
    P['K16n'] = P['K16'] * w   # const only (line 88): K16n = K16*w
    P['K17n'] = w / K17
    P['K20n'] = w / K20
    P['Ktn'] = (TCF0 * w) / Kt
    P['Kbn'] = w / Kb

    P['k1n'] = k1 / k5
    P['k2n'] = k2 / k5
    P['k3n'] = k3 * w / k5
    P['k4n'] = k4 / k5
    P['k6n'] = (k6 * K21 * w ** 2) / (k5 * K7)
    P['k_6n'] = k_6 / k5
    P['k9n'] = (k9 * w) / (k5 * K8)
    P['k10n'] = k10 / k5
    P['k11n'] = k11 / k5
    P['v12n'] = v12 / (w * k5)
    P['k13n'] = k13 / k5
    P['v14n'] = v14 / (k5 * w)
    P['k15n'] = k15 * w / k5
    P['v18n'] = v18 / (w * k5)
    P['k19n'] = k19 / k5

    # ---- RAS parameters (lines 118-146) ------------------------------------
    p = _p_vector()
    P['rkp1'] = p[0]
    P['rkm1'] = p[1]
    P['rk2'] = p[2]
    P['rkp3'] = p[3]
    P['rkm3'] = p[4]
    P['rkp4'] = p[5]
    P['rkm4'] = p[6]
    P['rkp5'] = p[7]
    P['rkm5'] = p[8]
    P['rkp6'] = p[9]
    P['rkm6'] = p[10]
    P['rk7'] = p[11]
    P['rk8'] = p[12]
    P['rkp9'] = p[13]
    P['rkm9'] = p[14]
    P['rk10'] = p[15]
    P['rk14'] = p[16]
    P['rk15'] = p[17]
    P['rv16'] = p[18]
    P['rv17'] = p[19]
    P['rv18'] = p[20]
    P['rv19'] = p[21]
    P['rk20'] = p[22]
    P['rk21'] = p[23]
    P['rk22'] = p[24]
    P['rk23'] = p[25]
    P['MCsynth'] = p[26]

    # ---- HOX parameters (lines 150-231) ------------------------------------
    k31 = 8.25 * 16.8182e2
    k32 = 20.85 * 1.25e1
    # k33 = 0.30 * 1.95e2   # overwritten by k33 = @(Nr) ... below; not used
    k34 = 0.25 * 10.0
    k36 = 0.13
    kp37 = 0.45 * 1.0e-1
    km37 = 0.1 * 1.65
    v31 = 0.15 * 3.333e1
    k38 = 0.083
    v32 = 0.20 * 3.333e1
    k39 = 0.15
    P['k34'] = k34

    # HOXA5 deg / MYC synth params (lines 175-185)
    a2 = 7.735
    a3 = 50.5
    a4 = 1.5   # noqa: F841 (only in commented v32n alt)
    Kc = 1.0
    a1 = 55.5  # noqa: F841 (only in commented k33n alt)
    Kd = 0.01
    k33_max = 1.0
    n = 1.0
    P['a2'] = a2
    P['a3'] = a3
    P['Kc'] = Kc
    P['Kd'] = Kd
    P['k33_max'] = k33_max
    P['nexp'] = n          # exponent n used in k33(Nr)

    # new constants (lines 188-198)
    k = 1500.0
    a5 = 1.0     # noqa: F841 (only in commented Hill-form dH5)
    a6 = 1.0     # noqa: F841
    n1 = 1.0     # noqa: F841
    n2 = 1.0     # noqa: F841
    n3 = 1.0
    phi = 0.00139
    P['n3'] = n3
    P['phi'] = phi

    # nondim HOX (lines 203-215)
    P['k31n'] = k31 / (w * k5)
    P['k32n'] = k32 / k5
    P['k36n'] = k36 / k5
    P['kp37n'] = kp37 * w / k5
    P['km37n'] = km37 / k5
    P['v31n'] = v31 / (w * k5)
    P['k38n'] = k38 / k5
    P['v32n'] = v32 / (w * k5)
    P['k39n'] = k39 / k5
    P['knn'] = k / (w * k5)   # noqa (only in commented Hill-form dH5)

    # linked params (lines 219-231)
    kp40 = 2.5e-6
    km40 = 1.0e-3
    k41 = 0.1475 * (1e-4)
    P['kp40n'] = (kp40 * w) / k5
    P['km40n'] = km40 / k5
    P['k41n'] = (k41 * w) / k5

    # ---- both-pathway nondim (lines 236-237) -------------------------------
    P['r'] = CCret
    P['a'] = 1.0 / k5

    # ---- linking params (lines 246-249 / 250-253) --------------------------
    P['retef'] = retef
    P['wntef'] = wntef
    P['Pmin'] = 0.002
    # gamma: const = 0.025*gamma1 (line 249); dyn = 0.025 fixed (line 253)
    if model == "const":
        P['gamma'] = 0.025 * gamma1
    else:
        P['gamma'] = 0.025

    # RAS periodic source params (gtild, lines 264-269)
    P['gA'] = 1e2     # A amplitude
    P['gB'] = np.pi / 6.0
    P['gC'] = 0.0

    # treat params -- dose_strength = 0 -> identically zero (lines 315-322)
    P['dose_strength'] = 0.0

    return P


# ----------------------------------------------------------------------------
# initial state (27,)  -- ICs differ in M_0, Mi_0, Ca_0
# ----------------------------------------------------------------------------
def initial_state(P):
    w = P['w']
    # WNT ICs (lines 502-515 / 532-545) -- identical both models, scaled by 1/w
    V_0, Di_0, Db_0, Bp_0 = 0.0, 4.83e-3, 2.02e-3, 1.0
    Da_0, P_0, Ba_0, X_0 = 9.66e-3, 18.116, 25.1, 4.93e-4
    W_0n = (1.0 / w) * np.array([V_0, Di_0, Db_0, Bp_0, Da_0, P_0, Ba_0, X_0])

    # RAS ICs (lines 518-533 / 548-563) -- identical, scaled by 1/1000
    R_0n = (1.0 / 1000.0) * np.array(
        [10.0, 10.0, 10.0, 100.0, 1.0, 0.0, 1.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0])

    # HOX ICs -- DIFFER between models
    H5_0 = 0.00146
    H13_0 = 0.04439
    Ci_0 = 0.0012
    if P['model'] == "const":
        M_0, Mi_0, Ca_0 = 0.055, 0.058, 0.8984          # lines 541,543,545
    else:
        M_0, Mi_0, Ca_0 = 0.04999, 0.0601, 0.84028      # lines 572,575,578
    H_0n = np.array([H5_0, M_0, Mi_0, Ca_0, H13_0, Ci_0])

    return np.concatenate([R_0n, W_0n, H_0n]).astype(np.float64)


# ----------------------------------------------------------------------------
# forcing functions (torch)
# ----------------------------------------------------------------------------
def forcing_W(t, P):
    """Wnt signal: amp on [3000,25000], zero otherwise."""
    active = (t >= 3000.0) & (t <= 25000.0)
    return P['Wamp'] * active.to(t.dtype)


def forcing_gtild(t, P):
    """RA periodic source: (a/r)*A*(1+cos(B*a*t - C))."""
    a, r = P['a'], P['r']
    return (a / r) * P['gA'] * (1.0 + torch.cos(P['gB'] * a * t - P['gC']))


def treat(t, P):
    """Treatment dose (identically 0 since dose_strength=0)."""
    return torch.zeros_like(t)


# ----------------------------------------------------------------------------
# couplings (torch)
# ----------------------------------------------------------------------------
def K16_dyn(Ci, P):
    """Fixed K16=30 for const model, Hill function of Ci for dyn model."""
    if P['model'] == "const":
        return torch.full_like(Ci, P['K16'])
    w = P['w']
    nc = P['nc']
    num = P['K16_max'] * ((w * Ci) ** nc)
    den = P['Cs'] ** nc + ((w * Ci) ** nc)
    return P['K16_0'] + num / den


def bctf(Ba, Ci, P):
    """Beta-catenin/TCF complex."""
    w = P['w']
    K16 = K16_dyn(Ci, P)
    return P['TCF0'] * Ba / (K16 + w * Ba)


def cyp_synth(R, Ba, Ci, P):
    """CYP synthesis rate."""
    r = P['r']
    bt = bctf(Ba, Ci, P)
    return (P['rv18'] + P['wntef'] * bt + P['MCsynth'] * (r * R) ** 2) / (1.0 + (r * R) ** 2)


def APCdeg(R, Pv, P):
    """APC degradation rate, piecewise on P vs 0.75*Pmin."""
    r = P['r']
    thr = 0.75 * P['Pmin']
    hi = P['k19n'] / (1.0 + P['retef'] * r * R ** 2)
    lo = P['k19n'] * (1.0 + P['retef'] * r * R ** 2)
    mask = (Pv >= thr).to(Pv.dtype)
    return hi * mask + lo * (1.0 - mask)


# ----------------------------------------------------------------------------
# residuals
# ----------------------------------------------------------------------------
def residuals(t, z, dz, P):
    """Return (res_expl [N,23], res_impl [N,4]): residuals for explicit states and WNT mass-matrix block."""
    w, r, a = P['w'], P['r'], P['a']

    def g(i):
        return z[:, i:i + 1]

    Ro, Ra, A, Rv, Bx, Br, Nx, Nr, Cx, Cr, Dc, Dn, Bc = [g(i) for i in range(0, 13)]
    V, Di, Db, Bp, Da, Pv, Ba, X = [g(i) for i in range(13, 21)]
    H5, M, Mi, Ca, H13, Ci = [g(i) for i in range(21, 27)]

    # ----- RAS explicit RHS (lines 425-437 / 450-464) -----------------------
    fRo = a * (P['rkm1'] * Ra - P['rkp1'] * Ro) + forcing_gtild(t, P)
    fRa = a * (P['rkp1'] * Ro - (P['rkm1'] + P['rk2'] * r * A) * Ra)
    fA = (a / r) * P['rv19'] - a * (P['rk23'] * A)
    fR = treat(t, P) + a * (
        P['rk2'] * r * Ra * A - P['rkp3'] * r * Rv * Bx + (P['rkm3'] + P['rk14']) * Br
        - P['rkp4'] * r * Rv * Nx + (P['rkm4'] + P['rk15']) * Nr
        - P['rkp5'] * r * Rv * Cx + P['rkm5'] * Cr)
    fB = (a / r) * P['rv16'] + a * (
        -(P['rk20'] + P['rkp3'] * r * Rv) * Bx + P['rkm3'] * Br + P['rk10'] * Dn)
    fBr = a * (
        P['rkp3'] * r * Rv * Bx - P['rkm3'] * Br - P['rkp6'] * r * Br * Cx
        + P['rkm6'] * Dc - P['rkp9'] * r * Br * Nx + P['rkm9'] * Dn - P['rk14'] * Br)
    fN = (a / r) * P['rv17'] + a * (
        -(P['rk21'] + P['rkp4'] * r * Rv) * Nx + P['rkm4'] * Nr
        - P['rkp9'] * r * Br * Nx + P['rkm9'] * Dn)
    fNr = a * (
        P['rkp4'] * r * Rv * Nx - (P['rkm4'] + P['rk15']) * Nr + P['rk10'] * Dn)
    fC = (a / r) * cyp_synth(Rv, Ba, Ci, P) + a * (
        -(P['rk22'] + P['rkp5'] * r * Rv) * Cx + (P['rkm5'] + P['rk8']) * Cr
        - P['rkp6'] * r * Br * Cx + P['rkm6'] * Dc)
    fCr = a * (P['rkp5'] * r * Rv * Cx - (P['rkm5'] + P['rk8']) * Cr)
    fDc = a * (P['rkp6'] * r * Br * Cx - (P['rkm6'] + P['rk7']) * Dc)
    fDn = a * (P['rkp9'] * r * Br * Nx - (P['rkm9'] + P['rk10']) * Dn)
    # dBcdtn DIFFERS:  const = a*rk7*Bc (line 437) ; dyn = a*rk7*Dc (line 464)
    if P['model'] == "const":
        fBc = a * P['rk7'] * Bc
    else:
        fBc = a * P['rk7'] * Dc

    # ----- WNT explicit RHS (lines 329-336 / 333-340) -----------------------
    fV = P['k1n'] * (P['dsh0'] - V) * forcing_W(t, P) - P['k2n'] * V
    pool = (P['gsk0'] - (1.0 + P['K8n'] * Ba) * Da - Di - Db)  # gsk0-(1+K8n*Ba)Da-Di-Db
    fDi = -(P['k3n'] * V + P['k4n'] + P['k_6n']) * Di + Da \
        + (P['k6n'] * Pv * X * pool) / (P['K21'] + w * X)
    fDb = P['k9n'] * Da * Ba - P['k10n'] * Db
    fBp = P['k10n'] * Db - P['k11n'] * Bp

    # ----- HOX explicit RHS (lines 444-465 / 471-493) -----------------------
    # k33(Nr) = k33_max*(w*Nr)^n/(Kd^n+(w*Nr)^n) ; k33n = k33/k5 (line 201,205)
    ne = P['nexp']
    k33n = ((P['k33_max'] * (w * Nr) ** ne) / (P['Kd'] ** ne + (w * Nr) ** ne)) / P['k5']
    # k35n(Ca) = (k34 + a3*(w*Ca)^n3)/(1+(phi*w*Ca)^n3)/k5  (line 444 / 471)
    n3 = P['n3']
    k35n = (P['k34'] + P['a3'] * (w * Ca) ** n3) / (1.0 + (P['phi'] * w * Ca) ** n3) / P['k5']
    # v30n(bctf) = (Kc + a2*w*bctf)/(1+w*bctf)/(w*k5)  (line 445 / 472)
    bt = bctf(Ba, Ci, P)
    v30n = (P['Kc'] + P['a2'] * w * bt) / (1.0 + w * bt) / (w * P['k5'])

    # dH5dtn = k31n + k32n*Mi + k33n(Nr)*Nr - k35n(Ca)*H5     (line 458 / 485)
    fH5 = P['k31n'] + P['k32n'] * Mi + k33n * Nr - k35n * H5
    # dMdtn  = v30n(bctf) - k36n*M - kp37n*M*Mi + km37n*Ca    (line 460 / 488)
    fM = v30n - P['k36n'] * M - P['kp37n'] * M * Mi + P['km37n'] * Ca
    # dMidtn = v31n - k38n*Mi - kp37n*M*Mi + km37n*Ca         (line 461 / 489)
    fMi = P['v31n'] - P['k38n'] * Mi - P['kp37n'] * M * Mi + P['km37n'] * Ca
    # dCadt  = kp37n*M*Mi - km37n*Ca                          (line 462 / 490)
    fCa = P['kp37n'] * M * Mi - P['km37n'] * Ca
    # dH13dtn= v32n - k39n*H13 - kp40n*H13*Ba + km40n*Ci      (line 463 / 491)
    fH13 = P['v32n'] - P['k39n'] * H13 - P['kp40n'] * H13 * Ba + P['km40n'] * Ci
    # dCidtn = kp40n*H13*Ba - km40n*Ci                        (line 465 / 493)
    fCi = P['kp40n'] * H13 * Ba - P['km40n'] * Ci

    # assemble the explicit RHS for all 27 (WNT implicit slots filled with 0;
    # they are dropped from res_expl).
    z0 = torch.zeros_like(Da)
    f = torch.cat([fRo, fRa, fA, fR, fB, fBr, fN, fNr, fC, fCr, fDc, fDn, fBc,
                   fV, fDi, fDb, fBp, z0, z0, z0, z0,
                   fH5, fM, fMi, fCa, fH13, fCi], dim=1)

    expl = [i for i in range(27) if i not in (iDa, iP, iBa, iX)]  # 23 indices
    res_expl = dz[:, expl] - f[:, expl]

    # ----- WNT implicit mass-matrix block -----------------------------------
    K8n, K7n, K17n, K20n, K21 = P['K8n'], P['K7n'], P['K17n'], P['K20n'], P['K21']
    K8 = P['K8']

    # Matrix entries An..Nn (lines 340-354 / 344-361). Identical EXCEPT In.
    An_ = -K8n * (K8 + w * Ba) * X / (K21 + w * X)
    Bn_ = K7n * X
    Cn_ = K20n * X - w * K8n * Da * X / (K21 + w * X)
    Dn_ = 1.0 + K7n * Pv + K20n * Ba + K21 * w * pool / ((K21 + w * X) ** 2)
    En_ = 1.0 + K8n * Ba
    Fn_ = K8n * Da
    Gn_ = K8n * Ba
    Hn_ = K17n * Ba
    # In DIFFERS:
    #   const  : 1 + K8n*Da + K16n*tcf0/((K16+w*Ba)^2) + K17n*P + K20n*X  (line 349, K16=30,K16n=K16*w)
    #   dyn    : 1 + K8n*Da + (K16(Ci)*tcf0)/((K16(Ci)+w*Ba)^2) + K17n*P + K20n*X  (line 354-355)
    if P['model'] == "const":
        In_ = 1.0 + K8n * Da + P['K16n'] * P['tcf0'] / ((P['K16'] + w * Ba) ** 2) \
            + K17n * Pv + K20n * X
    else:
        K16e = K16_dyn(Ci, P)
        In_ = 1.0 + K8n * Da + (K16e * P['tcf0']) / ((K16e + w * Ba) ** 2) \
            + K17n * Pv + K20n * X
    Jn_ = K20n * Ba
    Kn_ = 1.0 + K8n * Ba
    Ln_ = 1.0 + K7n * X + K17n * Ba
    Mn_ = K8n * Da + K17n * Pv
    Nn_ = K7n * Pv

    # RHS vector entries (lines 358-376 / 365-388).
    # RHS1n (line 358 / 365)
    RHS1 = P['v14n'] + (P['k3n'] * V + P['k_6n']) * Di \
        - P['k6n'] * Pv * X * pool / (K21 + w * X) \
        - P['k15n'] * Pv * X / (P['Km'] + w * Pv) \
        + w * X / (K21 + w * X) * (fDi + fDb)
    # RHS2n (line 365 / 372)
    RHS2 = P['k4n'] * Di - (1.0 + P['k9n'] * Ba) * Da + P['k10n'] * Db
    # RHS3n (line 368 / 375): v12n - (k13n+k9n*Da+kp40n*H13+k41n*H5)*Ba + km40n*Ci
    RHS3 = P['v12n'] - (P['k13n'] + P['k9n'] * Da + P['kp40n'] * H13
                        + P['k41n'] * H5) * Ba + P['km40n'] * Ci
    # RHS4n APC synthesis (line 373 / 385). DIFFERS in the denominator K16 term.
    #   const : v18n/(1 + Ktn*Ba/(K16 + w*Ba) + Kbn*Ba + gamma*w*H13) - APCdeg*P - (fDi+fDb)
    #           with K16=30
    #   dyn   : same but K16 -> K16_0 + K16_max*(w*Ci)^nc/(Cs^nc+(w*Ci)^nc)
    if P['model'] == "const":
        Kden = P['K16']                 # 30
    else:
        Kden = K16_dyn(Ci, P)           # dynamic K16(Ci)
    RHS4 = P['v18n'] / (1.0 + P['Ktn'] * Ba / (Kden + w * Ba)
                        + P['Kbn'] * Ba + P['gamma'] * w * H13) \
        - APCdeg(Rv, Pv, P) * Pv - (fDi + fDb)

    # unknown derivative vector order is [dDa, dP, dBa, dX]
    dDa = dz[:, iDa:iDa + 1]
    dP = dz[:, iP:iP + 1]
    dBa = dz[:, iBa:iBa + 1]
    dX = dz[:, iX:iX + 1]

    # Matrix rows (MATLAB lines 389-404 / 404-419):
    #   row1: [An, Bn, Cn, Dn] -> RHS1
    #   row2: [En, 0,  Fn, 0 ] -> RHS2
    #   row3: [Gn, Hn, In, Jn] -> RHS3
    #   row4: [Kn, Ln, Mn, Nn] -> RHS4
    r1 = An_ * dDa + Bn_ * dP + Cn_ * dBa + Dn_ * dX - RHS1
    r2 = En_ * dDa + 0.0 * dP + Fn_ * dBa + 0.0 * dX - RHS2
    r3 = Gn_ * dDa + Hn_ * dP + In_ * dBa + Jn_ * dX - RHS3
    r4 = Kn_ * dDa + Ln_ * dP + Mn_ * dBa + Nn_ * dX - RHS4

    res_impl = torch.cat([r1, r2, r3, r4], dim=1)
    return res_expl, res_impl


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    ok = True
    for model in ("const", "dyn"):
        P = build_params(model)
        z0 = initial_state(P)
        assert z0.shape == (27,), f"{model}: initial_state shape {z0.shape}"
        N = 4
        t = torch.rand(N, 1, dtype=torch.float64) * P['T']
        z = torch.rand(N, 27, dtype=torch.float64) + 0.1   # positive to avoid /0
        dz = torch.randn(N, 27, dtype=torch.float64)
        res_expl, res_impl = residuals(t, z, dz, P)
        print(f"[{model}] T={P['T']} N_t={P['N_t']} Wamp={P['Wamp']} "
              f"gamma={P['gamma']:.5g}  res_expl={tuple(res_expl.shape)} "
              f"res_impl={tuple(res_impl.shape)}")
        assert res_expl.shape == (N, 23), f"{model}: res_expl {res_expl.shape}"
        assert res_impl.shape == (N, 4), f"{model}: res_impl {res_impl.shape}"
        assert torch.isfinite(res_expl).all() and torch.isfinite(res_impl).all(), \
            f"{model}: non-finite residual"
        # spot-check a few model-specific values
        print(f"        IC[M,Mi,Ca]={z0[iM]:.5g},{z0[iMi]:.5g},{z0[iCa]:.5g} "
              f"K16n={P.get('K16n'):.5g}")
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
