import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve as lin_solve

def bctf(x, Ci, *, TCF0=15.0, w=1000.0, K16_max=200.0, K16_0=100.0,
         Cs=0.005, nc=2):
    
    K16 = K16_0 + (K16_max * (w * Ci) ** nc) / (Cs ** nc + (w * Ci) ** nc)
    return TCF0 * x / (K16 + w * x)

def modelsys_2025_k16dynamic(p, retef, wntef, CCret, CCwnt, funcpercent,
                              k8log, k17log, gamma1):
    
    N_t  = 5000
    t_0  = 0.0
    t_N  = 30000.0
    time = np.linspace(t_0, t_N, N_t)

    DSH0 = 100.0
    TCF0 = 15.0
    APC0 = 100.0
    GSK0 = 50.0

    K8  = (1.0 + (1.0 - funcpercent)) * 120.0 if k8log  else 120.0
    K17 = (1.0 + (1.0 - funcpercent)) * 1200.0 if k17log else 1200.0

    K16_max = 200.0 * gamma1
    K16_0   = 100.0
    Cs      = 0.005
    nc      = 2
    K7      = 50.0
    K20     = 1.0
    K21     = 1.0
    Km      = 98.0

    k1  = 0.182
    k2  = 1.82e-2
    k3  = 5e-2
    k4  = 0.267
    k5  = 0.133
    k6  = 9.09e-2
    k_6 = 0.909
    k9  = 206.0
    k10 = 206.0
    k11 = 0.417
    v12 = 0.423
    k13 = 2.57e-4
    v14 = 8.22e-5
    k15 = 0.33

    Parameters = np.array([212.8453, 39.9102, 34.1111])
    k19 = 1.0 / K17
    v18 = Parameters[0] * k19
    Kt  = Parameters[1]
    Kb  = Parameters[2]

    eta = 1.0
    if funcpercent < 1.0:
        v18 = v18 * (eta * funcpercent)

    w    = CCwnt
    gsk0 = GSK0 / w
    tcf0 = TCF0 / w
    dsh0 = DSH0 / w

    K7n  = w / K7
    K8n  = w / K8
    K17n = w / K17
    K20n = w / K20
    Ktn  = (TCF0 * w) / Kt
    Kbn  = w / Kb

    k1n  = k1 / k5
    k2n  = k2 / k5
    k3n  = k3 * w / k5
    k4n  = k4 / k5
    k6n  = (k6 * K21 * w**2) / (k5 * K7)
    k_6n = k_6 / k5
    k9n  = (k9 * w) / (k5 * K8)
    k10n = k10 / k5
    k11n = k11 / k5
    v12n = v12 / (w * k5)
    k13n = k13 / k5
    v14n = v14 / (k5 * w)
    k15n = k15 * w / k5
    v18n = v18 / (w * k5)
    k19n = k19 / k5

    rkp1 = p[0];  rkm1 = p[1];  rk2  = p[2]
    rkp3 = p[3];  rkm3 = p[4];  rkp4 = p[5]
    rkm4 = p[6];  rkp5 = p[7];  rkm5 = p[8]
    rkp6 = p[9];  rkm6 = p[10]; rk7  = p[11]
    rk8  = p[12]; rkp9 = p[13]; rkm9 = p[14]
    rk10 = p[15]
    rk11 = 0.0;   rk12 = 0.0;   rk13 = 0.0
    rk14 = p[16]; rk15 = p[17]
    rv16 = p[18]; rv17 = p[19]; rv18 = p[20]; rv19 = p[21]
    rk20 = p[22]; rk21 = p[23]; rk22 = p[24]; rk23 = p[25]
    MCsynth = p[26]

    k31  = 8.25  * 16.8182e2
    k32  = 20.85 * 1.25e1
    k34  = 0.25  * 10.0
    k36  = 0.13
    kp37 = 0.45  * 1.0e-1
    km37 = 0.1   * 1.65
    v31  = 0.15  * 3.333e1
    k38  = 0.083
    v32  = 0.20  * 3.333e1
    k39  = 0.15

    a2 = 7.735
    a3 = 50.5
    a4 = 1.5
    Kc = 1.0
    a1 = 55.5
    Kd = 0.01
    k33_max = 1.0
    n  = 1
    km_val = 0.25

    k_const = 1500.0
    a5 = 1.0;  a6 = 1.0
    n1 = 1;    n2 = 1;    n3 = 1
    knn1 = 200.5;  knn2 = 100.5
    phi  = 0.00139

    k31n  = k31  / (w * k5)
    k32n  = k32  / k5
    k36n  = k36  / k5
    kp37n = kp37 * w / k5
    km37n = km37 / k5
    v31n  = v31  / (w * k5)
    k38n  = k38  / k5
    v32n  = v32  / (w * k5)
    k39n  = k39  / k5
    knn   = k_const / (w * k5)

    kp40  = 2.5e-6
    km40  = 1.0e-3
    k41   = 0.1475e-4

    kp40n = (kp40 * w) / k5
    km40n = km40 / k5
    k41n  = (k41  * w) / k5

    r = CCret
    a = 1.0 / k5

    def _bctf(x, Ci):
        K16 = K16_0 + (K16_max * (w * Ci)**nc) / (Cs**nc + (w * Ci)**nc)
        return TCF0 * x / (K16 + w * x)

    def cyp_synthFunc(x, y, Ci):
        return (rv18 + wntef * _bctf(y, Ci) + MCsynth * (r * x)**2) / (1.0 + (r * x)**2)

    Pmin = 0.002

    def APCdeg(x, y):
        val = np.where(
            y >= 0.75 * Pmin,
            k19n / (1.0 + retef * r * x**2),
            k19n * (1.0 + retef * r * x**2)
        )
        return val

    gamma = 0.025

    def W(t):
        
        return 3.0 * float((t >= 3000.0) and (t <= 25000.0))

    A_amp = 1e2
    B_freq = np.pi / 6.0
    C_phase = 0.0

    def gtild(t):
        return (a / r) * A_amp * (1.0 + np.cos(B_freq * a * t - C_phase))

    t_start      = 10000.0
    t_end        = 15000.0
    dose_strength = 0.0
    q_smooth      = 100.0

    def treat(t):
        return (a / r) * (dose_strength / 2.0) * (
            np.tanh((t - t_start) / q_smooth) - np.tanh((t - t_end) / q_smooth)
        )

    treatvals = treat(time)

    def k33_func(Nr):
        return (k33_max * (w * Nr)**n) / (Kd**n + (w * Nr)**n)

    def k33n_func(Nr):
        return k33_func(Nr) / k5

    def k35n(Ca):
        return (k34 + a3 * (w * Ca)**n3) / (1.0 + (phi * w * Ca)**n3) / k5

    def v30n(bctf_val):
        return (Kc + a2 * w * bctf_val) / (1.0 + w * bctf_val) / (w * k5)

    def _K16dyn(Ci):
        return K16_0 + (K16_max * (w * Ci)**nc) / (Cs**nc + (w * Ci)**nc)

    def An(V, Di, Db, Bp, Da, P, Ba, X):
        return -K8n * (K8 + w * Ba) * X / (K21 + w * X)

    def Bn(V, Di, Db, Bp, Da, P, Ba, X):
        return K7n * X

    def Cn(V, Di, Db, Bp, Da, P, Ba, X):
        return K20n * X - w * K8n * Da * X / (K21 + w * X)

    def Dn(V, Di, Db, Bp, Da, P, Ba, X):
        return (1.0 + K7n * P + K20n * Ba
                + K21 * w * (gsk0 - (1.0 + K8n * Ba) * Da - Di - Db)
                / (K21 + w * X)**2)

    def En(V, Di, Db, Bp, Da, P, Ba, X):
        return 1.0 + K8n * Ba

    def Fn(V, Di, Db, Bp, Da, P, Ba, X):
        return K8n * Da

    def Gn(V, Di, Db, Bp, Da, P, Ba, X):
        return K8n * Ba

    def Hn(V, Di, Db, Bp, Da, P, Ba, X):
        return K17n * Ba

    def In(V, Di, Db, Bp, Da, P, Ba, X, Ci):
        K16 = _K16dyn(Ci)
        return (1.0 + K8n * Da
                + K16 * tcf0 / (K16 + w * Ba)**2
                + K17n * P + K20n * X)

    def Jn(V, Di, Db, Bp, Da, P, Ba, X):
        return K20n * Ba

    def Kn(V, Di, Db, Bp, Da, P, Ba, X):
        return 1.0 + K8n * Ba

    def Ln(V, Di, Db, Bp, Da, P, Ba, X):
        return 1.0 + K7n * X + K17n * Ba

    def Mn_fn(V, Di, Db, Bp, Da, P, Ba, X):
        return K8n * Da + K17n * P

    def Nn(V, Di, Db, Bp, Da, P, Ba, X):
        return K7n * P

    def dVdtn(t, V, Di, Db, Bp, Da, P, Ba, X):
        return k1n * (dsh0 - V) * W(t) - k2n * V

    def dDidtn(t, V, Di, Db, Bp, Da, P, Ba, X):
        return (-(k3n * V + k4n + k_6n) * Di + Da
                + k6n * P * X * (gsk0 - (1.0 + K8n * Ba) * Da - Di - Db)
                / (K21 + w * X))

    def dDbdtn(t, V, Di, Db, Bp, Da, P, Ba, X):
        return k9n * Da * Ba - k10n * Db

    def dBpdtn(t, V, Di, Db, Bp, Da, P, Ba, X):
        return k10n * Db - k11n * Bp

    def RHS1n(t, V, Di, Db, Bp, Da, P, Ba, X, Ci):
        return (v14n + (k3n * V + k_6n) * Di
                - k6n * P * X * (gsk0 - (1.0 + K8n * Ba) * Da - Di - Db) / (K21 + w * X)
                - k15n * P * X / (Km + w * P)
                + w * X / (K21 + w * X) * (
                    dDidtn(t, V, Di, Db, Bp, Da, P, Ba, X)
                    + dDbdtn(t, V, Di, Db, Bp, Da, P, Ba, X)))

    def RHS2n(t, V, Di, Db, Bp, Da, P, Ba, X):
        return k4n * Di - (1.0 + k9n * Ba) * Da + k10n * Db

    def RHS3n(t, V, Di, Db, Bp, Da, P, Ba, X, H5, H13, Ci):
        return (v12n
                - (k13n + k9n * Da + kp40n * H13 + k41n * H5) * Ba
                + km40n * Ci)

    def RHS4n(t, V, Di, Db, Bp, Da, P, Ba, X, R, H13, Ci):
        K16 = _K16dyn(Ci)
        return (v18n / (1.0 + Ktn * Ba / (K16 + w * Ba) + Kbn * Ba + gamma * w * H13)
                - APCdeg(R, P) * P
                - (dDidtn(t, V, Di, Db, Bp, Da, P, Ba, X)
                   + dDbdtn(t, V, Di, Db, Bp, Da, P, Ba, X)))

    def solve_implicit(t, V, Di, Db, Bp, Da, P, Ba, X, R, H5, H13, Ci):
        
        M = np.array([
            [An(V,Di,Db,Bp,Da,P,Ba,X), Bn(V,Di,Db,Bp,Da,P,Ba,X),
             Cn(V,Di,Db,Bp,Da,P,Ba,X), Dn(V,Di,Db,Bp,Da,P,Ba,X)],
            [En(V,Di,Db,Bp,Da,P,Ba,X), 0.0,
             Fn(V,Di,Db,Bp,Da,P,Ba,X), 0.0],
            [Gn(V,Di,Db,Bp,Da,P,Ba,X), Hn(V,Di,Db,Bp,Da,P,Ba,X),
             In(V,Di,Db,Bp,Da,P,Ba,X,Ci), Jn(V,Di,Db,Bp,Da,P,Ba,X)],
            [Kn(V,Di,Db,Bp,Da,P,Ba,X), Ln(V,Di,Db,Bp,Da,P,Ba,X),
             Mn_fn(V,Di,Db,Bp,Da,P,Ba,X), Nn(V,Di,Db,Bp,Da,P,Ba,X)],
        ])
        rhs_vec = np.array([
            RHS1n(t, V, Di, Db, Bp, Da, P, Ba, X, Ci),
            RHS2n(t, V, Di, Db, Bp, Da, P, Ba, X),
            RHS3n(t, V, Di, Db, Bp, Da, P, Ba, X, H5, H13, Ci),
            RHS4n(t, V, Di, Db, Bp, Da, P, Ba, X, R, H13, Ci),
        ])
        return lin_solve(M, rhs_vec)

    def dRodtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return a * (rkm1 * Ra - rkp1 * Ro) + gtild(t)

    def dRadtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return a * (rkp1 * Ro - (rkm1 + rk2 * r * A) * Ra)

    def dAdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return (a / r) * rv19 - a * rk23 * A

    def dRdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return (treat(t)
                + a * (rk2 * r * Ra * A
                       - rkp3 * r * R * B + (rkm3 + rk14) * Br
                       - rkp4 * r * R * N + (rkm4 + rk15) * Nr
                       - rkp5 * r * R * C + rkm5 * Cr))

    def dBdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return ((a / r) * rv16
                + a * (-(rk20 + rkp3 * r * R) * B + rkm3 * Br + rk10 * Dn_var))

    def dBrdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return a * (rkp3 * r * R * B - rkm3 * Br
                    - rkp6 * r * Br * C + rkm6 * Dc
                    - rkp9 * r * Br * N + rkm9 * Dn_var
                    - rk14 * Br)

    def dNdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return ((a / r) * rv17
                + a * (-(rk21 + rkp4 * r * R) * N + rkm4 * Nr
                       - rkp9 * r * Br * N + rkm9 * Dn_var))

    def dNrdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return a * (rkp4 * r * R * N - (rkm4 + rk15) * Nr + rk10 * Dn_var)

    def dCdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc, Ba, Ci):
        return ((a / r) * cyp_synthFunc(R, Ba, Ci)
                + a * (-(rk22 + rkp5 * r * R) * C
                       + (rkm5 + rk8) * Cr
                       - rkp6 * r * Br * C + rkm6 * Dc))

    def dCrdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return a * (rkp5 * r * R * C - (rkm5 + rk8) * Cr)

    def dDcdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return a * (rkp6 * r * Br * C - (rkm6 + rk7) * Dc)

    def dDndtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return a * (rkp9 * r * Br * N - (rkm9 + rk10) * Dn_var)

    def dBcdtn(t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_var, Bc):
        return a * rk7 * Dc

    def dH5dtn(t, H5, M, Mi, Ca, H13, Ci, Nr, Ba):
        return k31n + k32n * Mi + k33n_func(Nr) * Nr - k35n(Ca) * H5

    def dMdtn(t, H5, M, Mi, Ca, H13, Ci, Nr, Ba):
        return v30n(_bctf(Ba, Ci)) - k36n * M - kp37n * M * Mi + km37n * Ca

    def dMidtn(t, H5, M, Mi, Ca, H13, Ci, Nr, Ba):
        return v31n - k38n * Mi - kp37n * M * Mi + km37n * Ca

    def dCadt(t, H5, M, Mi, Ca, H13, Ci, Nr, Ba):
        return kp37n * M * Mi - km37n * Ca

    def dH13dtn(t, H5, M, Mi, Ca, H13, Ci, Nr, Ba):
        return v32n - k39n * H13 - kp40n * H13 * Ba + km40n * Ci

    def dCidtn(t, H5, M, Mi, Ca, H13, Ci, Nr, Ba):
        return kp40n * H13 * Ba - km40n * Ci

    def RHSn(t, U):
        (Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_v, Bc,
         V, Di, Db, Bp, Da, P, Ba, X,
         H5, M, Mi, Ca, H13, Ci) = U

        ras_args = (t, Ro, Ra, A, R, B, Br, N, Nr, C, Cr, Dc, Dn_v, Bc)
        dRo  = dRodtn(*ras_args)
        dRa  = dRadtn(*ras_args)
        dA   = dAdtn(*ras_args)
        dR   = dRdtn(*ras_args)
        dB   = dBdtn(*ras_args)
        dBr  = dBrdtn(*ras_args)
        dN   = dNdtn(*ras_args)
        dNr  = dNrdtn(*ras_args)
        dC   = dCdtn(*ras_args, Ba, Ci)
        dCr  = dCrdtn(*ras_args)
        dDc  = dDcdtn(*ras_args)
        dDn  = dDndtn(*ras_args)
        dBc  = dBcdtn(*ras_args)

        dV   = dVdtn(t, V, Di, Db, Bp, Da, P, Ba, X)
        dDi  = dDidtn(t, V, Di, Db, Bp, Da, P, Ba, X)
        dDb  = dDbdtn(t, V, Di, Db, Bp, Da, P, Ba, X)
        dBp  = dBpdtn(t, V, Di, Db, Bp, Da, P, Ba, X)

        impl = solve_implicit(t, V, Di, Db, Bp, Da, P, Ba, X, R, H5, H13, Ci)
        dDa, dP, dBa, dX = impl

        hox_args = (t, H5, M, Mi, Ca, H13, Ci, Nr, Ba)
        dH5  = dH5dtn(*hox_args)
        dM   = dMdtn(*hox_args)
        dMi  = dMidtn(*hox_args)
        dCa  = dCadt(*hox_args)
        dH13 = dH13dtn(*hox_args)
        dCi  = dCidtn(*hox_args)

        return np.array([
            dRo, dRa, dA, dR, dB, dBr, dN, dNr, dC, dCr, dDc, dDn, dBc,
            dV, dDi, dDb, dBp, dDa, dP, dBa, dX,
            dH5, dM, dMi, dCa, dH13, dCi
        ])

    V_0  = 0.0;    Di_0 = 4.83e-3; Db_0 = 2.02e-3
    Bp_0 = 1.0;    Da_0 = 9.66e-3; P_0  = 18.116
    Ba_0 = 25.1;   X_0  = 4.93e-4
    W_0n = (1.0 / w) * np.array([V_0, Di_0, Db_0, Bp_0, Da_0, P_0, Ba_0, X_0])

    ras_raw = np.array([10., 10., 10., 100., 1., 0., 1., 0.,
                        0.01, 0., 0., 0., 0.])
    R_0n = (1.0 / 1000.0) * ras_raw

    H5_0  = 0.00146;  M_0   = 0.04999; Mi_0 = 0.0601
    Ca_0  = 0.84028;  H13_0 = 0.04439; Ci_0 = 0.0012
    H_0n = np.array([H5_0, M_0, Mi_0, Ca_0, H13_0, Ci_0])

    U_0n = np.concatenate([R_0n, W_0n, H_0n])

    sol = solve_ivp(
        RHSn,
        [t_0, t_N],
        U_0n,
        method='Radau',
        t_eval=time,
        rtol=1e-6,
        atol=1e-9,
        dense_output=False,
    )

    t   = sol.t
    U   = sol.y.T

    BcatTcf_col = _bctf(U[:, 19], U[:, 26])[:, np.newaxis]
    Sol = np.hstack([U, BcatTcf_col])

    return t, Sol, treatvals


if __name__ == "__main__":
    import numpy as np

    # Example parameter vector, with non-real values lol
    p = np.ones(27) * 0.1

    t, Sol, treatvals = modelsys_2025_k16dynamic(
        p          = p,
        retef      = 0.0,
        wntef      = 0.0,
        CCret      = 1.0,
        CCwnt      = 1000.0,
        funcpercent= 1.0,
        k8log      = False,
        k17log     = False,
        gamma1     = 1.0
    )

    print("Shape of Sol:", Sol.shape)
    print("Time points:", len(t))