import numpy as np
from scipy.integrate import solve_ivp


def APC_Functioning_Change_NonDimensional(
    funcpercent, APC_IC, Axin_synth, k8log, k17log
):

    dt   = 1e-2
    tmin = 0.0
    tmax = 12000.0

    Dsh0 = 100.0
    TCF0 = 15.0
    APC0 = 100.0
    GSK0 = 50.0
    if k8log and k17log:
        K8  = funcpercent * 120.0
        K17 = funcpercent * 1200.0
    elif k8log:
        K8  = funcpercent * 120.0
        K17 = 1200.0
    elif k17log:
        K17 = funcpercent * 1200.0
        K8  = 120.0
    else:
        K17 = 1200.0
        K8  = 120.0

    K7  = 50.0
    K16 = 30.0
    K20 = 1.0
    K21 = 1.0
    Km  = 98.0

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
    v14 = (8.22e-5) * Axin_synth  
    k15 = 0.33

    Parameters = [212.8453, 39.9102, 34.1111]
    k19  = 1.0 / K17
    v18  = Parameters[0] * k19
    Kt   = Parameters[1]
    Kb   = Parameters[2]

    gsk0 = GSK0 * k5 / v14
    tcf0 = TCF0 / K16

    K7n  = K17 / K7
    K8n  = K16 / K8
    K16a = K16 / K21
    K16p = K16 / K17
    K20n = K21 / K20
    K21n = K21 / K7
    Kmn  = Km  / K17
    Ktn  = TCF0 / Kt
    Kbn  = K16 / Kb

    k1n  = k1  / k5
    k2n  = k2  / k5
    k3n  = Dsh0 * k3 / k5
    k4n  = k4  / k5
    k6n  = k6  * K17 * K21 / K7 / k5
    k_6n = k_6 / k5
    k9n  = k9  * K16 / k5 / K8
    k10n = k10 / k5
    v10n = k10 * k10n / k11
    k11n = k11 / k5
    v12n = v12 / K16 / k5
    k13n = k13 / k5
    v14a = v14 / K21 / k5
    v14b = v14 / K8  / k5
    v14p = v14 / K17 / k5
    k14n = k9  * v14b / k5
    k15n = k15 / k5
    v18n = v18 / K17 / k5
    k19n = k19 / k5

    def W(t):
        return float((t >= 100.0) and (t <= 8000.0))

    def dVdt(t, V, Di, Db, Bp, Da, P, Ba, X):
        return k1n * (1.0 - V) * W(t) - k2n * V

    def dDidt(t, V, Di, Db, Bp, Da, P, Ba, X):
        return (-(k3n * V + k4n + k_6n) * Di
                + Da
                + k6n * P * X * (gsk0 - (1.0 + K8n * Ba) * Da - Di - Db) / (1.0 + X))

    def dDbdt(t, V, Di, Db, Bp, Da, P, Ba, X):
        return k9n * Da * Ba - k10n * Db

    def dBpdt(t, V, Di, Db, Bp, Da, P, Ba, X):
        return v10n * Db - k11n * Bp

    def build_system(t, V, Di, Db, Bp, Da, P, Ba, X):
        dDi = dDidt(t, V, Di, Db, Bp, Da, P, Ba, X)
        dDb = dDbdt(t, V, Di, Db, Bp, Da, P, Ba, X)

        Gsk_free = gsk0 - (1.0 + K8n * Ba) * Da - Di - Db

        A = -v14a * (1.0 + K8n * Ba) * X / (1.0 + X)
        B = K7n * X
        C = K16a * (K20n * X - v14b * Da * X / (1.0 + X))
        D = (1.0 + K7n * P + K16a * K20n * Ba
             + v14a * Gsk_free / (1.0 + X) ** 2)
        E = 1.0 + K8n * Ba
        F = K8n * Da
        G = v14b * Ba
        H = Ba
        I = 1.0 + v14b * Da + tcf0 / (1.0 + Ba) ** 2 + P + K20n * X
        J = K20n * Ba
        K = v14p * (1.0 + K8n * Ba)
        L = 1.0 + K21n * X + K16p * Ba
        M = K16p * (v14b * Da + P)
        N = K21n * P

        RHS1 = (v14a
                + v14a * (k3n * V + k_6n) * Di
                - v14a * k6n * P * X * Gsk_free / (1.0 + X)
                - k15n * P * X / (Kmn + P)
                + v14a * X / (1.0 + X) * (dDi + dDb))

        RHS2 = k4n * Di - (1.0 + k9n * Ba) * Da + k10n * Db

        RHS3 = v12n - (k13n + k14n * Da) * Ba

        RHS4 = (v18n / (1.0 + Ktn * Ba / (1.0 + Ba) + Kbn * Ba)
                - k19n * P
                - v14p * (dDi + dDb))

        mat = np.array([
            [A, B, C, D],
            [E, 0, F, 0],
            [G, H, I, J],
            [K, L, M, N],
        ])
        vec = np.array([RHS1, RHS2, RHS3, RHS4])
        return mat, vec


    def RHS(t, U):
        V, Di, Db, Bp, Da, P, Ba, X = U

        dV  = dVdt (t, V, Di, Db, Bp, Da, P, Ba, X)
        dDi = dDidt(t, V, Di, Db, Bp, Da, P, Ba, X)
        dDb = dDbdt(t, V, Di, Db, Bp, Da, P, Ba, X)
        dBp = dBpdt(t, V, Di, Db, Bp, Da, P, Ba, X)

        mat, vec = build_system(t, V, Di, Db, Bp, Da, P, Ba, X)
        try:
            sol = np.linalg.solve(mat, vec)  
        except np.linalg.LinAlgError:
            sol = np.zeros(4)

        dDa, dP, dBa, dX = sol
        return [dV, dDi, dDb, dBp, dDa, dP, dBa, dX]

    V_0  = 0.0 / Dsh0
    Di_0 = 4.83e-3 * k5 / v14
    Db_0 = 2.02e-3 * k5 / v14
    Bp_0 = 1.0 * k5 * k10 / v14 / k11
    Da_0 = 9.66e-3 * k5 / v14
    P_0  = APC_IC / K17
    Ba_0 = 25.1 / K16
    X_0  = 4.93e-4 / K21

    U0 = [V_0, Di_0, Db_0, Bp_0, Da_0, P_0, Ba_0, X_0]
    t_eval = np.arange(tmin, tmax + dt, dt)
    sol = solve_ivp(
        RHS,
        [tmin, tmax],
        U0,
        method="Radau",
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-8,
        dense_output=False,
    )

    t_out = sol.t
    U     = sol.y  
    P_out  = U[5]                       
    Ba_out = U[6]                       
    u14    = tcf0 * Ba_out / (1.0 + Ba_out) 

    APC_conc      = P_out
    bcat_TCF_conc = u14
    tspan         = t_out

    return APC_conc, bcat_TCF_conc, tspan


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    APC_conc, bcat_TCF_conc, tspan = APC_Functioning_Change_NonDimensional(
        funcpercent=1.0,   
        APC_IC=18.116,     
        Axin_synth=1.0,    
        k8log=False,
        k17log=False,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(tspan, APC_conc, color="steelblue")
    axes[0].set_ylabel("APC (non-dim)")
    axes[0].set_title("APC Concentration")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tspan, bcat_TCF_conc, color="darkorange")
    axes[1].set_ylabel("β-catenin/TCF (non-dim)")
    axes[1].set_xlabel("Time")
    axes[1].set_title("β-catenin / TCF Complex")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/wnt_simulation.png", dpi=150)
    plt.show()
    print("Done.")