"""Equations (6)--(32) of ``wnt_hox_ra_revised_population_dynamics.pdf``.

Time is in hours. Molecular states have species-specific concentration scales;
HOX programs are scores in [0, 1]; populations are divided by 10**6 cells.
No network, fitting, smoothing, or prescribed response curves are used.

Equation (6) omits a printed dB multiplier while the adjacent text specifies
physical rates dX=dB/epsilonX. ``beta_clock='physical'`` includes dB; ``literal``
implements the printed equation without it and matches the reported beta-catenin
ratio. The study uses ``literal`` and compares both explicitly.
All other molecular rates are calculated from the stated half-lives, avoiding
rounding the four-digit rates in the prose. Population rates are already h^-1
and must NOT receive a second dN multiplier.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.special import expit

STATE_NAMES = ('b', 'p', 'h5', 'h13', 'm', 'r', 'c',
               'uBL', 'uCL', 'uBA', 'uCA', 'nL', 'nA', 'nD')
SEED = np.array([.20, 1., .80, .30, .30, .60, .40, .60, .40, .40, .60, .10, .15, .40])
REGIMES = {'Normal': (.8, 1.), 'Early adenoma': (1., .75),
           'Advanced adenoma': (1.5, .5), 'Severe APC loss': (2., .25)}
CONCENTRATION_SCALES_NM = (100., 150., 5., 5., 50., 10., 25.)


@dataclass(frozen=True)
class Parameters:
    """Tables 3--7 and Sections 4--5, using readable ASCII parameter names.

    ``gLB`` is gamma_L^B, ``gAC`` gamma_A^C, ``qLAC`` q_LA^C,
    and ``qALB`` q_AL^B. ``dL5``/``dA5`` and ``dLr``/``dAr``
    are differentiation rates; ``d5`` is the molecular HOXA5 clock.
    """
    W: float = 2.
    thetaP: float = .25
    K: float = .5
    n: float = 2.
    vb13: float = .90
    kbp: float = 1.92
    kb5: float = 1.56
    ap5: float = 1.10
    apb: float = 1.10
    ap13: float = 1.30
    deltaP: float = 3.50
    s5: float = .15
    v5r: float = 2.50
    k5m: float = 2.50
    s13: float = .18
    v13b: float = .95
    v13m: float = .55
    q13r: float = 1.
    sm: float = .18
    vmb: float = 1.35
    krc: float = .85
    sc: float = .08
    vcr: float = 1.50
    vcb: float = 1.50
    mu0: float = .35
    AR: float = .04
    TR: float = 24.
    phi: float = 0.
    q: float = .30
    dB: float = np.log(2.) / (5. / 6.)
    dP: float = np.log(2.) / 6.
    d5: float = np.log(2.) / 6.
    d13: float = np.log(2.) / 6.
    dM: float = np.log(2.) / .5
    dR: float = np.log(2.) / 1.
    dC: float = np.log(2.) / 4.
    dU: float = np.log(2.) / 8.
    zBL: float = .50
    zCL: float = -.50
    zBA: float = -.50
    zCA: float = .50
    wBb: float = 1.40
    wBr: float = .35
    wB13: float = .80
    wCb: float = .80
    wCr: float = 1.60
    wC13: float = .80
    gL0: float = .004
    gLb: float = .016
    gLB: float = .010
    gA0: float = .003
    gAb: float = .010
    gAC: float = .012
    qLA0: float = .0005
    qLAC: float = .0020
    qAL0: float = .0010
    qALB: float = .0080
    dL0: float = .0015
    dL5: float = .012
    dLr: float = .010
    dA0: float = .0015
    dA5: float = .009
    dAr: float = .015
    chiL: float = 1.50
    chiA: float = 2.
    muL: float = .0010
    muA: float = .0015
    muD: float = .00722

    def changed(self, **kwargs):
        return replace(self, **kwargs)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Protocol:
    """Smooth added RA input (Eq. 13) and optional selective block (Eq. 32).

    Each window is (onset hours, offset hours, amplitude multiplier).
    ``amplitude`` is dimensionless intracellular input, not a drug concentration.
    The four-pulse endpoints are an explicit assumption, as the PDF does not
    list them: 40--46, 54--60, 68--74, 82--88 h, all at twice nominal input.
    """
    name: str = 'ATRA'
    amplitude: float = 1.5
    windows: tuple = ((40., 88., 1.),)
    block: float = 0.

    @property
    def nominal_area(self):
        return self.amplitude * sum((end - start) * weight for start, end, weight in self.windows)

    @property
    def end(self):
        return max((end for _, end, _ in self.windows), default=88.)


CONTROL = Protocol('Control', 0.)
NOMINAL = Protocol()


def hill(x, p):
    """Shared Hill response (Eq. 5), also supporting whole trajectory arrays.

    Tiny negative *trial* values from an implicit solver are guarded here;
    actual integrated states are never clipped and are checked separately.
    """
    xn = np.maximum(x, 0.) ** p.n
    return xn / (p.K ** p.n + xn)


def exposure(t, p, protocol):
    """Sum of tanh windows; tails and smooth washout are part of the ODE."""
    return sum(weight * .5 * (np.tanh(p.q * (t - start)) - np.tanh(p.q * (t - end)))
               for start, end, weight in protocol.windows)


def ra_input(t, p, protocol=CONTROL):
    return (p.mu0 + p.AR * (1. + np.cos(2. * np.pi * t / p.TR - p.phi))
            + protocol.amplitude * exposure(t, p, protocol))


def population_rates(y, p):
    """Physical growth, switching, and differentiation rates, Eqs. (21)--(23)."""
    hb, _, h5, _, _, hr, hc, hBL, hCL, hBA, hCA = hill(np.asarray(y)[:11], p)
    gL = p.gL0 + p.gLb * hb + p.gLB * hBL
    gA = p.gA0 + p.gAb * hb + p.gAC * hCA
    qLA = p.qLA0 + p.qLAC * hCL
    qAL = p.qAL0 + p.qALB * hBA
    dL = p.dL0 + p.dL5 * h5 + p.dLr * hr / (1. + p.chiL * hc)
    dA = p.dA0 + p.dA5 * h5 + p.dAr * hr / (1. + p.chiA * hc)
    return gL, gA, qLA, qAL, dL, dA


def rhs(t, y, p, protocol=CONTROL, beta_clock='literal'):
    """Full 14-state derivative, retaining the state order from Eq. (1)."""
    b, apc, h5, h13, m, r, c, uBL, uCL, uBA, uCA, nL, nA, nD = y
    hb, _, _, hh13, hm, hr, *_ = hill(np.asarray(y)[:7], p)
    # The concurrent selective probe only weakens the two Eq. (32) couplings.
    gate = np.clip(exposure(t, p, protocol), 0., 1.) if protocol.block else 0.
    risk_factor = 1. - protocol.block * gate
    db = p.W + p.vb13 * hh13 - b - p.kbp * apc * b - p.kb5 * h5 * b / (p.K + b)
    if beta_clock == 'physical':
        db *= p.dB
    elif beta_clock != 'literal':
        raise ValueError('beta_clock must be physical or literal')
    dp = p.dP * ((1. + p.ap5 * h5) / (1. + p.apb * b + p.ap13 * h13)
                 - (1. + p.deltaP * (1. - p.thetaP)) * apc)
    dh5 = p.d5 * (p.s5 + p.v5r * hr - h5 - p.k5m * m * h5 / (p.K + m))
    dh13 = p.d13 * ((p.s13 + p.v13b * hb + p.v13m * hm)
                    / (1. + risk_factor * p.q13r * hr) - h13)
    dm = p.dM * (p.sm + p.vmb * hb - m)
    dr = p.dR * (ra_input(t, p, protocol) - r - p.krc * c * r)
    dc = p.dC * (p.sc + p.vcr * hr + p.vcb * hb - c)
    psiB = p.wBb * hb + p.wBr * hr - p.wB13 * hh13
    psiC = p.wCb * hb + risk_factor * p.wCr * hr - p.wC13 * hh13
    du = p.dU * (expit([p.zBL + psiB, p.zCL + psiC,
                        p.zBA + psiB, p.zCA + psiC]) - y[7:11])
    gL, gA, qLA, qAL, dL, dA = population_rates(y, p)
    free = 1. - nL - nA - nD
    dnL = gL * nL * free - (dL + p.muL + qLA) * nL + qAL * nA
    dnA = gA * nA * free - (dA + p.muA + qAL) * nA + qLA * nL
    dnD = dL * nL + dA * nA - p.muD * nD
    return np.array([db, dp, dh5, dh13, dm, dr, dc, *du, dnL, dnA, dnD])


def observables(y, p):
    """Derived quantities, Eqs. (24)--(30), for a state or a 14 x time array.

    NT and NS are population/KN; FA=nA/NT is the full-population ALDH fraction;
    SA=nA/NS is a different observable (composition of the residual stem pool).
    Ratios with zero denominator are undefined and returned as NaN.
    """
    y = np.asarray(y)
    nL, nA, nD = y[11:14]
    nt, ns = nL + nA + nD, nL + nA
    gL, gA, _, _, dL, dA = population_rates(y, p)
    renewal0 = gL * nL + gA * nA
    renewal = (1. - nt) * renewal0
    exit_flux = (dL + p.muL) * nL + (dA + p.muA) * nA
    def div(a, b):
        return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)
    uc = div(nL * y[8] + nA * y[10], ns)
    return {**dict(zip(STATE_NAMES, y)), 'NT': nt, 'NS': ns, 'FS': div(ns, nt),
            'FA': div(nA, nt), 'SA': div(nA, ns), 'uCbar': uc,
            'IH': .5 * (1. + uc - hill(y[3], p)), 'Jren': renewal, 'Jexit': exit_flux,
            'SSC': div(renewal, renewal + exit_flux), 'RSC': div(renewal, exit_flux),
            'RSC0': div(renewal0, exit_flux), 'gSC': div(renewal - exit_flux, ns)}
