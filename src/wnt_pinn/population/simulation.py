"""Deterministic SciPy integrations, matched controls, and numerical audits."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

from .model import CONTROL, NOMINAL, SEED, observables, rhs


@dataclass(frozen=True)
class Solver:
    method: str = 'BDF'
    rtol: float = 1e-9
    atol: float = 1e-11
    max_step: float = .5
    dt: float = .125
    precondition_hours: float = 720.
    horizon: float = 168.
    beta_clock: str = 'literal'


def integrate(p, protocol, y0, interval, solver=Solver(), times=None, full=True):
    """Integrate all states together; dense output locates events continuously."""
    sol = solve_ivp(rhs, interval, y0, args=(p, protocol, solver.beta_clock),
                    method=solver.method, rtol=solver.rtol, atol=solver.atol,
                    max_step=solver.max_step, t_eval=times, dense_output=full)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f'{protocol.name}: integration failed: {sol.message}')
    return sol


def precondition(p, solver=Solver()):
    """720 untreated hours ending at t=0, preserving the 24-hour RA phase.

    This is the prescribed finite warm-up, not an assumption that slow cell
    populations have reached exact equilibrium. We measure their residual drift.
    """
    return integrate(p, CONTROL, SEED, (-solver.precondition_hours, 0.), solver,
                     times=np.array([-24., 0.]))


def trajectory(p, protocol=NOMINAL, solver=Solver(), y0=None):
    if y0 is None:
        y0 = precondition(p, solver).y[:, -1]
    times = np.linspace(0., solver.horizon, round(solver.horizon / solver.dt) + 1)
    return integrate(p, protocol, y0, (0., solver.horizon), solver, times=times)


def paired(p, protocol=NOMINAL, solver=Solver(), y0=None):
    if y0 is None:
        y0 = precondition(p, solver).y[:, -1]
    return trajectory(p, CONTROL, solver, y0), trajectory(p, protocol, solver, y0)


def compare_at(control, treated, p, time):
    c = observables(control.sol(time), p)
    a = observables(treated.sol(time), p)
    result = {'time_h': float(time)}
    for key in ('NT', 'NS', 'FS', 'FA', 'SA', 'IH', 'nL', 'nA', 'nD',
                'b', 'p', 'h5', 'h13', 'm', 'r', 'c'):
        result.update({f'{key}_control': float(c[key]), f'{key}_treated': float(a[key]),
                       f'{key}_ratio': float(a[key] / c[key])})
    result['IH_change'] = float(a['IH'] - c['IH'])
    for key in ('SSC', 'RSC', 'RSC0', 'gSC'):
        result[key] = float(a[key])
    return result


def maintenance_events(control, treated, p, washout=88.):
    """First post-washout upward crossing of SSC=0.5, by Brent root finding."""
    times = treated.t[treated.t >= washout]
    values = observables(treated.sol(times), p)['SSC'] - .5
    candidates = np.flatnonzero((values[:-1] < 0.) & (values[1:] >= 0.))
    result = {'crossing_h': None}
    if len(candidates):
        i = candidates[0]
        crossing = brentq(lambda t: float(observables(treated.sol(t), p)['SSC'] - .5),
                          times[i], times[i+1], xtol=1e-10)
        result.update(crossing_h=float(crossing), after_washout_h=float(crossing-washout),
                      **compare_at(control, treated, p, crossing))
    # Local minimization in the adjacent output-grid interval around each minimum.
    for key, label in [('SSC', 'min_SSC'), ('NS', 'nadir_NS')]:
        o = observables(treated.y, p)[key]
        i = int(np.argmin(o))
        bounds = treated.t[max(0, i-1)], treated.t[min(len(o)-1, i+1)]
        opt = minimize_scalar(lambda t: float(observables(treated.sol(t), p)[key]),
                              bounds=bounds, method='bounded')
        result[label] = {'time_h': float(opt.x), 'value': float(opt.fun)}
    return result


def audit(sol, p, protocol, solver=Solver()):
    """Independent balance identities and invariant-region checks on saved states."""
    y = sol.y
    o = observables(y, p)
    derivatives = np.array([rhs(t, state, p, protocol, solver.beta_clock)
                            for t, state in zip(sol.t, y.T)]).T
    stem_error = np.max(np.abs(derivatives[11] + derivatives[12] - (o['Jren'] - o['Jexit'])))
    total_expected = o['Jren'] - p.muL*y[11] - p.muA*y[12] - p.muD*y[13]
    total_error = np.max(np.abs(derivatives[11:14].sum(axis=0) - total_expected))
    index_error = np.max(np.abs((o['Jren']+o['Jexit'])*(2.*o['SSC']-1.)
                                - derivatives[11] - derivatives[12]))
    result = {'minimum_state': float(y.min()), 'min_program': float(y[7:11].min()),
              'max_program': float(y[7:11].max()), 'max_total_population': float(o['NT'].max()),
              'stem_balance_max_error': float(stem_error), 'total_balance_max_error': float(total_error),
              'maintenance_identity_max_error': float(index_error), 'nfev': sol.nfev,
              'njev': sol.njev, 'nlu': sol.nlu}
    result['passed'] = bool(y.min() >= -1e-9 and y[7:11].max() <= 1.+1e-9
                            and o['NT'].max() <= 1.+1e-9
                            and max(stem_error, total_error, index_error) < 1e-12)
    if not result['passed']:
        raise RuntimeError(f'Numerical invariant/balance audit failed: {result}')
    return result
