import numpy as np
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor

from config import BASELINE, REGIMES, Y0
from odes import _ode_rhs


def _solve_one(args):
    name, T, n_pts = args
    p = {**BASELINE, **REGIMES[name]}
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p),
                    (0, T), Y0,
                    t_eval=np.linspace(0, T, n_pts),
                    method="Radau", rtol=1e-10, atol=1e-12)
    assert sol.success, f"{name} failed: {sol.message}"
    return name, sol.t, sol.y.T


def generate_references(T=150.0, n_pts=5000):
    """refs[regime] = (t, y) — the ground-truth forward solution per regime."""
    refs = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for name, t, y in pool.map(
                _solve_one, [(n, T, n_pts) for n in REGIMES]):
            refs[name] = (t, y)
            print(f"  {name:<20s}  t=[0, {t[-1]:.0f}]  shape={y.shape}")
    return refs
