import numpy as np
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor

from config import BASELINE, REGIMES, CONDITIONS, Y0
from odes import _ode_rhs


def _solve_one(args):
    """Solve one (regime, condition) pair. The condition only overrides the
    KNOWN RA-forcing protocol parameters; the biological parameters are the
    regime's true values."""
    name, cond_name, forcing, T, n_pts = args
    p = {**BASELINE, **REGIMES[name], **forcing}
    sol = solve_ivp(lambda t, y: _ode_rhs(t, y, p),
                    (0, T), Y0,
                    t_eval=np.linspace(0, T, n_pts),
                    method="Radau", rtol=1e-10, atol=1e-12)
    assert sol.success, f"{name}/{cond_name} failed: {sol.message}"
    return name, cond_name, sol.t, sol.y.T


def generate_references(T=150.0, n_pts=5000):
    """refs[regime][condition] = (t, y) for every (regime, condition) pair."""
    jobs = [(n, c["name"], c["forcing"], T, n_pts)
            for n in REGIMES for c in CONDITIONS]
    refs = {n: {} for n in REGIMES}
    with ProcessPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        for name, cond_name, t, y in pool.map(_solve_one, jobs):
            refs[name][cond_name] = (t, y)
            print(f"  {name:<20s} {cond_name:<10s} "
                  f"t=[0, {t[-1]:.0f}]  shape={y.shape}")
    return refs
