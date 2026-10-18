"""Compatibility imports for the shared integral rhs."""

from wnt_pinn.model.integral_rhs import (
    _hill_t, _pulse_t, _ra_t, physics_residual, physics_rhs,
)

__all__ = ["_hill_t", "_pulse_t", "_ra_t", "physics_residual", "physics_rhs"]
