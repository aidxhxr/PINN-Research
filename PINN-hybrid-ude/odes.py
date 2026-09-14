"""Compatibility imports for the shared seven-state reference equations."""

from wnt_pinn.model.numpy_rhs import (
    _hill, _myc_input, _ode_rhs, _pulse, _ra_input, _wnt_input,
)

__all__ = ["_hill", "_myc_input", "_ode_rhs", "_pulse", "_ra_input", "_wnt_input"]
