"""Interchangeable implementations of the existing integral physics objective."""

import torch

BACKENDS = ("eager", "triton", "compiled")


def trapezoidal_loss(z, f, dt, weight):
    """Original operation order and mean over intervals and seven states."""
    residual = ((z[1:] - z[:-1]) - 0.5 * dt * (f[1:] + f[:-1])) * weight
    return (residual**2).mean()


def make_integral_loss(rhs, backend="eager"):
    """Bind an RHS without changing its equations or learned mechanisms.

    ``triton`` replaces only the fixed-grid loss reduction and its backward.
    ``compiled`` lets Inductor optimize the complete RHS and loss together.
    Neither option changes the state networks or optimizer. Compilation is
    lazy; its cost belongs to the first call and must be reported separately.
    """
    if backend not in BACKENDS:
        raise ValueError(f"physics_backend must be one of {BACKENDS}")
    reduction = trapezoidal_loss
    if backend == "triton":
        from .trapezoid_triton import trapezoidal_loss as reduction

    def loss(t, z, parameters, weight, dt=None):
        f = rhs(t, z, parameters)
        return reduction(z, f, t[1:] - t[:-1] if dt is None else dt, weight)

    if backend == "compiled":
        # Let Dynamo generalize row counts after observing shape changes.
        # dynamic=True also symbolizes integer Hill exponents and triggers a
        # PowBackward SymInt assertion in PyTorch 2.12.
        # CUDA graphs are deliberately disabled: closures retain graphs and
        # their invocation count depends on the line search.
        compiled = torch.compile(loss, fullgraph=True, dynamic=None,
                                 options={"triton.cudagraphs": False})
        try:
            from torch.compiler import config as compiler_config
        except ImportError:  # PyTorch versions predating the public alias.
            from torch._dynamo import config as compiler_config
        limit_key = ("recompile_limit" if hasattr(compiler_config, "recompile_limit")
                     else "cache_size_limit")

        def dispatch(t, z, parameters, weight, dt=None):
            if not torch.is_grad_enabled():
                # Final scoring uses the reference implementation and needs
                # neither a backward graph nor another compiled specialization.
                return loss(t, z, parameters, weight, dt)
            # Ten forcing protocols, several grids and the frozen-state stage
            # legitimately exceed the default eight guarded specializations.
            # Scope the larger budget to this call; do not mutate global policy.
            with compiler_config.patch({limit_key: 128}):
                return compiled(t, z, parameters, weight, dt)

        return dispatch
    return loss
