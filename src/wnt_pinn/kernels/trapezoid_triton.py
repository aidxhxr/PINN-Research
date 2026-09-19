"""Fused seven-state trapezoidal MSE, with deterministic, atomics-free backward.

Inputs retain their floating-point dtype. The collocation intervals and state
weights are fixed, as in the maintained trainers. Gradients flow through both
predicted states and RHS values into all biological and learned parameters.
Triton is imported only when this optional backend is selected.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _forward(Z, F, DT, W, PARTS, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < (N - 1) * 7
    row, col = index // 7, index % 7
    z0 = tl.load(Z + index, mask, other=0)
    z1 = tl.load(Z + index + 7, mask, other=0)
    f0 = tl.load(F + index, mask, other=0)
    f1 = tl.load(F + index + 7, mask, other=0)
    dt = tl.load(DT + row, mask, other=0)
    weight = tl.load(W + col)
    residual = ((z1 - z0) - (0.5 * dt) * (f1 + f0)) * weight
    total = tl.sum(residual * residual, axis=0)
    tl.store(PARTS + tl.program_id(0), total)


@triton.jit
def _backward(Z, F, DT, W, GRAD, GZ, GF, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N * 7
    row, col = index // 7, index % 7
    left = valid & (row > 0)
    right = valid & (row < N - 1)
    z = tl.load(Z + index, valid, other=0)
    f = tl.load(F + index, valid, other=0)
    zm = tl.load(Z + index - 7, left, other=0)
    zp = tl.load(Z + index + 7, right, other=0)
    fm = tl.load(F + index - 7, left, other=0)
    fp = tl.load(F + index + 7, right, other=0)
    dm = tl.load(DT + row - 1, left, other=0)
    dp = tl.load(DT + row, right, other=0)
    weight = tl.load(W + col)
    # Match the eager chain rule, including both applications of the weight.
    factor = tl.load(GRAD) / ((N - 1) * 7)
    rm = ((z - zm) - (0.5 * dm) * (f + fm)) * weight
    rp = ((zp - z) - (0.5 * dp) * (fp + f)) * weight
    qm = tl.where(left, (factor * (2.0 * rm)) * weight, 0.0)
    qp = tl.where(right, (factor * (2.0 * rp)) * weight, 0.0)
    tl.store(GZ + index, qm - qp, valid)
    tl.store(GF + index, -(0.5 * dm) * qm - (0.5 * dp) * qp, valid)


@triton_op("wnt_pinn::trapezoid_backward", mutates_args={})
def _backward_op(z: torch.Tensor, f: torch.Tensor, dt: torch.Tensor,
                 weight: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gz, gf = torch.empty_like(z), torch.empty_like(f)
    wrap_triton(_backward)[(triton.cdiv(z.numel(), 256),)](
        z, f, dt, weight, grad, gz, gf, z.shape[0], 256, enable_fp_fusion=False)
    return gz, gf


@triton_op("wnt_pinn::trapezoid_loss", mutates_args={})
def _loss_op(z: torch.Tensor, f: torch.Tensor, dt: torch.Tensor,
             weight: torch.Tensor) -> torch.Tensor:
    count = (z.shape[0] - 1) * 7
    blocks = triton.cdiv(count, 256)
    parts = torch.empty((blocks,), dtype=z.dtype, device=z.device)
    wrap_triton(_forward)[(blocks,)](
        z, f, dt, weight, parts, z.shape[0], 256, enable_fp_fusion=False)
    # A second deterministic reduction avoids nondeterministic atomic adds.
    return parts.sum() / count


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _autograd_backward(ctx, grad):
    z, f, dt, weight = ctx.saved_tensors
    if torch.is_grad_enabled():
        # Preserve double-backward support with differentiable PyTorch ops.
        residual = ((z[1:] - z[:-1]) - 0.5 * dt * (f[1:] + f[:-1])) * weight
        q = (grad / residual.numel()) * (2 * residual) * weight
        zero = torch.zeros_like(z[:1])
        gz = torch.cat((zero, q)) - torch.cat((q, zero))
        v = -0.5 * dt * q
        gf = torch.cat((zero, v)) + torch.cat((v, zero))
    else:
        gz, gf = _backward_op(z, f, dt, weight, grad.contiguous())
    return gz, gf, None, None


_loss_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def trapezoidal_loss(z, f, dt, weight):
    """CUDA float32/float64 loss for fixed intervals and fixed state weights."""
    tensors = (z, f, dt, weight)
    if not z.is_cuda:
        raise ValueError("The triton physics backend requires CUDA")
    if z.dtype not in (torch.float32, torch.float64):
        raise ValueError("The triton physics backend requires float32 or float64")
    if any(t.device != z.device or t.dtype != z.dtype for t in tensors):
        raise ValueError("All integral-loss inputs must have the same device and dtype")
    if z.ndim != 2 or z.shape[1] != 7 or z.shape[0] < 2 or f.shape != z.shape:
        raise ValueError("z and f must both have shape (N, 7), N >= 2")
    if dt.shape != (z.shape[0] - 1, 1) or weight.shape != (1, 7):
        raise ValueError("dt must have shape (N-1, 1) and weight (1, 7)")
    if dt.requires_grad or weight.requires_grad:
        raise ValueError("The triton backend requires fixed collocation intervals and weights")
    return _loss_op(*(t.contiguous() for t in tensors))
