"""Numerical and autograd contracts for optional integral-loss acceleration."""

import pytest

from wnt_pinn.model.parameters import REGIMES, UNKNOWN, parameters
from wnt_pinn.runs.config import validate_config

torch = pytest.importorskip("torch")

from wnt_pinn.kernels.integral import make_integral_loss, trapezoidal_loss  # noqa: E402
from wnt_pinn.model.hybrid_rhs import physics_rhs  # noqa: E402


def cuda_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    if not hasattr(torch.library, "triton_op"):
        pytest.skip("Custom Triton operations require torch >= 2.6")
    from wnt_pinn.kernels.trapezoid_triton import trapezoidal_loss as kernel
    return kernel


def test_backend_validation_and_default():
    assert validate_config({})["training"]["physics_backend"] == "eager"
    for backend in ("typo", "triton"):
        with pytest.raises(ValueError):
            validate_config({"training": {"physics_backend": backend}})
    with pytest.raises(ValueError):
        make_integral_loss(physics_rhs, "typo")


@pytest.mark.parametrize("n", [2, 37, 1025, 8000, 40001])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_triton_loss_and_backward(n, dtype):
    kernel = cuda_kernel()
    generator = torch.Generator(device="cuda").manual_seed(1042)
    # Deliberately strided inputs and nonuniform intervals exercise masking,
    # both endpoints, contiguous copies and non-power-of-two row counts.
    z = torch.randn((n, 14), generator=generator, device="cuda", dtype=dtype)[:, ::2]
    f = torch.randn((n, 14), generator=generator, device="cuda", dtype=dtype)[:, ::2]
    z.requires_grad_()
    f.requires_grad_()
    dt = torch.rand((n - 1, 1), generator=generator, device="cuda", dtype=dtype) + 0.01
    weight = torch.tensor([[0.05, 1, 20, 3, 0.4, 7, 2]], device="cuda", dtype=dtype)
    expected = trapezoidal_loss(z, f, dt, weight)
    actual = kernel(z, f, dt, weight)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-13
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    multiplier = torch.tensor(-0.37, device="cuda", dtype=dtype)
    reference_grads = torch.autograd.grad(expected * multiplier, (z, f))
    actual_grads = torch.autograd.grad(actual * multiplier, (z, f))
    for got, wanted in zip(actual_grads, reference_grads):
        torch.testing.assert_close(got, wanted, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("parts_count", [1, 219, 1024, 1025, 4097])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("in_place", [False, True])
def test_final_reduction_masks_tiles_and_is_repeatable(parts_count, dtype, in_place):
    cuda_kernel()
    import triton

    from wnt_pinn.kernels.trapezoid_triton import _reduce_partials

    generator = torch.Generator(device="cuda").manual_seed(81)
    parts = torch.rand(parts_count, generator=generator, device="cuda", dtype=dtype)
    # Counts need not fill the final 256-element residual block.
    count = max(1, parts_count * 256 - 17)
    tile = min(1024, triton.next_power_of_2(parts_count))
    outputs = []
    for _ in range(3):
        working = parts.clone() if in_place else parts
        output = working[0] if in_place else torch.empty((), device="cuda", dtype=dtype)
        _reduce_partials[(1,)](working, output, parts_count, count, tile, enable_fp_fusion=False)
        outputs.append(output)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-13
    torch.testing.assert_close(outputs[0], parts.sum() / count, rtol=tolerance, atol=0)
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


def test_final_reduction_retains_tiny_float64_values():
    cuda_kernel()
    from wnt_pinn.kernels.trapezoid_triton import _reduce_partials

    parts = torch.full((1025,), 1e-290, device="cuda", dtype=torch.float64)
    output = torch.empty((), device="cuda", dtype=torch.float64)
    _reduce_partials[(1,)](parts, output, 1025, 262400, 1024, enable_fp_fusion=False)
    assert output.item() > 0
    torch.testing.assert_close(output, parts.sum() / 262400, rtol=2e-13, atol=0)


def test_triton_gradcheck_and_double_backward():
    kernel = cuda_kernel()
    z = torch.randn(4, 7, device="cuda", dtype=torch.float64, requires_grad=True)
    f = torch.randn_like(z, requires_grad=True)
    dt = torch.full((3, 1), 0.2, device="cuda", dtype=torch.float64)
    w = torch.ones((1, 7), device="cuda", dtype=torch.float64)
    def function(a, b):
        return kernel(a, b, dt, w)

    assert torch.autograd.gradcheck(function, (z, f), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (z, f), fast_mode=True)
    with pytest.raises(ValueError, match="fixed"):
        kernel(z, f, dt.requires_grad_(), w)


def test_float64_near_zero_residual_is_not_rounded_to_float32():
    kernel = cuda_kernel()
    z = (torch.arange(37, device="cuda", dtype=torch.float64)[:, None] * 0.25
         + torch.arange(7, device="cuda", dtype=torch.float64)[None, :])
    z[10, 3] += 2.0**-40
    z.requires_grad_()
    f = torch.full_like(z, 2.0, requires_grad=True)
    dt = torch.full((36, 1), 0.125, device="cuda", dtype=torch.float64)
    w = torch.full((1, 7), 1e6, device="cuda", dtype=torch.float64)
    expected, actual = trapezoidal_loss(z, f, dt, w), kernel(z, f, dt, w)
    assert expected.item() > 0
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=0)
    for a, b in zip(torch.autograd.grad(actual, (z, f)),
                    torch.autograd.grad(expected, (z, f))):
        torch.testing.assert_close(a, b, rtol=2e-13, atol=0)


@pytest.mark.parametrize("regime", list(REGIMES))
@pytest.mark.parametrize("term", [None, "bm_myc", "ra_h5", "apc_mutation"])
def test_triton_biological_and_neural_gradients(regime, term):
    cuda_kernel()
    torch.manual_seed(17)
    t = torch.linspace(0, 150, 79, device="cuda", dtype=torch.float64).reshape(-1, 1)
    z = (torch.rand(79, 7, device="cuda", dtype=torch.float64) + 0.1).requires_grad_()
    p = parameters(regime, {"DW": 0.8, "DM": 1.2, "DR": 2.0})
    for name in UNKNOWN:
        p[name] = torch.tensor(p[name], device="cuda", dtype=torch.float64, requires_grad=True)
    net = torch.nn.Sequential(torch.nn.Linear(1, 5), torch.nn.Tanh(),
                              torch.nn.Linear(5, 1), torch.nn.Softplus()).cuda().double()
    terms = {term: net} if term else {}
    def rhs(t, z, p):
        return physics_rhs(t, z, p, terms)

    inputs = [z] + [p[name] for name in UNKNOWN] + (list(net.parameters()) if term else [])
    w = torch.ones((1, 7), device="cuda", dtype=torch.float64)
    values, grads = [], []
    for backend in ("eager", "triton"):
        value = make_integral_loss(rhs, backend)(t, z, p, w)
        values.append(value)
        grads.append(torch.autograd.grad(value, inputs, allow_unused=True))
    torch.testing.assert_close(values[0], values[1], rtol=2e-13, atol=2e-13)
    for a, b in zip(*grads):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-12)


def test_compiled_physics_value_and_parameter_gradients():
    cuda_kernel()
    t = torch.linspace(0, 150, 71, device="cuda", dtype=torch.float64).reshape(-1, 1)
    z = torch.full((71, 7), 0.7, device="cuda", dtype=torch.float64, requires_grad=True)
    p = parameters()
    for name in UNKNOWN:
        p[name] = torch.tensor(p[name], device="cuda", dtype=torch.float64, requires_grad=True)
    w = torch.ones((1, 7), device="cuda", dtype=torch.float64)
    inputs = [z] + [p[name] for name in UNKNOWN]
    values, grads = [], []
    for backend in ("eager", "compiled"):
        loss = make_integral_loss(physics_rhs, backend)
        value = loss(t, z, p, w)
        values.append(value)
        grads.append(torch.autograd.grad(value, inputs))
        with torch.no_grad():
            torch.testing.assert_close(loss(t, z, p, w), values[0], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(values[0], values[1], rtol=2e-12, atol=2e-12)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-11)
