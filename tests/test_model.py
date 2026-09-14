"""Numerical parity and checkpoint compatibility for the shared model."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wnt_pinn.model import INITIAL_STATE, REGIMES, STATE_ORDER, numpy_rhs, parameters
from wnt_pinn.model.parameters import UNKNOWN, conditions

FIXTURES = Path(__file__).parent / "fixtures" / "legacy_model"


def legacy(name):
    spec = importlib.util.spec_from_file_location(f"legacy_{name}", FIXTURES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_import_does_not_import_torch():
    code = "import sys; import wnt_pinn.model; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_state_and_parameter_definitions():
    assert STATE_ORDER == ("b", "p", "h5", "h13", "m", "r", "c")
    assert len(UNKNOWN) == 36
    assert len(conditions()) == 10
    p = parameters()
    assert [p[k] for k in ("rho5", "rhoB", "rho13")] == [1.1, 1.1, 1.3]
    p["W"] = 100
    assert parameters()["W"] == 0.8
    with pytest.raises(ValueError):
        parameters(overrides={"typo": 1})


@pytest.mark.parametrize("regime", REGIMES)
def test_rhs_matches_legacy_for_treatment_and_perturbations(regime):
    old = legacy("numpy_rhs")
    for condition in conditions() + [{"forcing": {"kW": 0, "k13": 0, "kaM": 0.5}}]:
        p = parameters(regime, condition["forcing"])
        for time in (0, 39.9, 40, 64, 88, 88.1, 150):
            for state in (INITIAL_STATE, np.linspace(0.05, 2.0, 7), np.zeros(7)):
                np.testing.assert_array_equal(numpy_rhs(time, state, p), old._ode_rhs(time, state, p))


def test_untreated_keeps_circadian_input():
    from wnt_pinn.model.numpy_rhs import _ra_input
    p = parameters(overrides={"DR": 0})
    assert _ra_input(0, p) != _ra_input(12, p)
    assert _ra_input(40, parameters()) > _ra_input(40, p)


@pytest.mark.parametrize("backend", ["integral_rhs", "hybrid_rhs"])
def test_torch_rhs_values_and_parameter_gradients(backend):
    torch = pytest.importorskip("torch")
    new = importlib.import_module(f"wnt_pinn.model.{backend}")
    old = legacy(backend)
    times = torch.tensor([[0.], [40.], [88.], [150.]], dtype=torch.float64)
    states = torch.tensor([INITIAL_STATE] * 4, dtype=torch.float64)
    for regime in REGIMES:
        p = parameters(regime)
        p["W"] = torch.tensor(p["W"], dtype=torch.float64, requires_grad=True)
        got, expected = new.physics_rhs(times, states, p), old.physics_rhs(times, states, p)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(got.sum(), p["W"])[0]
        legacy_grad = torch.autograd.grad(expected.sum(), p["W"])[0]
        torch.testing.assert_close(actual_grad, legacy_grad, rtol=0, atol=0)
        refs = np.array([numpy_rhs(float(t), z.numpy(), parameters(regime)) for t, z in zip(times[:, 0], states)])
        np.testing.assert_allclose(got.detach().numpy(), refs, rtol=1e-14, atol=1e-14)


def test_state_network_seed_and_checkpoint_compatibility():
    torch = pytest.importorskip("torch")
    from wnt_pinn.networks import ForwardPINN
    old = legacy("networks")
    with torch.random.fork_rng():
        torch.manual_seed(43)
        expected = old.ForwardPINN(150, width=12, depth=2).double()
        torch.manual_seed(43)
        actual = ForwardPINN(150, width=12, depth=2).double()
        assert actual.state_dict().keys() == expected.state_dict().keys()
        for key in actual.state_dict():
            torch.testing.assert_close(actual.state_dict()[key], expected.state_dict()[key], rtol=0, atol=0)
        actual.load_state_dict(expected.state_dict(), strict=True)
        times = torch.linspace(0, 150, 23, dtype=torch.float64).reshape(-1, 1)
        torch.testing.assert_close(actual(times), expected(times), rtol=0, atol=0)


def test_shared_import_preserves_torch_runtime():
    pytest.importorskip("torch")
    code = '''import json, torch
before = (torch.get_num_threads(), str(torch.get_default_dtype()))
import wnt_pinn.model.hybrid_rhs
import wnt_pinn.networks
assert before == (torch.get_num_threads(), str(torch.get_default_dtype()))
print(json.dumps(before))
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert len(json.loads(result.stdout)) == 2


def test_fisher_definitions_match_historical_excitation_protocol():
    from wnt_pinn.model.parameters import BASELINE, CONDITIONS, UNKNOWN
    old = json.loads((FIXTURES / "excite_parameters.json").read_text())
    assert {k: BASELINE[k] for k in old["BASELINE"]} == old["BASELINE"]
    assert CONDITIONS == old["CONDITIONS"]
    assert list(UNKNOWN) == old["UNKNOWN"]
