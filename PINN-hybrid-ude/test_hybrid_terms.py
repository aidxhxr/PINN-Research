"""Focused regression tests for learned mechanistic RHS terms."""
import unittest

import torch
import torch.nn as nn

from config import BASELINE
from hybrid import APCMutationNN, MechanisticNN
from residual import physics_rhs


class ConstantTerm(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.zeros_like(x) + self.value


class HybridTermTests(unittest.TestCase):
    def setUp(self):
        self.t = torch.tensor([[0.0], [1.0], [2.0]])
        self.z = torch.tensor([
            [0.20, 1.00, 0.80, 0.30, 0.30, 0.60, 0.40],
            [0.24, 0.90, 0.75, 0.35, 0.36, 0.55, 0.45],
            [0.28, 0.82, 0.70, 0.40, 0.42, 0.50, 0.50],
        ])
        self.params = {**BASELINE, "thetaP": 0.50, "W": 1.50}

    def test_control_apc_formula_is_unchanged(self):
        got = physics_rhs(self.t, self.z, self.params)
        b, apc, h5, h13 = (self.z[:, i:i + 1] for i in range(4))
        delta_p = 1.0 + self.params["deltaP1"] * (
            1.0 - self.params["thetaP"])
        production = ((1.0 + self.params["rho5"] * h5) /
                      (1.0 + self.params["rhoB"] * b +
                       self.params["rho13"] * h13))
        expected = (production - delta_p * apc) / self.params["epsP"]
        torch.testing.assert_close(got[:, 1:2], expected)

    def test_myc_term_changes_only_myc_equation(self):
        control = physics_rhs(self.t, self.z, self.params)
        hybrid = physics_rhs(
            self.t, self.z, self.params,
            {"bm_myc": ConstantTerm(0.25)},
        )
        unchanged = [0, 1, 2, 3, 5, 6]
        torch.testing.assert_close(hybrid[:, unchanged], control[:, unchanged])
        self.assertFalse(torch.allclose(hybrid[:, 4], control[:, 4]))

    def test_apc_term_changes_only_apc_equation_and_keeps_mass_action(self):
        control = physics_rhs(self.t, self.z, self.params)
        hybrid = physics_rhs(
            self.t, self.z, self.params,
            {"apc_mutation": ConstantTerm(2.0)},
        )
        unchanged = [0, 2, 3, 4, 5, 6]
        torch.testing.assert_close(hybrid[:, unchanged], control[:, unchanged])
        b, apc, h5, h13 = (self.z[:, i:i + 1] for i in range(4))
        production = ((1.0 + self.params["rho5"] * h5) /
                      (1.0 + self.params["rhoB"] * b +
                       self.params["rho13"] * h13))
        expected = (production - (1.0 + 2.0) * apc) / self.params["epsP"]
        torch.testing.assert_close(hybrid[:, 1:2], expected)

    def test_apc_network_is_anchored_positive_monotone_and_differentiable(self):
        torch.manual_seed(7)
        net = APCMutationNN(width=5, depth=2)
        x = torch.linspace(0.0, 1.0, 101).reshape(-1, 1)
        y = net(x)
        self.assertEqual(float(y[0].detach()), 0.0)
        self.assertGreaterEqual(float(torch.min(y).detach()), 0.0)
        self.assertGreaterEqual(
            float(torch.min(torch.diff(y[:, 0])).detach()), -1e-12)
        y.sum().backward()
        self.assertTrue(all(
            parameter.grad is not None
            for parameter in net.parameters()
        ))

    def test_existing_activation_network_retains_exact_zero_anchor(self):
        torch.manual_seed(7)
        net = MechanisticNN(x_scale=1.0, constraint="anchored")
        value = net(torch.zeros(1, 1))[0, 0].detach()
        self.assertEqual(float(value), 0.0)


if __name__ == "__main__":
    unittest.main()
