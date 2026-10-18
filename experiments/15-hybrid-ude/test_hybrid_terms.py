"""Focused regression tests for learned mechanistic RHS terms."""
import unittest

import numpy as np
import torch
import torch.nn as nn

from config import BASELINE, HYBRID_TERMS
from hybrid import (APCMutationNN, MechanisticNN, ShapeConstrainedNN,
                    build_one_term, true_term)
from odes import _ode_rhs
from residual import physics_rhs


class ConstantTerm(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        # every mechanism net returns one column, whatever its input width
        return torch.zeros_like(x[:, :1]) + self.value


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

    # ---- the expanded registry: every edge, and the sc parameterisation -----

    def test_mechanistic_rhs_still_matches_the_scipy_model_exactly(self):
        """The refactor that exposed the modulator/ratio terms must not have
        changed the closed-form model by even one ulp -- every hybrid delta is
        measured against a control run of exactly this RHS."""
        for regime in ({"W": 0.80, "thetaP": 1.00},
                       {"W": 2.00, "thetaP": 0.25}):
            p = {**self.params, **regime}
            got = physics_rhs(self.t, self.z, p).numpy()
            for row in range(self.z.shape[0]):
                want = np.asarray(_ode_rhs(float(self.t[row, 0]),
                                           self.z[row].numpy(), p))
                np.testing.assert_allclose(got[row], want, rtol=0, atol=1e-15)

    def test_every_registered_term_is_wired_into_the_rhs(self):
        """A term in the registry that the residual ignores would silently
        train a network that changes nothing."""
        control = physics_rhs(self.t, self.z, self.params)
        eq_row = {"db": 0, "dapc": 1, "dh5": 2, "dh13": 3,
                  "dm": 4, "dr": 5, "dc": 6}
        for name, spec in HYBRID_TERMS.items():
            with self.subTest(term=name):
                n_in = len(spec["inputs"])
                hybrid = physics_rhs(
                    self.t, self.z, self.params,
                    {name: ConstantTerm(0.37 if n_in == 1 else 0.37)})
                row = eq_row[spec["eq"]]
                self.assertFalse(
                    torch.allclose(hybrid[:, row], control[:, row]),
                    f"{name} does not move its own equation {spec['eq']}")
                others = [i for i in range(7) if i != row]
                torch.testing.assert_close(hybrid[:, others],
                                           control[:, others])

    def test_shape_constrained_net_is_anchored_nonneg_and_monotone(self):
        for seed in (0, 3, 11):
            torch.manual_seed(seed)
            net = ShapeConstrainedNN(n_in=1, x_scale=1.5, signs=(1,),
                                     anchor=0.0)
            x = torch.linspace(0.0, 1.5, 201).reshape(-1, 1)
            y = net(x)
            self.assertEqual(float(y[0, 0].detach()), 0.0)
            self.assertGreaterEqual(float(y.min().detach()), 0.0)
            self.assertGreaterEqual(
                float(torch.diff(y[:, 0]).min().detach()), -1e-12)

    def test_shape_constrained_bound_is_respected(self):
        torch.manual_seed(5)
        net = ShapeConstrainedNN(n_in=1, x_scale=1.0, signs=(1,), anchor=0.0,
                                 u_max=1.35)
        y = net(torch.linspace(0.0, 50.0, 500).reshape(-1, 1))
        self.assertLessEqual(float(y.max().detach()), 1.35 + 1e-12)

    def test_multivariate_ratio_net_has_the_right_monotone_directions(self):
        torch.manual_seed(2)
        spec = HYBRID_TERMS["apc_prod"]
        net = ShapeConstrainedNN(n_in=3, x_scale=[1.0, 2.0, 1.0],
                                 signs=spec["mono"], anchor=None)
        base = torch.tensor([[0.5, 0.4, 0.3]])
        for col, sign in enumerate(spec["mono"]):
            lo = base.clone()
            hi = base.clone()
            hi[0, col] += 0.4
            delta = float((net(hi) - net(lo))[0, 0].detach())
            self.assertGreaterEqual(delta * sign, -1e-12,
                                    f"input {col} moves the wrong way")

    def test_true_term_matches_the_closed_form_it_replaces(self):
        """Scoring is meaningless if `true_term` is not the expression the
        residual actually drops."""
        p = self.params
        z = self.z
        cases = {
            "m_h5":  (z[:, 4], p["etaM"]*z[:, 4]/(p["kappaM"]+z[:, 4])),
            "h5_b":  (z[:, 0], p["lambda5"]*z[:, 0]/(p["kappa5"]+z[:, 0])),
            "apc_b": (z[:, 1], p["lambdaP"]*z[:, 1]),
            "c_ra":  (z[:, 6], p["lambdaC"]*z[:, 6]),
            "rc_cyp": (z[:, 5], p["etaRC"]*z[:, 5]/(p["kappaRC"]+z[:, 5])),
            "h13_b": (z[:, 3], p["eta13"]*z[:, 3]**2/(p["kappa13"]**2
                                                      + z[:, 3]**2)),
        }
        for name, (x, want) in cases.items():
            with self.subTest(term=name):
                np.testing.assert_allclose(
                    true_term(name, x.numpy(), p), want.numpy(), rtol=1e-12)
        ratio = true_term("apc_prod", z[:, [2, 0, 3]].numpy(), p)
        want = ((1 + p["rho5"]*z[:, 2]) /
                (1 + p["rhoB"]*z[:, 0] + p["rho13"]*z[:, 3])).numpy()
        np.testing.assert_allclose(ratio, want, rtol=1e-12)

    def test_build_one_term_handles_every_registry_entry(self):
        refs = {"ctrl": (np.linspace(0, 1, 50),
                         np.abs(np.random.default_rng(0).normal(
                             0.8, 0.2, size=(50, 7))))}
        for name in HYBRID_TERMS:
            for param in ("sc", "sc_bounded"):
                with self.subTest(term=name, param=param):
                    net = build_one_term(name, refs, torch.device("cpu"),
                                         param=param)
                    n_in = len(HYBRID_TERMS[name]["inputs"])
                    x = torch.full((4, n_in), 0.3)
                    self.assertEqual(net(x).shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
