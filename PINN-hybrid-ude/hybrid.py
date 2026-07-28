"""Neural-mechanistic hybrid (UDE) terms: a closed-form regulatory relationship
in the ODE right-hand side is REPLACED by a small neural network.

Design follows the npj UDE review (2025) and Giampiccolo et al. 2024 (HNODE):

  * SMALL nets.  The review's grid search found 1-2 hidden layers x 5-10 units
    (tanh) optimal and explicitly "no advantage to larger networks".  Default
    here is 2 x 5, tanh, Glorot-uniform, final bias zero.
  * OUTPUT CONSTRAINED to the biology.  Every term we replace is a saturating
    ACTIVATION: non-negative, zero at zero regulator, bounded above.  The review
    found biologically-informed output constraints to be what rescued
    interpretability, and Giampiccolo showed the obvious alternative (penalising
    NN magnitude) FAILS to restore identifiability.
  * The anchor f(0)=0 is not cosmetic.  In `dh5 ~ a5 + f(r) - h5 - ...` a
    constant offset moves freely between `a5` and `f`, so they are identifiable
    only up to that constant -- exactly the Lotka-Volterra `alpha` compensation
    Giampiccolo diagnosed.  The gate kills that degeneracy structurally.
    `constraint="none"` is the ablation that should re-open it.

The registry `TERMS` is shared with config.py (which owns the params-removed
bookkeeping) so adding a new learnable term is a one-line change in both.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BASELINE, HYBRID_TERMS

# state vector order: [b, apc, h5, h13, m, r, c]
VAR_INDEX = {"b": 0, "apc": 1, "h5": 2, "h13": 3, "m": 4, "r": 5, "c": 6}


def _hill_np(x, K, n=1):
    x = np.maximum(x, 0.0)
    return x**n / (K**n + x**n)


def true_term(term, x, p=None):
    """The closed form the network replaces -- GROUND TRUTH for scoring.

    Only meaningful because our data are synthetic from a known model; this is
    what makes `functional identifiability` (arXiv:2510.14140) measurable at
    all.  `x` is a numpy array of the regulator state.
    """
    p = dict(BASELINE) if p is None else p
    if term == "ra_h5":
        return p["etaR"] * _hill_np(x, p["kappaR"], 1)
    if term == "bm_myc":
        return p["etaBM"] * _hill_np(x, p["kappaBM"], p["nB"])
    if term == "b_h13":
        return p["etaB13"] * _hill_np(x, p["kappaB13"], p["nB"])
    if term == "bc_cyp":
        return p["etaBC"] * _hill_np(x, p["kappaBC"], p["nB"])
    if term == "apc_mutation":
        # x is APC functional loss, 1-thetaP. This is the EXCESS over the
        # healthy basal degradation coefficient 1, not the full deltaP.
        return p["deltaP1"] * np.maximum(x, 0.0)
    raise KeyError(term)


class MechanisticNN(nn.Module):
    """A small MLP standing in for one saturating activation term f(x).

    forward:  f(x) = softplus(g(x/x_scale)) * gate(x)
              gate(x) = 1 - exp(-x/x0)      ("anchored")   -> f(0) = 0 exactly
                      = 1                   ("none")       -> free constant

    softplus enforces non-negativity (an activation term never removes mass);
    the gate enforces the zero-at-zero anchor.  Shape between those endpoints is
    entirely free -- the net is not told the relationship is a Hill function.
    """

    def __init__(self, n_in=1, width=5, depth=2, x_scale=1.0,
                 constraint="anchored", act="tanh"):
        super().__init__()
        self.constraint = constraint
        self.register_buffer("x_scale", torch.as_tensor(
            float(x_scale), dtype=torch.get_default_dtype()))
        # x0 sets how fast the gate rises. NOTE (measured): the anchor only
        # binds where the DATA approach x=0. For ra_h5 the observed r range is
        # [0.14, 1.17] with x0=0.117, so the gate runs 0.70->1.00 and is never
        # near zero anywhere a trajectory visits -- f(x)=softplus(g)*gate can
        # still absorb a constant over that support, so the a5 degeneracy is
        # NOT structurally closed there. For the b-driven terms (b_min ~0.04)
        # the gate reaches ~0.22 and does bind. `gate_lo()` reports this so a
        # writeup cannot silently over-claim the constraint.
        self.register_buffer("x0", torch.as_tensor(
            0.1 * float(x_scale), dtype=torch.get_default_dtype()))

        Act = {"tanh": nn.Tanh, "gelu": nn.GELU}[act]
        layers = [nn.Linear(n_in, width), Act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), Act()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

        # Glorot uniform, final-layer bias zero (npj UDE review).
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for m in linears:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        xp = torch.clamp(x, min=0.0)
        f = F.softplus(self.net(xp / self.x_scale))
        if self.constraint == "anchored":
            f = f * (1.0 - torch.exp(-xp / self.x0))
        return f

    def gate_lo(self, x_lo):
        """Gate value at the low end of the OBSERVED range: how strongly the
        f(0)=0 anchor actually constrains the fit. ~1.0 => no constraint where
        the data live; <0.5 => genuinely binding. 1.0 when unconstrained."""
        if self.constraint != "anchored":
            return 1.0
        return float(1.0 - np.exp(-max(x_lo, 0.0) / float(self.x0)))

    def l2(self):
        """MEAN squared weight (biases excluded) -- the review's lambda term.

        Added explicitly to the loss rather than via optimizer weight_decay so
        the identical penalty applies under Adam, L-BFGS and the Stage-3
        frozen-net refine (torch's L-BFGS has no weight_decay argument).

        MEAN, not sum, so lambda keeps its meaning when width/depth change --
        otherwise every architecture ablation silently re-tunes the penalty.
        NOTE ON SCALE: this repo's loss is not the review's normalised MLE
        objective (our physics residual runs ~1e-4, and ~1e-5 in the Stage-3
        refine where the term matters most), so the review's lambda in [0.1, 10]
        would dominate by orders of magnitude and collapse f_NN to zero -- which
        would look like "the hybrid failed" but is really a units error. Our
        lambda is calibrated against OUR residual scale; see HYBRID_WD.
        """
        ws = [m.weight for m in self.net if isinstance(m, nn.Linear)]
        n = sum(w.numel() for w in ws)
        return sum((w**2).sum() for w in ws) / n

    @torch.no_grad()
    def curve(self, x_np, device=None):
        """Evaluate on a numpy grid -> numpy, for scoring/plotting."""
        t = torch.as_tensor(np.asarray(x_np, dtype=float).reshape(-1, 1),
                            dtype=torch.get_default_dtype(),
                            device=device or self.x_scale.device)
        return self(t).cpu().numpy().ravel()


class MonotoneLinear(nn.Module):
    """Linear layer with non-negative effective weights."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(
            out_features, in_features, dtype=torch.get_default_dtype()))
        self.bias = nn.Parameter(torch.empty(
            out_features, dtype=torch.get_default_dtype()))
        nn.init.normal_(self.raw_weight, mean=-2.0, std=0.2)
        nn.init.normal_(self.bias, mean=0.0, std=0.05)
        self.in_features = in_features

    @property
    def weight(self):
        return F.softplus(self.raw_weight) / max(self.in_features, 1)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class APCMutationNN(nn.Module):
    """Monotone APC-loss -> excess-degradation relationship.

    For u = 1-thetaP in [0,1],

        f_APC(u) = u * softplus(g_monotone(u)).

    Thus f_APC(0)=0 exactly, f_APC(u)>=0, and f_APC is non-decreasing. The
    residual keeps the healthy baseline and APC state dependence explicit as
    deltaP = 1 + f_APC(u), followed by deltaP * apc.
    """

    def __init__(self, n_in=1, width=5, depth=2, **_ignored):
        super().__init__()
        layers = [MonotoneLinear(n_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [MonotoneLinear(width, width), nn.Tanh()]
        layers.append(MonotoneLinear(width, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("x_scale", torch.ones(
            (), dtype=torch.get_default_dtype()))
        self.register_buffer("x0", torch.ones(
            (), dtype=torch.get_default_dtype()))
        self.constraint = "anchored_monotone"

    def forward(self, loss):
        u = torch.clamp(loss, min=0.0, max=1.0)
        rate = F.softplus(self.net(u))
        return u * rate

    def gate_lo(self, _x_lo):
        return 0.0

    def l2(self):
        ws = [m.weight for m in self.net if isinstance(m, MonotoneLinear)]
        n = sum(w.numel() for w in ws)
        return sum((w**2).sum() for w in ws) / max(n, 1)

    @torch.no_grad()
    def curve(self, x_np, device=None):
        t = torch.as_tensor(np.asarray(x_np, dtype=float).reshape(-1, 1),
                            dtype=torch.get_default_dtype(),
                            device=device or self.x_scale.device)
        return self(t).cpu().numpy().ravel()


def build_terms(term, refs_for_regime, device, *, width=5, depth=2,
                constraint="anchored", act="tanh"):
    """Instantiate the learnable term(s) for a run.

    ONE network shared across every experimental condition -- it is part of the
    MECHANISM, exactly like the shared `InverseParams`, not a per-condition
    state net.  `x_scale` is the largest value the regulator reaches across all
    conditions, so the net always sees an order-1 input.
    """
    if not term:
        return {}
    spec = HYBRID_TERMS[term]
    if spec.get("input_kind") == "parameter":
        net = APCMutationNN(n_in=1, width=width, depth=depth).to(device)
        return {term: net}
    idx = VAR_INDEX[spec["inputs"][0]]
    x_max = max(float(np.abs(y_ref[:, idx]).max())
                for (_t, y_ref) in refs_for_regime.values())
    net = MechanisticNN(n_in=1, width=width, depth=depth,
                        x_scale=max(x_max, 1e-3), constraint=constraint,
                        act=act).to(device)
    return {term: net}


def observed_range(term, refs_for_regime):
    """(lo, hi) span of the regulator over all conditions.

    Scoring the learned curve OUTSIDE this range is not a result -- the data
    never constrained it there.
    """
    spec = HYBRID_TERMS[term]
    if spec.get("input_kind") == "parameter":
        # Cross-severity calibration uses thetaP in [0.25, 1].
        return 0.0, 0.75
    idx = VAR_INDEX[spec["inputs"][0]]
    lo = min(float(y_ref[:, idx].min()) for (_t, y_ref) in refs_for_regime.values())
    hi = max(float(y_ref[:, idx].max()) for (_t, y_ref) in refs_for_regime.values())
    return max(lo, 0.0), hi
