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
    all.  `x` is a numpy array of the regulator state -- shape (n,) for the
    single-input terms, (n, 3) for the multivariate `apc_prod` ratio.
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
    if term == "rc_cyp":
        return p["etaRC"] * _hill_np(x, p["kappaRC"], 1)
    if term == "m_h13":
        return p["etaM13"] * _hill_np(x, p["kappaM13"], p["nM"])
    if term == "h13_b":
        return p["eta13"] * _hill_np(x, p["kappa13"], p["nH"])
    # --- modulators inside a bilinear product: f(u) multiplies a second state
    if term == "m_h5":          # etaM * m/(kappaM+m), multiplies h5
        return p["etaM"] * _hill_np(x, p["kappaM"], 1)
    if term == "h5_b":          # lambda5 * b/(kappa5+b), multiplies h5
        return p["lambda5"] * _hill_np(x, p["kappa5"], 1)
    if term == "apc_b":         # lambdaP * apc, multiplies b (mass action)
        return p["lambdaP"] * np.maximum(x, 0.0)
    if term == "c_ra":          # lambdaC * c, multiplies r (mass action)
        return p["lambdaC"] * np.maximum(x, 0.0)
    if term == "apc_prod":      # multivariate saturating RATIO, x is (n, 3)
        x = np.atleast_2d(np.asarray(x, dtype=float))
        h5, b, h13 = x[:, 0], x[:, 1], x[:, 2]
        return ((1.0 + p["rho5"]*h5)
                / (1.0 + p["rhoB"]*b + p["rho13"]*h13))
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

    def __init__(self, in_features, out_features, raw_mean=-2.0, raw_std=0.2):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(
            out_features, in_features, dtype=torch.get_default_dtype()))
        self.bias = nn.Parameter(torch.empty(
            out_features, dtype=torch.get_default_dtype()))
        nn.init.normal_(self.raw_weight, mean=raw_mean, std=raw_std)
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


class ShapeConstrainedNN(nn.Module):
    """Monotone + exactly-anchored (+ optionally bounded) mechanism network.

    This is the `sc` parameterisation, and the reason it exists is a measured
    failure of the `gated` one (MechanisticNN above): the soft gate
    1-exp(-x/x0) only pins f(0)=0 AT x=0, so wherever the trajectories keep the
    regulator away from zero the network can still absorb a constant out of the
    equation's basal-production parameter. That is exactly the damage the
    bm_myc run recorded (aM error 32-52%, dm equation -4 recovered params).

    Loman & Baker (arXiv:2510.14140, Fig. 6) tested precisely this on chemical
    reaction networks: a UDE whose network is constrained MONOTONE + BOUNDED
    "achieves parameter identifiability on par with the fully known model",
    while non-negativity alone (their default, and ours until now) loses a
    lot. They encode monotonicity by making every weight one-signed and every
    activation monotone. We do the same, and add an EXACT anchor, which our
    terms admit for free (no regulator => no activation).

    Construction, for x >= 0 and per-input monotone signs s_i in {+1, -1}:

        h(x)  = MLP_{W>=0, tanh}( s * x / x_scale )        monotone in each x_i
        f(x)  = softplus(h(x)) - softplus(h(x_a))          unbounded
              = u_max * (sig(h(x)) - sig(h(x_a))) / (1 - sig(h(x_a)))  bounded

    with x_a the anchor point (0 for every activation term). Because h is
    monotone increasing along +s and the anchor is the domain minimum along
    that direction, f >= 0 holds automatically -- non-negativity, monotonicity
    and f(x_a)=0 are all structural, none is a penalty.

    `signs=None` (used by the multivariate apc_prod ratio, which is increasing
    in h5 and decreasing in b/h13 and equals 1 at the origin rather than 0)
    keeps monotonicity and non-negativity but drops the anchor.
    """

    def __init__(self, n_in=1, width=5, depth=2, x_scale=1.0, signs=(1,),
                 anchor=0.0, u_max=None, act="tanh"):
        super().__init__()
        self.constraint = ("sc_bounded" if u_max is not None else "sc")
        self.n_in = n_in
        xs = np.atleast_1d(np.asarray(x_scale, dtype=float))
        if xs.size == 1:
            xs = np.repeat(xs, n_in)
        self.register_buffer("x_scale", torch.as_tensor(
            np.maximum(xs, 1e-3), dtype=torch.get_default_dtype()).reshape(1, -1))
        self.register_buffer("signs", torch.as_tensor(
            np.asarray(signs, dtype=float).reshape(1, -1),
            dtype=torch.get_default_dtype()))
        self.anchor = None if anchor is None else float(anchor)
        if self.anchor is not None:
            self.register_buffer("x_anchor", torch.full(
                (1, n_in), self.anchor, dtype=torch.get_default_dtype()))
        self.u_max = None if u_max is None else float(u_max)

        Act = {"tanh": nn.Tanh, "softplus": nn.Softplus}[act]
        layers = [MonotoneLinear(n_in, width, raw_mean=-0.5, raw_std=0.3),
                  Act()]
        for _ in range(depth - 1):
            layers += [MonotoneLinear(width, width, raw_mean=-0.5, raw_std=0.3),
                       Act()]
        layers.append(MonotoneLinear(width, 1, raw_mean=-0.5, raw_std=0.3))
        self.net = nn.Sequential(*layers)

    def _h(self, x):
        return self.net(self.signs * (x / self.x_scale))

    def forward(self, x):
        x = torch.clamp(x, min=0.0)
        if x.shape[-1] != self.n_in:              # single-input convenience
            x = x.reshape(-1, self.n_in)
        h = self._h(x)
        if self.anchor is None:
            return (F.softplus(h) if self.u_max is None
                    else self.u_max * torch.sigmoid(h))
        ha = self._h(self.x_anchor.to(x.dtype))
        if self.u_max is None:
            return F.softplus(h) - F.softplus(ha)
        sa = torch.sigmoid(ha)
        return self.u_max * (torch.sigmoid(h) - sa) / torch.clamp(1.0 - sa,
                                                                  min=1e-3)

    def gate_lo(self, _x_lo):
        """0.0 => the f(anchor)=0 constraint is EXACT (it always is here)."""
        return 0.0 if self.anchor is not None else 1.0

    def l2(self):
        ws = [m.weight for m in self.net if isinstance(m, MonotoneLinear)]
        n = sum(w.numel() for w in ws)
        return sum((w**2).sum() for w in ws) / max(n, 1)

    @torch.no_grad()
    def curve(self, x_np, device=None):
        a = np.asarray(x_np, dtype=float)
        a = a.reshape(-1, self.n_in)
        t = torch.as_tensor(a, dtype=torch.get_default_dtype(),
                            device=device or self.x_scale.device)
        return self(t).cpu().numpy().ravel()


class AnchoredRateNN(nn.Module):
    """Linear-anchored mechanism network:  f(x) = (x / x_scale) * rate(x).

    This is the generalisation of the one construction in this repo that cost
    the mechanism NOTHING. `APCMutationNN` (used for the calibrated APC term,
    0.1% functional NRMSE and +1 parameters versus control) multiplies a
    non-negative rate by the input itself; `MechanisticNN` multiplies it by a
    soft gate 1-exp(-x/x0) with x0 = 0.1*x_max. Both give f(0)=0, so both were
    described as "anchored" -- but they bound f very differently just above 0:

        exponential gate:  f(x) <~ (x / x0)      * rate = (10 x / x_max) * rate
        linear anchor:     f(x) <=  (x / x_max)  * rate

    a factor of TEN. That gap is the whole compensation story. In dm, at the
    lowest beta-catenin the trajectories reach (b=0.036, x_max=1.43), the gate
    permits f(0.036) ~ 0.31*rate while the truth is 0.007 -- ample room to
    absorb most of the basal production aM (true value 0.18), which is exactly
    the 32-52% aM error the bm_myc run recorded. The linear anchor permits only
    0.025*rate at the same point.

    `mono=True` additionally makes the rate monotone via non-negative weights,
    reproducing APCMutationNN exactly for a linear ground truth while still
    allowing saturation (a decreasing rate is what makes a Hill saturate, so
    monotone-rate is a strictly stronger assumption -- it is the ablation, not
    the default).
    """

    def __init__(self, n_in=1, width=5, depth=2, x_scale=1.0, mono=False,
                 act="tanh"):
        super().__init__()
        self.constraint = "linear_anchored" + ("_monotone" if mono else "")
        self.n_in = n_in
        self.mono = mono
        xs = np.atleast_1d(np.asarray(x_scale, dtype=float))
        self.register_buffer("x_scale", torch.as_tensor(
            np.maximum(xs, 1e-3), dtype=torch.get_default_dtype()
        ).reshape(1, -1))

        Act = {"tanh": nn.Tanh, "gelu": nn.GELU}[act]
        Lin = (lambda i, o: MonotoneLinear(i, o, raw_mean=-0.5, raw_std=0.3)) \
            if mono else nn.Linear
        layers = [Lin(n_in, width), Act()]
        for _ in range(depth - 1):
            layers += [Lin(width, width), Act()]
        layers.append(Lin(width, 1))
        self.net = nn.Sequential(*layers)
        if not mono:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.clamp(x, min=0.0)
        if x.shape[-1] != self.n_in:
            x = x.reshape(-1, self.n_in)
        u = x / self.x_scale
        return u * F.softplus(self.net(u))

    def gate_lo(self, x_lo):
        """The anchor's strength at the low end of the observed range: the
        prefactor that multiplies the rate there. Directly comparable to
        MechanisticNN.gate_lo, which reports 1-exp(-x_lo/x0) for the same x_lo,
        so the two constructions can be put in one column."""
        return float(max(x_lo, 0.0) / float(self.x_scale[0, 0]))

    def l2(self):
        ws = [m.weight for m in self.net
              if isinstance(m, (nn.Linear, MonotoneLinear))]
        n = sum(w.numel() for w in ws)
        return sum((w**2).sum() for w in ws) / max(n, 1)

    @torch.no_grad()
    def curve(self, x_np, device=None):
        a = np.asarray(x_np, dtype=float).reshape(-1, self.n_in)
        t = torch.as_tensor(a, dtype=torch.get_default_dtype(),
                            device=device or self.x_scale.device)
        return self(t).cpu().numpy().ravel()


def _regulator_scale(spec, refs_for_regime):
    """Largest value each regulator input reaches across all conditions."""
    out = []
    for v in spec["inputs"]:
        idx = VAR_INDEX[v]
        out.append(max(float(np.abs(y_ref[:, idx]).max())
                       for (_t, y_ref) in refs_for_regime.values()))
    return np.maximum(np.asarray(out, dtype=float), 1e-3)


def supports(term, param):
    """Can `term` be built under parameterisation `param`? (reason if not)

    Not every constraint set applies to every registry entry: `gated` is a
    single-input construction, and the linear-anchored ones need an anchor to
    be linear about. A sweep over terms x parameterisations must skip the
    invalid cells rather than die on one -- the 2026-08-01 edge atlas lost
    three of its four regimes to an unguarded `apc_prod`/`gated` pair after
    2.8 GPU-hours.
    """
    spec = HYBRID_TERMS[term]
    if spec.get("input_kind") == "parameter":
        return True, ""
    if param == "gated" and len(spec["inputs"]) > 1:
        return False, (f"{term} has {len(spec['inputs'])} inputs and `gated` "
                       f"is single-input only")
    if param in ("lin", "lin_mono") and spec.get("anchor") is None:
        return False, f"{term} has no zero anchor for `{param}` to be linear about"
    return True, ""


def build_one_term(term, refs_for_regime, device, *, width=5, depth=2,
                   constraint="anchored", act="tanh", param="gated"):
    """Instantiate the network for a single term under one parameterisation.

    `param` selects the constraint SET, which is the experimental variable of
    the shape-constraint study:
       gated       -> MechanisticNN: non-negative + soft f(0)=0 gate  (baseline)
       sc          -> monotone + EXACT anchor + non-negative
       sc_bounded  -> the above, plus the ORACLE saturation bound
       lin         -> non-negative + LINEAR anchor f(x) = (x/x_max) * rate(x)
       lin_mono    -> the above with a monotone rate (== APCMutationNN)
    """
    spec = HYBRID_TERMS[term]
    if spec.get("input_kind") == "parameter" and param in ("gated",
                                                           "lin_mono"):
        return APCMutationNN(n_in=1, width=width, depth=depth).to(device)

    n_in = len(spec["inputs"])
    if spec.get("input_kind") == "parameter":
        x_scale = np.array([1.0])
    else:
        x_scale = _regulator_scale(spec, refs_for_regime)

    if param == "gated":
        if n_in > 1:
            raise ValueError(f"{term}: the gated parameterisation is "
                             f"single-input only; use param=sc")
        return MechanisticNN(n_in=1, width=width, depth=depth,
                             x_scale=float(x_scale[0]), constraint=constraint,
                             act=act).to(device)

    if param in ("lin", "lin_mono"):
        if spec.get("anchor") is None:
            raise ValueError(f"{term}: no zero anchor, so the linear-anchored "
                             f"parameterisation does not apply; use sc")
        return AnchoredRateNN(n_in=n_in, width=width, depth=depth,
                              x_scale=x_scale, mono=(param == "lin_mono"),
                              act=("tanh" if act == "gelu" else act)
                              ).to(device)

    u_max = spec.get("u_max") if param == "sc_bounded" else None
    return ShapeConstrainedNN(
        n_in=n_in, width=width, depth=depth, x_scale=x_scale,
        signs=spec.get("mono", (1,)*n_in), anchor=spec.get("anchor", 0.0),
        u_max=u_max, act=("tanh" if act == "gelu" else act)).to(device)


def build_terms(term, refs_for_regime, device, *, width=5, depth=2,
                constraint="anchored", act="tanh", param="gated"):
    """Instantiate the learnable term(s) for a run.

    ONE network per term, shared across every experimental condition -- each is
    part of the MECHANISM, exactly like the shared `InverseParams`, not a
    per-condition state net.  `x_scale` is the largest value the regulator
    reaches across all conditions, so the net always sees an order-1 input.

    `term` may be a single name, or a list/tuple of names for the MULTI-TERM
    hybrid (several regulatory edges handed to networks at once). Loman & Baker
    found that generalising one function to a network "has little effect on the
    identifiability of the other function"; our system is far sloppier
    (cond(FIM) 1e7-1e17), so that is a claim worth testing here rather than
    assuming.
    """
    if not term:
        return {}
    names = [term] if isinstance(term, str) else list(term)
    return {t: build_one_term(t, refs_for_regime, device, width=width,
                              depth=depth, constraint=constraint, act=act,
                              param=param)
            for t in names}


def observed_range(term, refs_for_regime):
    """(lo, hi) span of the regulator over all conditions.

    Scoring the learned curve OUTSIDE this range is not a result -- the data
    never constrained it there. For a multi-input term this is the span of the
    FIRST input only; use `observed_points` to score those honestly.
    """
    spec = HYBRID_TERMS[term]
    if spec.get("input_kind") == "parameter":
        # Cross-severity calibration uses thetaP in [0.25, 1].
        return 0.0, 0.75
    idx = VAR_INDEX[spec["inputs"][0]]
    lo = min(float(y_ref[:, idx].min()) for (_t, y_ref) in refs_for_regime.values())
    hi = max(float(y_ref[:, idx].max()) for (_t, y_ref) in refs_for_regime.values())
    return max(lo, 0.0), hi


def observed_points(term, refs_for_regime, max_points=4000):
    """The regulator values the trajectories ACTUALLY visit, shape (n, n_in).

    For multi-input terms a rectangular grid would score the network in
    combinations the system never reaches; the honest support is the visited
    set itself.
    """
    spec = HYBRID_TERMS[term]
    idx = [VAR_INDEX[v] for v in spec["inputs"]]
    rows = [y_ref[:, idx] for (_t, y_ref) in refs_for_regime.values()]
    x = np.concatenate(rows, axis=0)
    if len(x) > max_points:
        x = x[:: max(1, len(x) // max_points)]
    return np.maximum(x, 0.0)
