import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardPINN(nn.Module):
    """MLP in time with a Fourier-feature input embedding and per-variable
    OUTPUT SCALING.

    A plain MLP on the scalar time input suffers from spectral bias and
    cannot represent the fast circadian RA forcing (period TR=24) or the
    sharp ATRA pulse. We lift the (normalised) time to a bank of sines and
    cosines at random frequencies (Tancik et al. 2020) so those modes are
    directly representable. The raw normalised time is kept for the slow
    trend / initial condition.

    The 7 state variables span different magnitudes; we multiply the raw
    network output by a per-variable scale (`out_scale`) so the network
    learns order-1 quantities and the 7-component physics residual is
    balanced rather than dominated by the large-amplitude states
    (output scaling, as in systems-biology-informed NNs, Daneker et al.).
    """

    def __init__(self, T_max, n_vars=7, width=256, depth=4,
                 n_fourier=16, fourier_sigma=4.0, out_scale=None):
        super().__init__()
        self.T_max = T_max
        self.n_fourier = n_fourier
        if n_fourier > 0:
            self.register_buffer("B", torch.randn(n_fourier) * fourier_sigma)
            in_dim = 1 + 2 * n_fourier
        else:
            in_dim = 1

        if out_scale is None:
            out_scale = torch.ones(n_vars)
        self.register_buffer("out_scale", torch.as_tensor(
            out_scale, dtype=torch.get_default_dtype()).reshape(1, n_vars))

        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, n_vars))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def _embed(self, t):
        tn = t / self.T_max
        if self.n_fourier == 0:
            return tn
        proj = 2.0 * np.pi * tn * self.B
        return torch.cat([tn, torch.sin(proj), torch.cos(proj)], dim=1)

    def forward(self, t):
        return self.net(self._embed(t)) * self.out_scale


def time_derivatives(net, t):
    t = t.clone().requires_grad_(True)
    z = net(t)
    dz = torch.zeros_like(z)
    for i in range(z.shape[1]):
        dz[:, i:i+1] = torch.autograd.grad(
            z[:, i].sum(), t, create_graph=True)[0]
    return z, dz


def _logit(y):
    return np.log(y / (1.0 - y))


class InverseParams(nn.Module):
    """Trainable ODE parameters recovered by the inverse PINN.

    Conditioning matters here: the unknowns span 0.08 -> 3.5, so a single
    learning rate either crawls on the large parameters or diverges on the
    small ones. We therefore use a LOG / RELATIVE parametrisation for every
    positive parameter:

        value_k = nominal_k * exp(s_k)

    so `s_k` is an order-1, scale-invariant variable (a fractional log-offset
    from the nominal value) for every parameter regardless of magnitude, and
    the gradient w.r.t. `s_k` is the value-scaled (relative) sensitivity.
    `thetaP in (0, 1)` keeps a sigmoid map. `s_k` is initialised so the
    constrained value reproduces the requested init guess exactly.

    `.dict()` returns the current estimates as differentiable tensors, ready
    to splice into the BASELINE parameter dict for the physics residual.
    """

    def __init__(self, init_guess, param_range, nominal):
        super().__init__()
        self.param_range = dict(param_range)
        self.nominal = dict(nominal)
        self.raw = nn.ParameterDict()
        for k, (lo, hi) in self.param_range.items():
            if hi is None:
                # log/relative: value = nominal * exp(s), s = log(init/nominal)
                s0 = np.log(init_guess[k] / self.nominal[k])
            else:
                # bounded: value = lo + (hi-lo)*sigmoid(s)
                s0 = _logit((init_guess[k] - lo) / (hi - lo))
            self.raw[k] = nn.Parameter(torch.tensor(float(s0)))

    def dict(self):
        out = {}
        for k, raw in self.raw.items():
            lo, hi = self.param_range[k]
            if hi is None:
                out[k] = self.nominal[k] * torch.exp(raw)
            else:
                out[k] = lo + (hi - lo) * torch.sigmoid(raw)
        return out

    def values(self):
        with torch.no_grad():
            return {k: float(v) for k, v in self.dict().items()}
