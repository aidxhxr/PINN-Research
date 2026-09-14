"""Fourier state networks and biological parameter transforms."""

import math
import numpy as np
import torch
import torch.nn as nn

class Sine(nn.Module):

    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class ForwardPINN(nn.Module):

    def __init__(self, T_max, n_vars=7, width=256, depth=4, n_fourier=16, fourier_sigma=4.0, out_scale=None, activation='gelu', siren_w0=30.0):
        super().__init__()
        self.T_max = T_max
        self.n_fourier = n_fourier
        self.activation = activation
        self.siren_w0 = siren_w0
        if n_fourier > 0:
            self.register_buffer('B', torch.randn(n_fourier) * fourier_sigma)
            in_dim = 1 + 2 * n_fourier
        else:
            in_dim = 1
        if out_scale is None:
            out_scale = torch.ones(n_vars)
        self.register_buffer('out_scale', torch.as_tensor(out_scale, dtype=torch.get_default_dtype()).reshape(1, n_vars))
        act = (lambda: Sine(siren_w0)) if activation == 'siren' else nn.GELU
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers.append(nn.Linear(width, n_vars))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        if self.activation == 'siren':
            linears = [m for m in self.net if isinstance(m, nn.Linear)]
            for i, m in enumerate(linears):
                fan_in = m.weight.shape[1]
                with torch.no_grad():
                    if i == 0:
                        m.weight.uniform_(-1.0 / fan_in, 1.0 / fan_in)
                    else:
                        bnd = math.sqrt(6.0 / fan_in) / self.siren_w0
                        m.weight.uniform_(-bnd, bnd)
                    nn.init.zeros_(m.bias)
        else:
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
        dz[:, i:i + 1] = torch.autograd.grad(z[:, i].sum(), t, create_graph=True)[0]
    return (z, dz)

def _logit(y):
    return np.log(y / (1.0 - y))

class InverseParams(nn.Module):

    def __init__(self, init_guess, param_range, nominal):
        super().__init__()
        self.param_range = dict(param_range)
        self.nominal = dict(nominal)
        self.raw = nn.ParameterDict()
        for k, (lo, hi) in self.param_range.items():
            if hi is None:
                s0 = np.log(init_guess[k] / self.nominal[k])
            else:
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
