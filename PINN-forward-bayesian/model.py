import numpy as np
import torch
import torch.nn as nn


class ForwardPINN(nn.Module):
    """MLP in time with a Fourier-feature input embedding and per-variable
    OUTPUT SCALING.

    A plain MLP on the scalar time input suffers from spectral bias and cannot
    represent the fast circadian RA forcing (period TR=24) or the sharp ATRA
    pulse. We lift the (normalised) time to a bank of sines/cosines at random
    frequencies (Tancik et al. 2020) so those modes are directly representable.
    The raw normalised time is kept for the slow trend / initial condition.

    The 7 states span different magnitudes; multiplying the raw output by a
    per-variable scale (`out_scale`) keeps the network learning order-1
    quantities and balances the 7-component physics residual.

    IMPORTANT (Bayesian forward PINN): this net is deliberately SMALL. Unlike
    the point-estimate forward PINN (256x4 = 207k weights), here the WEIGHTS are
    the quantity we put a posterior on and sample with HMC, so the parameter
    count must stay ~O(10^4) for the sampler to be tractable. The Fourier
    frequencies `B` are a fixed (non-sampled) buffer, so the fast forcing is
    representable without spending sampled degrees of freedom on it.
    """

    def __init__(self, T_max, n_vars=7, width=64, depth=3,
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
