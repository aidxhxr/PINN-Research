import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardPINN(nn.Module):
    """MLP in time with a Fourier-feature input embedding.

    A plain MLP on the scalar time input suffers from spectral bias and
    cannot represent the fast circadian RA forcing (period TR=24) or the
    sharp ATRA pulse. We lift the (normalised) time to a bank of sines and
    cosines at random frequencies (Tancik et al. 2020) so those modes are
    directly representable. The raw normalised time is kept in the
    embedding for the slow trend / initial condition.
    """

    def __init__(self, T_max, n_vars=7, width=256, depth=4,
                 n_fourier=16, fourier_sigma=4.0):
        super().__init__()
        self.T_max = T_max
        self.n_fourier = n_fourier
        if n_fourier > 0:
            # fixed (non-trained) random frequencies, in cycles over [0,1]
            self.register_buffer("B", torch.randn(n_fourier) * fourier_sigma)
            in_dim = 1 + 2 * n_fourier
        else:
            in_dim = 1

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
        tn = t / self.T_max                      # (N,1) normalised to ~[0,1]
        if self.n_fourier == 0:
            return tn
        proj = 2.0 * np.pi * tn * self.B         # (N, n_fourier)
        return torch.cat([tn, torch.sin(proj), torch.cos(proj)], dim=1)

    def forward(self, t):
        return self.net(self._embed(t))


def time_derivatives(net, t):
    t = t.clone().requires_grad_(True)
    z = net(t)
    dz = torch.zeros_like(z)
    for i in range(z.shape[1]):
        dz[:, i:i+1] = torch.autograd.grad(
            z[:, i].sum(), t, create_graph=True)[0]
    return z, dz


def _inv_softplus(y):
    # numerically stable inverse of softplus, so softplus(raw) == y at init
    return np.log(np.expm1(y))


def _logit(y):
    return np.log(y / (1.0 - y))


class InverseParams(nn.Module):
    """Trainable ODE parameters recovered by the inverse PINN.

    Each unknown is stored as an unconstrained raw scalar and mapped onto
    its valid range so the optimiser never has to respect a hard bound:
        W       > 0          via softplus
        thetaP  in (0, 1)    via sigmoid
    `.dict()` returns the current estimates as differentiable tensors, ready
    to drop into the BASELINE parameter dict for the physics residual.
    """

    def __init__(self, init_guess):
        super().__init__()
        self.raw_W = nn.Parameter(
            torch.tensor(_inv_softplus(init_guess["W"])))
        self.raw_thetaP = nn.Parameter(
            torch.tensor(_logit(init_guess["thetaP"])))

    @property
    def W(self):
        return F.softplus(self.raw_W)

    @property
    def thetaP(self):
        return torch.sigmoid(self.raw_thetaP)

    def dict(self):
        return {"W": self.W, "thetaP": self.thetaP}

    def values(self):
        with torch.no_grad():
            return {"W": float(self.W), "thetaP": float(self.thetaP)}
