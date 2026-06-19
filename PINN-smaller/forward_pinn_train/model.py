import torch
import torch.nn as nn


class ForwardPINN(nn.Module):
    def __init__(self, T_max, n_vars=7, width=256, depth=4):
        super().__init__()
        self.T_max = T_max
        layers = [nn.Linear(1, width), nn.GELU()]
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

    def forward(self, t):
        return self.net(t / self.T_max)


def time_derivatives(net, t):
    t = t.clone().requires_grad_(True)
    z = net(t)
    dz = torch.zeros_like(z)
    for i in range(z.shape[1]):
        dz[:, i:i+1] = torch.autograd.grad(
            z[:, i].sum(), t, create_graph=True)[0]
    return z, dz
