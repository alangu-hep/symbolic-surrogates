import pysr
from pysr import PySRRegressor

import torch

class BasicSurrogate(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()

        if isinstance(modules, list):
            self.eqs = nn.ModuleList(modules)
        else:
            self.eqs = nn.ModuleList([modules])

    def forward(self, x):

        return [eq(x) for eq in self.eqs]

class FullSurrogate(torch.nn.Module):
    def __init__(self, modules, dr):
        super(FullSurrogate, self).__init__()
        self.dr = dr
        self.surrogate = Surrogate(modules)

    def forward(self, points, features, lorentz_vectors, mask):
        _, mean, log_var, z = self.dr(points, features, lorentz_vectors, mask)
        return self.surrogate(torch.cat([mean, log_var], axis=1))