import pysr
from pysr import PySRRegressor

import torch

class BasicSurrogate(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()

        if isinstance(modules, list):
            self.eqs = torch.nn.ModuleList(modules)
        else:
            self.eqs = torch.nn.ModuleList([modules])

    def forward(self, x):
        outputs = [eq(x) for eq in self.eqs]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)

class FullSurrogate(torch.nn.Module):
    def __init__(self, modules, dr):
        super(FullSurrogate, self).__init__()
        self.dr = dr
        self.surrogate = Surrogate(modules)

    def forward(self, points, features, lorentz_vectors, mask):
        _, mean, log_var, z = self.dr(points, features, lorentz_vectors, mask)
        return self.surrogate(torch.cat([mean, log_var], axis=1))