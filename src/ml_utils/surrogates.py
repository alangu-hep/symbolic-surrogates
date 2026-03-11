import pysr
from pysr import PySRRegressor

import torch

class Surrogate(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()

        self.bkg, self.signal = modules

    def forward(self, x):

        bkg_logit = self.bkg(x).unsqueeze(1)
        signal_logit = self.signal(x).unsqueeze(1)

        return torch.cat([bkg_logit, signal_logit], axis=1)

class FullSurrogate(torch.nn.Module):
    def __init__(self, modules, dr):
        super(FullSurrogate, self).__init__()
        self.dr = dr
        self.surrogate = Surrogate(modules)

    def forward(self, points, features, lorentz_vectors, mask):
        _, mean, log_var, z = self.dr(points, features, lorentz_vectors, mask)
        return self.surrogate(torch.cat([mean, log_var], axis=1))

class HingeSurrogate(torch.nn.Module):
    def __init__(self, modules, dr):
        super(HingeSurrogate, self).__init__()
        self.dr = dr
        self.surrogate = modules

    def forward(self, points, features, lorentz_vectors, mask):
        _, mean, log_var, z = self.dr(points, features, lorentz_vectors, mask)
        return self.mod(x)