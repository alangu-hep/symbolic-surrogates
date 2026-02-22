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