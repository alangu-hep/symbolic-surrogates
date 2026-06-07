import pysr
from pysr import PySRRegressor

import torch
import numpy as np
import sympy

from collections import defaultdict
import copy

from weaver.utils.logger import _logger

import sklearn
from sklearn.base import BaseEstimator, RegressorMixin

def latent_variances(latents):

    import numpy as np
    variances = []

    for i in range(len(latents[0])):
        variances.append(np.var(latents[:, i]))

    return variances

def active_units(samples):

    import numpy as np
    variances = []

    for i in range(len(samples[0])):
        variances.append(np.var(samples[:, i]))

    return variances

class SymPyWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, func):
        self.func = func
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.dummy_ = "dummy"

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.func(*X.T)


class EquationVisualizer:
    def __init__(self, model, inputs: torch.Tensor, labels: torch.Tensor):
        '''
        Create visualizations for SymPy symbolic expressions based on the diff between logits
        '''

        self.model = model
        self.inputs = inputs.requires_grad_()

        bkg_indices = np.where(labels == 0)[0]
        sig_indices = np.where(labels == 1)[0]
        self.sig_inputs = self.inputs[sig_indices]
        self.bkg_inputs = self.inputs[bkg_indices]

        self.data_points = defaultdict(lambda: {"bkg": [], "sig": [], "none": []})

    def pdp(self, idx, segments, order, event=None):
        if event == 'sig':
            input_tensor = self.sig_inputs
        elif event == 'bkg':
            input_tensor = self.bkg_inputs
        elif event == 'none':
            input_tensor = self.inputs
        else:
            print('Need Valid Event!')
            return

        max_val = torch.max(input_tensor[idx])
        min_val = torch.min(input_tensor[idx])

        for val in torch.linspace(min_val, max_val, segments):
            pdp_tensor = input_tensor.mean(dim=1).unsqueeze(0)
            pdp_tensor[:, idx] = val
            model_output = self.model(pdp_tensor)
            logit_diff = model_output[:, 1] - model_output[:, 0]

            for _ in range(order):
                logit_diff = torch.autograd.grad(logit_diff, pdp_tensor, create_graph=True, retain_graph=True)[0][idx]

            self.data_points[f'dim_{idx}'][event].append([val.item(), logit_diff.detach().cpu().item()])

        return self.data_points


def reparametrize(latent_vector, autoencoder, mean_only=True):
    latent_dim = autoencoder.encoder.latent_dim
    mean = latent_vector[:, :latent_dim]
    log_var = latent_vector[:, latent_dim:]
    if mean_only:
        return mean
    return autoencoder.reparametrize(mean, log_var)

def traversals(autoencoder: torch.nn.Module, latents: np.ndarray, mask, labels: np.ndarray,
               selection: tuple, mag=3, samples=5):

    autoencoder.cpu()
    autoencoder.eval()
    bkg_indices = np.where(labels == 0)[0]
    sig_indices = np.where(labels == 1)[0]

    rng = np.random.default_rng()
    bkg_sample_idx = rng.choice(bkg_indices, size=samples, replace=False)
    sig_sample_idx = rng.choice(sig_indices, size=samples, replace=False)

    bkg_samples = torch.from_numpy(latents[bkg_sample_idx])
    bkg_mask = torch.from_numpy(mask[bkg_sample_idx])
    sig_samples = torch.from_numpy(latents[sig_sample_idx])
    sig_mask = torch.from_numpy(mask[sig_sample_idx])

    recons = defaultdict(lambda: {"bkg": [], "sig": []})

    with torch.no_grad():
        for dim in selection:
            for i in range(-mag, mag + 1):
                bkg_traversal = bkg_samples.clone()
                sig_traversal = sig_samples.clone()
                bkg_traversal[:, dim] += i
                z_bkg = reparametrize(bkg_traversal, autoencoder)
                sig_traversal[:, dim] += i
                z_sig = reparametrize(sig_traversal, autoencoder)
                bkg_outputs = autoencoder.decoder(z_bkg, bkg_mask)
                sig_outputs = autoencoder.decoder(z_sig, sig_mask)
        
                recons[f'dim_{dim}']["bkg"].append(bkg_outputs.detach().cpu().numpy())
                recons[f'dim_{dim}']["sig"].append(sig_outputs.detach().cpu().numpy())

    return recons