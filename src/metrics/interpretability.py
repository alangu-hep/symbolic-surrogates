import pysr
from pysr import PySRRegressor

import torch
import numpy as np
import sympy

from collections import defaultdict
import copy

from weaver.utils.logger import _logger

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

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.func(*X.T)


class EquationVisualizer:
    def __init__(self, eq_list, latents: np.ndarray, labels: np.ndarray):
        '''
        Create visualizations for SymPy symbolic expressions based on the diff between logits
        '''

        self.eq_list = eq_list
        self.bkg = eq_list[0]
        self.signal = eq_list[1]

        self.diff_eq = self.signal - self.bkg
        self.all_symbols = sorted(self.diff_eq.free_symbols, key=lambda s: s.name)
        self.diff = sympy.lambdify(self.all_symbols, self.diff_eq, 'numpy')
        self.diff_model = SymPyWrapper(self.diff)

        _logger.info(f'Background Equation: {self.bkg}')
        _logger.info(f'Signal Equation: {self.signal}')
        _logger.info(f'Difference Equation: {self.diff_eq}')

        bkg_indices = np.where(labels == 0)[0]
        sig_indices = np.where(labels == 1)[0]
        self.sig_latents = latents[sig_indices]
        self.bkg_latents = latents[bkg_indices]
        self.latents = latents

    def pdp(self, var_idx, event=None):
        if event =='signal':
            pdp = PartialDependenceDisplay.from_estimator(self.diff_model, self.sig_latents, features=[var_idx])
        elif event=='background':
            pdp = PartialDependenceDisplay.from_estimator(self.diff_model, self.bkg_latents, features=[var_idx])
        else:
            pdp = PartialDependenceDisplay.from_estimator(self.diff_model, self.latents, features=[var_idx])
        return pdp

class HingeEquationVisualizer:
    def __init__(self, equation, latents: np.ndarray, labels: np.ndarray):
        '''
        Create visualizations for SymPy symbolic expressions based on the diff between logits
        '''

        self.diff_eq = equation

        self.all_symbols = sorted(self.diff_eq.free_symbols, key=lambda s: s.name)
        self.diff = sympy.lambdify(self.all_symbols, self.diff_eq, 'numpy')
        self.diff_model = SymPyWrapper(self.diff)

        _logger.info(f'Background Equation: {self.bkg}')
        _logger.info(f'Signal Equation: {self.signal}')
        _logger.info(f'Difference Equation: {self.diff_eq}')

        bkg_indices = np.where(labels == 0)[0]
        sig_indices = np.where(labels == 1)[0]
        self.sig_latents = latents[sig_indices]
        self.bkg_latents = latents[bkg_indices]
        self.latents = latents

    def pdp(self, var_idx, event=None):
        if event =='signal':
            pdp = PartialDependenceDisplay.from_estimator(self.diff_model, self.sig_latents, features=[var_idx])
        elif event=='background':
            pdp = PartialDependenceDisplay.from_estimator(self.diff_model, self.bkg_latents, features=[var_idx])
        else:
            pdp = PartialDependenceDisplay.from_estimator(self.diff_model, self.latents, features=[var_idx])
        return pdp

def reparametrize(latent_vector, autoencoder):
    latent_dim = autoencoder.encoder.latent_dim
    mean = latent_vector[:, :latent_dim]
    log_var = latent_vector[:, latent_dim:]
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