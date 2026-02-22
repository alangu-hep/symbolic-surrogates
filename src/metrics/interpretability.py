import pysr
from pysr import PySRRegressor
import torch

def latent_variances(latents):

    import numpy as np
    variances = []

    for i in range(len(latents[0])):
        variances.append(np.var(latents[:, i]))

    return variances