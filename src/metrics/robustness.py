import torch

def lorentz_inner(v1, v2):
    """Lorentz inner product, i.e v1^T @ g @ v2

    Parameters
    ----------
    v1, v2 : torch.Tensor
        Tensors of shape (..., 4)

    Returns
    -------
    torch.Tensor
        Lorentz inner product of shape (..., )
    """
    t = v1[..., 0] * v2[..., 0]
    s = (v1[..., 1:] * v2[..., 1:]).sum(dim=-1)
    return t - s


def lorentz_squarednorm(v):
    """Lorentz norm, i.e. v^T @ g @ v

    Parameters
    ----------
    v : torch.Tensor
        Tensor of shape (..., 4)

    Returns
    -------
    torch.Tensor
        Lorentz norm of shape (..., )
    """
    return lorentz_inner(v, v)

def restframe_boost(fourmomenta, checks=False):
    """Construct a Lorentz transformation that boosts four-momenta into their rest frame.

    Parameters
    ----------
    fourmomenta : torch.Tensor
        Tensor of shape (..., 4) representing the four-momenta.
    checks : bool
        If True, perform additional assertion checks on predicted vectors.
        It may cause slowdowns due to GPU/CPU synchronization, use only for debugging.

    Returns
    -------
    trafo : torch.Tensor
        Tensor of shape (..., 4, 4) representing the Lorentz transformation
        that boosts the four-momenta into their rest frame.
    """
    if checks:
        assert (lorentz_squarednorm(fourmomenta) > 0).all(), (
            "Trying to boost spacelike vectors into their restframe (not possible). Consider changing the nonlinearity in equivectors."
        )

    # compute relevant quantities
    t0 = fourmomenta.narrow(-1, 0, 1)
    beta = fourmomenta[..., 1:] / t0.clamp_min(1e-10)
    beta2 = beta.square().sum(dim=-1, keepdim=True)
    one_minus_beta2 = torch.clamp_min(1 - beta2, min=1e-10)
    gamma = torch.rsqrt(one_minus_beta2)
    boost = -gamma * beta

    # prepare rotation part
    eye3 = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye3 = eye3.reshape(*(1,) * len(fourmomenta.shape[:-1]), 3, 3).expand(
        *fourmomenta.shape[:-1], 3, 3
    )
    scale = (gamma - 1) / torch.clamp_min(beta2, min=1e-10)
    outer = beta.unsqueeze(-1) * beta.unsqueeze(-2)
    rot = eye3 + scale.unsqueeze(-1) * outer

    # collect trafo
    row0 = torch.cat((gamma, boost), dim=-1)
    lower = torch.cat((boost.unsqueeze(-1), rot), dim=-1)
    trafo = torch.cat((row0.unsqueeze(-2), lower), dim=-2)
    return trafo

def rand_boost(
    shape: torch.Size,
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """Create a general pure boost, i.e. a symmetric Lorentz transformation.

    Parameters
    ----------
    shape: torch.Size
        Shape of the transformation matrices
    std_eta: float
        Standard deviation of rapidity
    n_max_std_eta: float
        Allowed number of standard deviations;
        used to sample from a truncated Gaussian
    device: str
    dtype: torch.dtype
    generator: torch.Generator

    Returns
    -------
    final_trafo: torch.tensor
        The resulting Lorentz transformation matrices of shape (..., 4, 4).
    """
    shape = torch.Size((*shape, 3))
    beta = sample_rapidity(
        shape,
        std_eta,
        n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    beta2 = (beta**2).sum(dim=-1, keepdim=True)
    gamma = 1 / (1 - beta2).clamp(min=1e-10).sqrt()
    fourmomenta = torch.cat([gamma, beta], axis=-1)

    boost = restframe_boost(fourmomenta)
    return boost


def sample_rapidity(
    shape: torch.Size,
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """Sample rapidity from a clipped gaussian distribution.

    Parameters
    ----------
    shape: torch.Size
        Shape of the output tensor
    std_eta: float
        Standard deviation of the rapidity
    n_max_std_eta: float
        Maximum number of standard deviations for truncation
    device: str
    dtype: torch.dtype
    generator: torch.Generator
    """
    eta = randn_wrapper(shape, device, dtype, generator=generator)
    angle = eta * std_eta
    angle.clamp(min=-std_eta * n_max_std_eta, max=std_eta * n_max_std_eta)
    return angle


def rand_wrapper(shape, device, dtype, generator=None):
    # ugly solution to make the code work with torch.compile
    # torch.compile doesn't accept the generator argument,
    # but we also don't use the generator argument in compiled code.
    # But we may use the generator in uncompiled code, so we have to keep it.
    if generator is None:
        return torch.rand(shape, device=device, dtype=dtype)
    else:
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def randn_wrapper(shape, device, dtype, generator=None):
    # ugly solution to make the code work with torch.compile
    # torch.compile doesn't accept the generator argument,
    # but we also don't use the generator argument in compiled code.
    # But we may use the generator in uncompiled code, so we have to keep it.
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    else:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

def augment_with_boost_general(data, beta_max):
    """
    Apply random Lorentz boost in a random direction.
    
    This uses rand_boost directly which samples boost direction uniformly.
    """
    batch_shape = data.shape[:-2]
    device = data.device
    dtype = data.dtype
    
    # Sample rapidity with std proportional to beta_max
    # For beta_max ~ 0.9, we want std_eta ~ 1.5
    std_eta = 1.5 * beta_max  # Approximate conversion
    
    boost_matrix = rand_boost(
        shape=batch_shape,
        std_eta=std_eta,
        n_max_std_eta=3.0,
        device=device,
        dtype=dtype
    )
    
    # Apply boost to all particles
    boosted_data = torch.einsum('...ij,...nj->...ni', boost_matrix, data)
    
    return boosted_data