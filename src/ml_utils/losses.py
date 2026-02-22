import torch
import torch.nn as nn
import math

class BaseLoss(nn.Module):
    def __init__(self, weight = 1.0, annealer = None):
        super(BaseLoss, self).__init__()
        self.weight = weight
        self.annealer = annealer

    def apply_weight(self, loss_tensor):
        if self.annealer is not None:
            wt = self.weight * self.annealer()
            self.annealer.step()
        else:
            wt = self.weight
        return wt * loss_tensor

class HuberLoss(BaseLoss):
    def __init__(self, threshold, **kwargs):
        super(HuberLoss, self).__init__(**kwargs)
        self.loss = nn.HuberLoss(reduction='mean', delta=threshold)
    def forward(self, inputs, targets):
        return self.loss(inputs, targets), self.weight

class KD_DKL(BaseLoss):
    def __init__(self, T=3.0, temp_annealer=None, reduction='batchmean', **kwargs):
        super(KD_DKL, self).__init__(**kwargs)

        self.temp = T
        self.dist = torch.nn.KLDivLoss(reduction='batchmean')
        
        self.temp_annealer = temp_annealer

    def forward(self, student_logits, teacher_logits):
        if self.temp_annealer is not None:
            T = self.temp * self.temp_annealer()
            self.temp_annealer.step()
        else:
            T = self.temp
        
        teacher_probs = torch.nn.functional.softmax(teacher_logits/T, dim=-1)
        student_probs = torch.nn.functional.log_softmax(student_logits/T, dim=-1)

        distance = self.dist(student_probs, teacher_probs)
        distance = distance*(T**2)
        return self.apply_weight(distance)

class ChamferDist(BaseLoss):
    def __init__(self, p_order = 2, squared=True, **kwargs):
        super(ChamferDist, self).__init__(**kwargs)

        self.p = p_order
        self.squared = squared

    def forward(self, recon, target, mask):

        batch_losses = []

        for i in range(recon.size(0)):
            valid_mask = mask[i].bool() 
            
            recon_valid = recon[i, :, valid_mask]
            target_valid = target[i, :, valid_mask] 
            
            recon_valid = recon_valid.T
            target_valid = target_valid.T
            
            dist = torch.cdist(recon_valid, target_valid, p=2)
            
            if self.squared:
                dist = dist.pow(2)
            
            loss = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
            batch_losses.append(loss)
        
        return torch.stack(batch_losses).mean()

class TCVAELoss(BaseLoss):
    def __init__(self, alpha, beta, gamma, use_mss=True, **kwargs):
        super(TCVAELoss, self).__init__(**kwargs)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mss = use_mss

    def forward(self, latent_sample, distribution, data_size):
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                             distribution,
                                                                             data_size,
                                                                             self.mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        if self.annealer():
            self.beta *= self.annealer()
            self.annealer.step()

        return mi_loss, tc_loss, dw_kl_loss, self.alpha, self.beta, self.gamma

def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    B = batch_size

    strat_weight = (N - B) / (N * (B - 1))

    W = torch.full((B, B), strat_weight)
    diag = torch.arange(B)
    W[diag, diag] = 1.0 / N

    return W.log()


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        mat_log_qz_sum = mat_log_qz.sum(2)
        
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz_sum = mat_log_qz_sum + log_iw_mat
        
        log_qz = torch.logsumexp(mat_log_qz_sum, dim=1, keepdim=False)
        
        mat_log_qz_weighted = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)
        log_prod_qzi = torch.logsumexp(mat_log_qz_weighted, dim=1, keepdim=False).sum(1)
    else:
        log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
        log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx
