import torch
from weaver.utils.logger import _logger

def total_params(model: torch.nn):

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return params, trainable_params