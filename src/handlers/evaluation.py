import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from scipy.special import softmax, kl_div
from scipy.stats import spearmanr
import logging
import torch
import matplotlib.pyplot as plt
import mplhep

class ModelStats:
  def __init__(self, model: torch.nn.Module, model_name: str) -> None:
    
    self.model = model
    self.name = model_name

    logging.log(f'Initialized {model_name} for Evaluation')

  def initialize(self, args, use_val=False):
    if use_val:
        logging.log('Using Validation Set for Inference')
    else:
        logging.log('Using Test Set for Inference')
        
    import .
        
    