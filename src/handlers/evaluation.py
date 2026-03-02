import torch
import logging
import copy

from ml_utils import losses
from ml_utils.tracker import AverageTracker
from ml_utils.flattening import _flatten_label, _flatten_preds
from metrics import classification, complexity, faithfulness, interpretability, robustness

from weaver.utils.logger import _logger

import numpy as np
import awkward as ak
import tqdm

from collections import defaultdict

class ModelStats:
    def __init__(self, model, device, loader, split='val') -> None:
        model.eval()
        self.model = model
        self.device = device
        if split not in ['val', 'test']:
            raise KeyError("ModelStats must be initialized for validation or testing") 
        self.split = split

        self.loaders = []

        if isinstance(loader, torch.utils.data.DataLoader):
            self.loaders.append(loader)
        else:
            if split=='test':
                for name, get_test_loader in loader.items():
                    test_loader = get_test_loader()
                    self.loaders.append(test_loader)
            else:
                raise KeyError('Provide valid loader')

        self._schema = {
            'inputs': defaultdict(list),
            'preds': defaultdict(list),
            'classification': defaultdict(list),
            'complexity': defaultdict(list),
            'interpretability': defaultdict(list),
            'faithfulness': defaultdict(list),
            'robustness': defaultdict(list)
        }
        
        self.output_dict = copy.deepcopy(self._schema)
        self.avg_trackers = defaultdict(AverageTracker)
    def run(self):
        _logger.info(f'Starting!')
        for loader in self.loaders:
            self.evaluate(loader)
        self.concat_outputs()
        self.extra_processing()
        
        if self.split == 'val':
            _logger.info(f'Commencing Model Validation')
            self.val_metrics()
        elif self.split == 'test':
            _logger.info(f'Commencing Model Testing')
            self.test_metrics()

        return self.output_dict

    def concat_outputs(self):
        pred_dict = self.output_dict['preds']
        input_dict = self.output_dict['inputs']
        for key, value in pred_dict.items():
            if not value:
                continue
            if isinstance(value[0], np.ndarray):
                pred_dict[key] = np.concatenate(value)
            elif isinstance(value[0], torch.Tensor):
                pred_dict[key] = torch.cat(value).cpu().numpy()
            else:
                raise TypeError(f"Unexpected type in preds['{key}']: {type(value[0])}")
        for key, value in input_dict.items():
            if not value:
                continue
            if isinstance(value[0], torch.Tensor):
                input_dict[key] = torch.cat(value).cpu().numpy()
            else:
                raise TypeError(f"Unexpected type in inputs['{key}']: {type(value[0])}")

    def extra_processing(self):
        return
    
    def val_metrics(self):
        raise NotImplementedError("Validation Metrics must be manually implemented")

    def test_metrics(self):
        raise NotImplementedError("Test Metrics must be manually implemented")
    
    def compute_loss(self, inputs, label, mask):
        '''
        Returns (loss, extra_display)
        Appends outputs to output_dict
        '''
        raise NotImplementedError("Loss-based computation depends on the type of ModelStats!")

    def track_avg(self, metrics: dict[str, float]):
        for key, value in metrics.items():
            self.avg_trackers[key].update(value)
    def fetch_means(self) -> dict[str, float]:
        return {k: v.mean for k, v in self.avg_trackers.items()}

    def evaluate(self, loader):
        data_config = loader.dataset.config
        with tqdm.tqdm(loader) as tq:
            for X, y, _ in tq:
                inputs = [X[k].to(self.device) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(self.device)
                try:
                    mask = y[data_config.label_names[0] + '_mask'].bool().to(self.device)
                except KeyError:
                    mask = None

                loss, extra_display = self.compute_loss(inputs, label, mask)
                
                if mask is not None:
                    mask = mask.cpu()
                for k, v in y.items():
                    self.output_dict['preds']['labels'].append(v.numpy(force=True))
                
                loss=loss.item()
                self.track_avg({'Avg Loss': loss})
            
                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    **extra_display,
                    **{k: f'{v:.5f}' for k, v in self.fetch_means().items()}
                })
        del loader

class ClassificationStats(ModelStats):
    def __init__(self, loss, **kwargs):
        super(ClassificationStats, self).__init__(**kwargs)
        self.loss = loss    
    def extra_processing(self):
        from scipy.special import softmax
        self.output_dict['preds']['probs'] = softmax(self.output_dict['preds']['logits'], axis=-1)
    def val_metrics(self):
        labels = self.output_dict['preds']['labels']
        preds = self.output_dict['preds']['probs']
        
        cce, avg_accuracy = classification.accuracy_metrics(labels, preds)
        self.output_dict['classification']['cce_loss'].append(cce)
        self.output_dict['classification']['avg_accuracy'].append(avg_accuracy)
        
    def test_metrics(self):

        self.val_metrics()
        labels = self.output_dict['preds']['labels']
        preds = self.output_dict['preds']['probs']
        
        fpr, tpr, auc = classification.roc_metrics(labels, preds)
        self.output_dict['classification']['tpr'] = tpr
        self.output_dict['classification']['fpr'] = fpr
        self.output_dict['classification']['roc_auc'].append(auc)
        self.output_dict['classification']['confusion_matrix'] = classification.confusion_matrices(labels, preds)

        self.output_dict['complexity']['num_params'].append(complexity.total_params(self.model)[0])
    
    def compute_loss(self, inputs, label, mask):
        model_output = self.model(*inputs)
        logits, labels, mask = _flatten_preds(model_output, label=label, mask=mask)
        loss = self.loss(logits, label)

        _, preds = logits.max(1)
        correct = (preds == label).sum().item()
        num_examples = label.shape[0]

        self.track_avg({'Avg Accuracy': (correct/num_examples)})
        self.output_dict['preds']['logits'].append(logits.detach())

        extra_display = {
            'Acc': '%.5f' % (correct / num_examples),
        }
        return loss, extra_display

class KDStats(ModelStats):
    def __init__(self, teacher, cl_loss, kd_loss, **kwargs):
        super(KDStats, self).__init__(**kwargs)
        teacher.eval()
        self.teacher = teacher
        self.base_loss = cl_loss
        self.kd_loss = kd_loss
        
    def extra_processing(self):
        from scipy.special import softmax
        self.output_dict['preds']['dl_probs'] = softmax(self.output_dict['preds']['dl_logits'], axis=-1)
        self.output_dict['preds']['teacher_probs'] = softmax(self.output_dict['preds']['teacher_logits'], axis=-1)
    def val_metrics(self):
        labels = self.output_dict['preds']['labels']
        preds = self.output_dict['preds']['dl_probs']
        dl_logits = self.output_dict['preds']['dl_logits']
        teacher_logits = self.output_dict['preds']['teacher_logits']
        
        cce, avg_accuracy = classification.accuracy_metrics(labels, preds)
        self.output_dict['classification']['cce_loss'].append(cce)
        self.output_dict['classification']['avg_accuracy'].append(avg_accuracy)

        self.output_dict['faithfulness']['kl_div'].append(faithfulness.kl_div(teacher_logits, dl_logits))
        
    def test_metrics(self):

        self.val_metrics()
        labels = self.output_dict['preds']['labels']
        preds = self.output_dict['preds']['dl_probs']
        teacher_preds = self.output_dict['preds']['teacher_probs']
        
        fpr, tpr, auc = classification.roc_metrics(labels, preds)
        self.output_dict['classification']['tpr'] = tpr
        self.output_dict['classification']['fpr'] = fpr
        self.output_dict['classification']['roc_auc'].append(auc)
        self.output_dict['classification']['confusion_matrix'] = classification.confusion_matrices(labels, preds)

        self.output_dict['complexity']['num_params'].append(complexity.total_params(self.model)[0])
    
    def compute_loss(self, inputs, label, mask):
        with torch.no_grad():
            teacher_output = self.teacher(*inputs)
        student_output = self.model(*inputs)
        
        t_logits, t_label, _ = _flatten_preds(teacher_output, label=label, mask=mask)
        s_logits, s_label, _ = _flatten_preds(student_output, label=label, mask=mask)
        
        class_loss = self.base_loss(s_logits, label)
        kd_loss = self.kd_loss(s_logits, t_logits)
        loss = class_loss + kd_loss

        _, preds = s_logits.max(1)
        correct = (preds == label).sum().item()
        num_examples = label.shape[0]

        self.output_dict['preds']['dl_logits'].append(s_logits.detach())
        self.output_dict['preds']['teacher_logits'].append(t_logits.detach())

        class_loss = class_loss.item()
        kd_loss = kd_loss.item()

        self.track_avg({'Avg CL': class_loss, 'Avg KDL': kd_loss, 'AvgAcc': (correct/num_examples)})

        extra_display = {
            'Acc': '%.5f' % (correct / num_examples),
            'Class Loss': '%.5f' % (class_loss),
            'KD Loss': '%.5f' % (kd_loss)
        }
        return loss, extra_display

class BVAEStats(ModelStats):
    def __init__(self, vae_loss, recon_loss, **kwargs):
        super(BVAEStats, self).__init__(**kwargs)
        self.vae_loss = vae_loss
        self.recon_loss = recon_loss
        
    def val_metrics(self):
        
        latents = self.output_dict['preds']['latents']
        samples = self.output_dict['preds']['samples']
        
        self.output_dict['interpretability']['latent_variances'] = interpretability.latent_variances(latents)
        self.output_dict['interpretability']['active_units'] = interpretability.active_units(samples)
        
    def test_metrics(self):

        self.val_metrics()
        
        self.output_dict['complexity']['num_params'].append(complexity.total_params(self.model)[0])
    
    def compute_loss(self, inputs, label, mask):
        reconstructed, mean, log_var, z = self.model(*inputs)
        target = inputs[1]
        rec_mask = inputs[3].squeeze(1)
        kld_loss, beta = self.vae_loss(mean, log_var)
        recon_loss = self.recon_loss(reconstructed, target, rec_mask)
        loss = beta*kld_loss + recon_loss

        self.output_dict['inputs']['features'].append(target.detach())
        self.output_dict['inputs']['mask'].append(inputs[3].detach())
        self.output_dict['preds']['recon'].append(reconstructed.detach())
        self.output_dict['preds']['latents'].append(torch.cat([mean, log_var], axis=1).detach())
        self.output_dict['preds']['samples'].append(z.detach())

        kld_loss = kld_loss.item()
        recon_loss = recon_loss.item()

        self.track_avg({
            'AvgKLD': kld_loss,
            'AvgReconLoss': recon_loss,
        })

        extra_display = {
            'KLD': '%.5f' % kld_loss,
            'Recon Loss': '%.5f' % recon_loss,
            'Beta': '%.2f' % beta
        }
        return loss, extra_display

class TCVAEStats(ModelStats):
    def __init__(self, vae_loss, recon_loss, dataset_size, **kwargs):
        super(TCVAEStats, self).__init__(**kwargs)
        self.vae_loss = vae_loss
        self.recon_loss = recon_loss
        self.data_size = dataset_size
        
    def val_metrics(self):
        
        latents = self.output_dict['preds']['latents']

        self.output_dict['interpretability']['latent_variances'] = interpretability.latent_variances(latents)
        
    def test_metrics(self):

        self.val_metrics()
        
        self.output_dict['complexity']['num_params'].append(complexity.total_params(self.model)[0])
    
    def compute_loss(self, inputs, label, mask):
        reconstructed, mean, log_var, z = self.model(*inputs)
        target = inputs[1]
        rec_mask = inputs[3].squeeze(1)
        mi_loss, tc_loss, dw_kl_loss, alpha, beta, gamma = self.vae_loss(z, (mean, log_var), self.data_size)
        recon_loss = self.recon_loss(reconstructed, target, rec_mask)
        loss = alpha*mi_loss + beta*tc_loss + gamma*dw_kl_loss + recon_loss

        self.output_dict['inputs']['features'].append(target.detach())
        self.output_dict['inputs']['mask'].append(inputs[3].detach())
        self.output_dict['preds']['recon'].append(reconstructed.detach())
        self.output_dict['preds']['latents'].append(torch.cat([mean, log_var], axis=1).detach())


        mi_loss = mi_loss.item()
        tc_loss = tc_loss.item()
        dw_kl_loss = dw_kl_loss.item()
        recon_loss = recon_loss.item()

        self.track_avg({
            'AvgMIL': mi_loss,
            'AvgTCL': tc_loss,
            'AvgDWKL': dw_kl_loss,
            'AvgReconLoss': recon_loss,
        })

        extra_display = {
            'MIL': '%.5f' % mi_loss,
            'TCL': '%.5f' % tc_loss,
            'DWKL': '%.5f' % dw_kl_loss,
            'ReconLoss': '%.5f' % recon_loss,
            'Beta': '%.2f' % beta
        }
        return loss, extra_display

class STCVAEStats(TCVAEStats):
    def __init__(self, teacher, sup_loss, **kwargs):
        super(STCVAEStats, self).__init__(**kwargs)
        teacher.eval()
        self.teacher = teacher
        self.sup_loss = sup_loss
        
    def val_metrics(self):
        
        latents = self.output_dict['preds']['latents']

        self.output_dict['interpretability']['latent_variances'] = interpretability.latent_variances(latents)
        
    def test_metrics(self):

        self.val_metrics()
        
        self.output_dict['complexity']['num_params'].append(complexity.total_params(self.model)[0])
    
    def compute_loss(self, inputs, label, mask):
        with torch.no_grad():
            teacher_logits = self.teacher(*inputs)
        reconstructed, mean, log_var, z, logits = self.model(*inputs)
        target = inputs[1]
        rec_mask = inputs[3].squeeze(1)
        mi_loss, tc_loss, dw_kl_loss, alpha, beta, gamma = self.vae_loss(z, (mean, log_var), self.data_size)
        recon_loss = self.recon_loss(reconstructed, target, rec_mask)
        sup_loss, sup_wt = self.sup_loss(logits, teacher_logits)
        loss = alpha*mi_loss + beta*tc_loss + gamma*dw_kl_loss + recon_loss + sup_wt*sup_loss

        self.output_dict['inputs']['features'].append(target.detach())
        self.output_dict['inputs']['mask'].append(inputs[3].detach())
        self.output_dict['preds']['recon'].append(reconstructed.detach())
        self.output_dict['preds']['latents'].append(torch.cat([mean, log_var], axis=1).detach())
        self.output_dict['preds']['logits'].append(logits.detach())
        self.output_dict['preds']['teacher_logits'].append(teacher_logits.detach())

        mi_loss = mi_loss.item()
        tc_loss = tc_loss.item()
        dw_kl_loss = dw_kl_loss.item()
        recon_loss = recon_loss.item()
        sup_loss = sup_loss.item()

        self.track_avg({
            'AvgMIL': mi_loss,
            'AvgTCL': tc_loss,
            'AvgDWKL': dw_kl_loss,
            'AvgReconLoss': recon_loss,
            'AvgSupLoss': sup_loss
        })

        
        extra_display = {
            'MIL': '%.5f' % mi_loss,
            'TCL': '%.5f' % tc_loss,
            'DWKL': '%.5f' % dw_kl_loss,
            'ReconLoss': '%.5f' % recon_loss,
            'SupLoss': '%.5f' % sup_loss,
            'Beta': '%.2f' % beta
        }
        return loss, extra_display

class SurrogateStats(ClassificationStats):
    def __init__(self, dr, **kwargs):
        super(SurrogateStats, self).__init__(**kwargs)
        dr.eval()
        self.dr = dr
        
    def compute_loss(self, inputs, label, mask):
        reconstructed, mean, log_var, z = self.dr(*inputs)
        model_output = self.model(torch.cat([mean, log_var], axis=1))
        logits, labels, _ = _flatten_preds(model_output, label=label, mask=mask)
        loss = self.loss(logits, label)

        _, preds = logits.max(1)
        correct = (preds == label).sum().item()
        num_examples = label.shape[0]

        self.track_avg({'Avg Accuracy': (correct/num_examples)})
        
        self.output_dict['preds']['logits'].append(logits.detach())

        extra_display = {
            'Acc': '%.5f' % (correct / num_examples)
        }
        return loss, extra_display