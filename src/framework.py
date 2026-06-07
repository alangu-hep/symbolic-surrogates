import pysr
from pysr import PySRRegressor

import pysr
from pysr import PySRRegressor

import sys
import os
import argparse
import glob

workdir = os.getenv('WORKDIR')
sys.path.append(f'{workdir}/src')

from handlers import annealing
from handlers.args import setup_argparse
from handlers.model_loader import TorchLoader, SurrogateLoader
from handlers.model_handler import TorchHandler, SurrogateHandler, EquationHandler
from handlers.recorder import StandardRecorder, SurrogateRecorder

from ml_utils import losses
from ml_utils import surrogates
from ml_utils.optimizers import optim

from preprocessing.dataloaders import train_load, test_load
from preprocessing.datasets import SimpleIterDataset

from postprocessing.io_writer import _write_outputs_to_root

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from weaver.utils.logger import _logger, warn_n_times, _configLogger
import time
from collections import defaultdict

import energyflow

def assemble_loaders(args):

    loaderdict = {
        'train': [],
        'val': [],
        'test': [],
    }

    if args.data_train and args.data_val:
        train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(args)
        loaderdict['train'] = train_loader
        loaderdict['val'] = val_loader
    if args.data_test:
        test_loaders, test_config = test_load(args)
        loaderdict['test'] = test_loaders

    return loaderdict

class DistillationFramework:
    def __init__(
        self,
        args
    ):
        self.args = args

        '''
        _logger.info('Started!')
        _logger.info('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))
        '''

        # Assemble General Dataset
        self.loader_dict = assemble_loaders(args)
        
        # Assemble dataset used for surrogate SR
        args.data_fraction = args.data_fraction * args.surrogate_fraction
        self.loader_dict_s = assemble_loaders(args)

        # Assemble dataset used for observable SR
        args.data_fraction = args.data_fraction * args.feature_fraction
        self.loader_dict_e = assemble_loaders(args)

        # Initialize Model Loaders
        self.dl_loader = TorchLoader(
                model_name=self.args.dl_name,
                network_path=self.args.model_network,
                model_path=self.args.model_path,
                data_config=self.args.data_config
            )

        self.vae_loader = TorchLoader(
                model_name=self.args.vae_name,
                network_path=self.args.dr_network,
                model_path=self.args.dr_path,
                data_config=self.args.data_config
            )

        self.surrogate_loader = SurrogateLoader(
                model_name=self.args.surrogate_name,
                teacher_name=self.args.dl_name,
                sr_path=self.args.surrogate_prefix,
                equations=None
            )

        self.grad_scaler = torch.amp.GradScaler("cuda")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_dl(
        self,
        split = 'train',
        dl_loss = None,
        dl_schema = None
    ):
        dataloader = self.loader_dict_s[split]
        data_config = dataloader.dataset.config
        dl_recorder = None
        if dl_schema:
            dl_recorder = StandardRecorder(
                data_config = data_config,
                schema = dl_schema
            )
        self.dl_handler = TorchHandler(
            args = self.args,
            device = self.device,
            dataloader = dataloader,
            opt_func = optim,
            grad_scaler = self.grad_scaler,
            clip_norm = None,
            model_loader = self.dl_loader,
            recorder = dl_recorder,
            tracker = None,
            loss = dl_loss
        )

    def prepare_vae(
        self,
        split = 'train',
        vae_loss = None,
        vae_schema = None
    ):
        dataloader = self.loader_dict_s[split]
        data_config = dataloader.dataset.config
        vae_recorder = None
        if vae_schema:
            vae_recorder = StandardRecorder(
                data_config = data_config,
                schema = vae_schema
            )
        self.vae_handler = TorchHandler(
            args = self.args,
            device = self.device,
            dataloader = dataloader,
            opt_func = optim,
            grad_scaler = self.grad_scaler,
            clip_norm = None,
            model_loader = self.vae_loader,
            recorder = vae_recorder,
            tracker = None,
            loss = vae_loss
        )

    def create_eq_dataset(self):
        '''
        Requires data recorders for VAE & DL
        '''

        dl_outputs = self.dl_handler.predict()
        self.dl_outputs = dl_outputs
        vae_outputs = self.vae_handler.predict()
        self.vae_outputs = vae_outputs

    def fit_surrogate(self):
        '''
        Expects:
        Predictions dir to be called preds
        DL model outputs to be called logits
        VAE latents to be called mu and log_var
        '''

        logits = self.dl_outputs['preds']['logits']
        logit_diff = logits[:, 1] - logits[:, 0]
        mu = self.vae_outputs['preds']['mu']
        log_var = self.vae_outputs['preds']['log_var']

        latents = np.concatenate([mu, log_var], axis=1)

        self.equation_handler_s = EquationHandler(
            args = self.args,
            inputs = latents,
            targets = logit_diff,
            model_loader = None
        )
        
        run_id = self.args.surrogate_prefix[self.args.surrogate_prefix.rfind('/') + 1:]
        output_dir = self.args.surrogate_prefix[:self.args.surrogate_prefix.rfind('/')]
        surrogate_regressor = self.equation_handler_s.fit(run_id, output_dir)

    def fit_observables(self):
        '''
        Expects:
        Inputs dir to be called inputs
        '''

        data_config = self.vae_handler.data_config
        input_dict = self.vae_outputs['inputs']
        mu = self.vae_outputs['preds']['mu']
        
        jet_features = {}
        feature_names = []

        observer_names = [
            'jet_pt',
            'jet_eta',
            'jet_phi',
            'jet_energy',
            'jet_nparticles',
            'jet_sdmass',
            'jet_tau1',
            'jet_tau2',
            'jet_tau3',
            'jet_tau4'
        ]

        for name in observer_names:
            jet_features.append(input_dict[name])
            feature_names.append(name)

        part_px = input_dict['part_px']
        part_py = input_dict['part_py']
        part_pz = input_dict['part_pz']
        part_pt = np.hypot(part_px, part_py)
        part_e = input_dict['part_energy']
        part_deta = input_dict['part_deta']
        part_dphi = input_dict['part_dphi']
        
        def wrap_phi(phi):
            return (phi + np.pi) % (2 * np.pi) - np.pi
        
        part_eta = part_deta + input_dict['jet_eta'].reshape(4000, 1)
        part_phi = wrap_phi(part_dphi + input_dict['jet_phi'].reshape(4000, 1))
        
        part_m2 = part_e**2 - (part_px**2 + part_py**2 + part_pz**2)
        part_m = np.sqrt(np.maximum(part_m2, 0))

        hadr_package = [
            part_pt,
            part_eta,
            part_phi,
            part_m
        ]

        ptetaphim = np.concatenate([np.expand_dims(k, axis=1) for k in hadr_package], axis=1)
        hadr_coords = np.transpose(ptetaphim, (0, 2, 1))

        dmax = 4
        beta = 1.0
        
        EFP = energyflow.EFPSet(('d<=', dmax), measure='hadr', beta=beta)

        EFP.specs
        EFP.graphs()
        efps = EFP.batch_compute(hadr_coords)
        
        observables = []
        approximable = []
    
        for key, value in jet_features.items():
            if key in approximable:
                continue
            observables.append(value)
    
        observables.extend(efps.T)
        feature_names = [name for name in feature_names if name not in approximable]
        
        for i in range(len(efps.T)):
            feature_names.append(f'efp_{i}')

        defined_observables = np.array(observables).transpose((-1, 0))
        latents = mu

        self.equation_handler_e = EquationHandler(
            args = self.args,
            inputs = defined_observables,
            targets = latents,
            model_loader = None
        )
        
        run_id = self.args.observable_prefix[self.args.observable_prefix.rfind('/') + 1:]
        output_dir = self.args.observable_prefix[:self.args.observable_prefix.rfind('/')]
        surrogate_regressor = self.equation_handler_e.fit(run_id, output_dir)