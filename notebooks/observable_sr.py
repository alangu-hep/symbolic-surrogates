#!/usr/bin/env python3

import pysr
from pysr import PySRRegressor

import sys
import os
import argparse
import glob
from datetime import datetime

workdir = os.getenv('WORKDIR')
sys.path.append(f'{workdir}/src')

from handlers import trainer, evaluation, annealing, sr_trainer, visualizer
from handlers.args import setup_argparse
from handlers.model_loader import TorchLoader, SurrogateLoader

from ml_utils import losses
from ml_utils import surrogates
from ml_utils.optimizers import optim

from metrics import complexity, faithfulness, interpretability

from preprocessing.dataloaders import train_load, test_load
from preprocessing.datasets import SimpleIterDataset

from postprocessing.io_writer import _write_outputs_to_root

import torch
import torch.nn as nn
import numpy as np

from importlib.util import spec_from_file_location, module_from_spec

from weaver.utils.logger import _logger, warn_n_times, _configLogger
import copy
from collections import defaultdict

from main import assemble_loaders

import energyflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Args:
    def __init__(self, **kwargs):
        # defaults
        self.data_train = []
        self.data_test = []
        self.data_val = []
        self.num_workers = 0
        self.num_epochs = 0
        self.data_config = ''
        self.file_fraction = 1
        self.data_fraction = 1
        self.batch_size = 0
        self.local_rank = None
        self.model_prefix = None
        self.lr_finder = None
        self.optimizer_option = []
        self.optimizer = 'ranger'
        self.start_lr = 1e-3
        self.final_lr = 1e-6
        self.lr_scheduler = 'flat+decay'
        self.kl_weight = 0.1
        self.class_weight = 1.0
        self.kl_anneal = False
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.bit_size = None
        self.dr_path = None
        self.dr_network = None
        
        for key, value in kwargs.items():
            setattr(self, key, value)

signals = [
    'TTBar',
    'WToQQ',
    'HToGG'
]

jc_paths = {
    'train': f'{workdir}/datasets/JetClass/Pythia/train_100M',
    'val': f'{workdir}/datasets/JetClass/Pythia/val_5M',
    'test': f'{workdir}/datasets/JetClass/Pythia/test_20M'
}

num_classes = 2
background = '/ZJetsToNuNu_*.root'

datasets = {signal: {} for signal in signals}

for signal in signals:
    for name, path in jc_paths.items():
    
        signal_glob = f'/{signal}_*.root'
        signal_files = glob.glob(path+signal_glob)
        background_files = glob.glob(path+background)
    
        datasets[signal][name] = signal_files + background_files

model_dir = f'{workdir}/outputs/models'
sr_dir = f'{workdir}/outputs/sr_runs'

config_paths = {signal: f'{workdir}/data_config/JetClass/JetClass_{signal}.yaml' for signal in signals}
fig_path = f'{workdir}/figures'

def register_models(signal):
    model_registry = {
        'TTBar': [
            SurrogateLoader(
                model_name='PN-S',
                teacher_name='ParticleNet',
                vae_network=f'{workdir}/wrappers/vae.py',
                vae_path=f'{model_dir}/TTBar/BVAE/BVAE_M6-BVAE_DR_epoch-4_state.pt',
                data_config=config_paths['TTBar'],
                sr_path=f'{sr_dir}/TTBar/Surrogate/Surrogate_M13-SRPARTICLENET_ParticleNet_BVAE_SR',
                equations=[16, 16]
            )
        ],
        'WToQQ': [
            SurrogateLoader(
                model_name='PN-S',
                teacher_name='ParticleNet',
                vae_network=f'{workdir}/wrappers/vae.py',
                vae_path=f'{model_dir}/WToQQ/BVAE/BVAE_M6-BVAE_DR_epoch-4_state.pt',
                data_config=config_paths['WToQQ'],
                sr_path=f'{sr_dir}/WToQQ/Surrogate/Surrogate_M13-SRPARTICLENET_ParticleNet_BVAE_SR',
                equations=[15, 15]
            )
        ],
        'HToGG': [
            SurrogateLoader(
                model_name='PN-S',
                teacher_name='ParticleNet',
                vae_network=f'{workdir}/wrappers/vae.py',
                vae_path=f'{model_dir}/HToGG/BVAE/BVAE_M6-BVAE_DR_epoch-4_state.pt',
                data_config=config_paths['HToGG'],
                sr_path=f'{sr_dir}/HToGG/Surrogate/Surrogate_M13-SRPARTICLENET_ParticleNet_BVAE_SR',
                equations=[23, 16]
            )
        ]
    }

    return model_registry[signal]

_model_dicts = {signal: {} for signal in signals}

skip = []

print('Variables all configured!')

def create_sr_dataset(
    input_dict,
    latents,
    approximable=[],
    dmax=4,
    beta=1.0
):

    model_inputs = copy.deepcopy(input_dict)

    jet_features = {}
    feature_names = []

    for key, value in model_inputs.items():
        if key == 'mask' or key=='features' or key=='p4':
            continue
    
        jet_features[key] = value
        feature_names.append(key)

    dlp4 = model_inputs['p4']
    dlf = model_inputs['features']
    
    part_px = dlp4[:, 0]
    part_py = dlp4[:, 1]
    part_pz = dlp4[:, 2]
    part_pt = np.hypot(part_px, part_py)
    part_e = dlp4[:, 3]
    part_deta = dlf[:, 5]
    part_dphi = dlf[:, 6]

    def wrap_phi(phi):
        return (phi + np.pi) % (2 * np.pi) - np.pi

    print(jet_features['jet_eta'].shape)
    print(part_deta.shape)
    part_eta = part_deta + np.expand_dims(jet_features['jet_eta'], axis=-1)
    part_phi = wrap_phi(part_dphi + np.expand_dims(jet_features['jet_phi'], axis=-1))
    
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
    
    EFP = energyflow.EFPSet(('d<=', dmax), measure='hadr', beta=beta)
    efps = EFP.batch_compute(hadr_coords)

    observables = []

    for key, value in jet_features.items():
        if key in approximable:
            continue
        observables.append(value)

    observables.extend(efps.T)
    feature_names = [name for name in feature_names if name not in approximable]

    
    for i in range(len(efps.T)):
        feature_names.append(f'efp_{i}')

    for arr in observables:
        print(arr.shape)

    print(len(observables))
    print(len(feature_names))
    
    x = np.array(observables).transpose((-1, 0))
    y = latents[:, :8]

    return x, y, feature_names

for signal in signals:
    if signal in skip:
        continue
    print(f'WORKING ON: {signal}')
    models = register_models(signal)

    yaml_config = f'{workdir}/data_config/JetClass/JetClass_{signal}.yaml'
    args = Args(
        data_train = datasets[signal]['train'],
        data_val = datasets[signal]['val'],
        data_test = datasets[signal]['test'],
        data_config = yaml_config,
        batch_size = 128,
        file_fraction = 1,
        data_fraction = 0.001,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    loader_dict = assemble_loaders(args)
    
    for model_loader in models:
        model = copy.deepcopy(model_loader.load()).to(device)
        name = model_loader.get_label()
        print(f'Currently working on {name}')
        eq_list = model_loader.fetch_equations()
        tester = evaluation.SurrogateStats(
            loss=loss_fn,
            eq_list = eq_list,
            model=model,
            device=device,
            loader=loader_dict['test'],
            split='test'
        )
        print(f'Initialized Classification Stats for {name}')

        with torch.no_grad():
            test_dict = tester.run()
        if isinstance(model_loader, SurrogateLoader):
            test_dict['complexity']['num_params'][0] -= complexity.total_params(model.dr)[0]
        _model_dicts[signal][name] = test_dict

        x, y, feature_names = create_sr_dataset(
            _model_dicts[signal][name]['inputs'],
            _model_dicts[signal][name]['preds']['latents']
        )

        now = datetime.now()
        outputdir = f'{sr_dir}/{signal}/BVAE_demo_{now}'
        os.makedirs(outputdir, exist_ok=True)

        regressor = PySRRegressor(
            maxsize=30,
            niterations=1500,
            populations=31,
            population_size = 27,
            ncycles_per_iteration = 760,
            binary_operators=["+", "*", "/", "^"],
            unary_operators = ["log", "sqrt"],
            constraints = {'^': (-1, 1)},
            nested_constraints = {'log': {'log': 0, 'sqrt': 1, '^': 1, '*': 1}, 'sqrt': {'log': 1, 'sqrt': 0}},
            output_directory=outputdir,
            elementwise_loss="L2DistLoss()")
        
        regressor.fit(x, y, variable_names=feature_names)

        del model
        del model_loader
        del regressor
        torch.cuda.empty_cache()