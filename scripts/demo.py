#!/usr/bin/env python3

import pysr
from pysr import PySRRegressor

import sys
import os

from pathlib import Path
import glob
workdir = os.getenv("PATH_TO_PARTGP")
sys.path.append(workdir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import uproot
import awkward as ak
import copy
import shutil
import fastjet

from weaver.utils.import_tools import import_module
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.nn.tools import evaluate_classification, train_classification
from weaver.train import optim
from weaver.utils.logger import _logger, warn_n_times

from utils.nn_utils.part_prediction import test_load, train_load, knowledge_distillation
from utils.nn_utils.hook_handler import HookHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_datasets = os.getenv("PART_DATA")

class Args:
    def __init__(self, **kwargs):
        # defaults
        self.data_train = []
        self.data_test = []
        self.data_val = []
        self.num_workers = 0
        self.num_epochs = 0
        self.data_config = ''
        self.extra_selection = None
        self.extra_test_selection = None
        self.file_fraction = 1
        self.data_fraction = 1
        self.fetch_by_files = False
        self.fetch_step = 0.01
        self.batch_size = 0
        self.in_memory = False
        self.local_rank = None
        self.copy_inputs = False
        self.demo = False
        self.no_remake_weights = False
        self.steps_per_epoch = None
        self.steps_per_epoch_val = None
        self.backend = None
        self.model_prefix = None
        self.lr_finder = None
        self.optimizer_option = []
        self.optimizer = 'ranger'
        self.start_lr = 1e-3
        self.lr_scheduler = 'flat+decay'
        self.load_epoch = None
        self.gpus = 0
        self.predict_gpus = 0
        self.regression_mode = False
        
        for key, value in kwargs.items():
            setattr(self, key, value)

def initialize_models(network_path, config_path, training = True, model_path = None) -> dict:

    models = {}
    
    network_module = import_module(network_path, name='_network_module')
    data_config = SimpleIterDataset({}, config_path, for_training=training).config
    model, model_info = network_module.get_model(data_config, num_layers = complexity['particle_attn'], num_cls_layers = complexity['class_attn'])

    if model_path is not None:
        wts = torch.load(model_path, map_location = 'cpu', weights_only = True)
        model.load_state_dict(wts)
    
    model_metadata = {
        'model': model,
        'info': model_info,
        'loss': network_module.get_loss(data_config)
    }

    return model_metadata

yaml_config = workdir + '/data_config/JetClass/JetClass_kin.yaml'
network_path = workdir + '/models/networks/part_wrapper.py'

jc_paths = {
    'train': path_datasets+'/JetClass/Pythia/train_100M',
    'val': path_datasets+'/JetClass/Pythia/val_5M',
    'test': path_datasets+'/JetClass/Pythia/test_20M'
}

num_classes = 2
signal = '/TTBar_*.root'
background = '/ZJetsToNuNu_*.root'

datasets = {}

for name, path in jc_paths.items():

    if isinstance(signal, str):
        signal_files = glob.glob(path+signal)

    if isinstance(background, str):
        background_files = glob.glob(path+background)

    datasets[name] = signal_files + background_files

complexity = {
    'particle_attn': 2,
    'class_attn': 1
}

'''
Generating the SR Dataset
'''

weights_path = workdir + '/models/torch_saved/student_models/ParT_student_1_7_epoch-4_state.pt'  

pred_metadata = initialize_models(network_path, yaml_config, training=False, model_path = weights_path)

pred_model = copy.deepcopy(pred_metadata['model']).to(device)

pred_args = Args(
    data_test = datasets['train'],
    data_config = yaml_config,
    batch_size = 64,
    file_fraction = 1,
    data_fraction = 0.01
)

test_loaders, data_config = test_load(pred_args)

hook_manager = {
    'forward_hooks': {
        'logits': 'fc'
    },
    'pre_forward_hooks': {
        'inputs': 'trimmer'
    }
}

handles, outputs = HookHandler(pred_model, hook_manager).registration(safety_remove = True)
outputs['pre_forward_hooks']['pairwise'] = []
pairwise_hook = pred_model.mod.pair_embed.embed.register_forward_pre_hook(HookHandler.save_inputs('pairwise', outputs['pre_forward_hooks']))

test_loaders, data_config = test_load(pred_args)

for name, get_test_loader in test_loaders.items():
    
    test_loader = get_test_loader()
    
    test_metric, scores, labels, observers = evaluate_classification(pred_model, test_loader, device, epoch=None, for_training=False)
    
    del test_loader

logits = torch.cat(outputs['forward_hooks']['logits']).cpu().numpy()
inputs = outputs['pre_forward_hooks']['inputs']
pw_int = outputs['pre_forward_hooks']['pairwise']

concat_inputs = []

for v in range(3):
    if inputs[0][v] is None:
        continue
    
    concat_inputs.append(
        torch.cat(
            [inputs[i][v].cpu() for i in range(len(inputs))],
            dim=0
        ).numpy()
    )

mask_indices = np.array([i for i, arr in enumerate(concat_inputs[2]) if 0 in arr]) 
nonpadded_mask = np.ones(200000, dtype=bool)
nonpadded_mask[mask_indices] = False

flattened_batches = []

for iteration in pw_int:
    tensor = iteration[0]
    
    batch_flattened = tensor.reshape(64, -1) 
    
    target_pairwise_len = 16 * 17 // 2
    target_features = 4 * target_pairwise_len 
    
    current_features = batch_flattened.shape[1]
    
    if current_features < target_features:
        padding = torch.zeros(64, target_features - current_features, 
                             device=batch_flattened.device, 
                             dtype=batch_flattened.dtype)
        batch_flattened = torch.cat([batch_flattened, padding], dim=1)
    elif current_features > target_features:
        batch_flattened = batch_flattened[:, :target_features]
    
    flattened_batches.append(batch_flattened)

primary_features = concat_inputs[0].reshape(200000, 7 * 16)[nonpadded_mask]
pairwise_features = torch.cat(flattened_batches, dim=0).detach().cpu().numpy()[nonpadded_mask]
logits = logits[nonpadded_mask]

x = np.concatenate((primary_features, pairwise_features), axis=1)

feature_names = data_config.input_dicts['pf_features']
pairwise_names = ['lnkt', 'lnz', 'lndelta', 'lnm2']

columns = []

for feature in feature_names:
    for particle in range(16):
        columns.append(f"{feature}_p{particle}")

i, j = torch.tril_indices(16, 16, offset=0)
i, j = i.numpy(), j.numpy()

for pairwise in pairwise_names:
    for interaction in range(16*17 // 2):
        p1, p2 = i[interaction], j[interaction]
        columns.append(f"{pairwise}_{p1}{p2}")

df = pd.DataFrame(x, columns=columns)

outputdir = workdir + '/outputs/pysr_outputs/sr_tests'

model = PySRRegressor(
    maxsize=60,
    niterations=1600,
    populations=48,
    population_size = 27,
    ncycles_per_iteration = 760,
    weight_optimize=0.001,
    binary_operators=[
        "+", 
        "-", 
        "*", 
        "/", 
        "^",
    ],
    unary_operators = [
        "sqrt", 
        "tanh",
        "sin",
    ],
    constraints = {
        '^': (-1, 1)
    },
    nested_constraints = {
        "*": {"tanh": 2},
        "tanh": {"tanh": 0, "^": 1, "sin": 1},
        "sin": {"sin": 0}       
    }, 
    output_directory = outputdir,
    parsimony = 0.01,
    batching=True,
    annealing=True,
    elementwise_loss = 'L1DistLoss()',
    random_state=42
)

model.fit(df, logits)

