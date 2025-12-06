#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import pysr


# In[2]:


import os
import sys
workdir = os.getenv("PATH_TO_PARTGP")
sys.path.append(workdir)

import torch
import uproot
import awkward as ak
import numpy
import matplotlib.pyplot as plt
import vector
from pysr import PySRRegressor
import pandas as pd

from weaver.utils.dataset import SimpleIterDataset
from utils.nn_utils.part_prediction import test_load

path_to_outputs = workdir + '/outputs/part_outputs/model_preds/demo_outputs.root'
path_to_config = workdir + '/data_config/JetClass/JetClass_full.yaml'

rootdir = uproot.open(path_to_outputs)

preds = rootdir['predictions']
houtputs = rootdir['hook_outputs']

def calc_mass(pt, phi, eta, e):
    """
    Calculate Invariant Mass
    """
    invariant_mass = vector.zip({'pt': pt, 'phi': phi, 'eta': eta, 'E': e}).M
    return invariant_mass

filter_out = ['scores', 'labels__label_']
features = [key for key in preds.keys() if key not in filter_out]

sample_data = []

for data in preds.iterate(features, library = 'ak'):

    data['mass'] = calc_mass(data['jet_pt'], data['jet_phi'], data['jet_eta'], data['jet_energy'])
    sample_data.append(data)

all_data = ak.concatenate(sample_data)

df = pd.DataFrame.from_records(all_data.to_list())

inputs = df.values
input_names = df.columns.tolist()
outputs = houtputs['logits'].array(library = 'np')

sr_outputs = workdir + '/outputs/pysr_outputs/demo_run'

model = PySRRegressor(
    maxsize=40,
    niterations=50,
    populations=31,
    population_size = 20,
    ncycles_per_iteration = 100,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators = ["abs", "log", "sqrt", "relu"],
    constraints = {'^': (-1, 1)},
    output_directory = sr_outputs,
    elementwise_loss="loss(prediction, target) = (prediction - target)^2")

model.fit(inputs, outputs, variable_names = input_names)
