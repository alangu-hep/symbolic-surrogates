import sys
import os

from pysr import PySRRegressor
import torch

from preprocessing.datasets import SimpleIterDataset
from ml_utils import surrogates

from importlib.util import spec_from_file_location, module_from_spec

from abc import ABC, abstractmethod
from collections import defaultdict

def import_module(path, name='_mod'):
    pkg_root = os.path.dirname(os.path.abspath(path))
    not_inserted = pkg_root not in sys.path
    if not_inserted:
        sys.path.insert(0, pkg_root)

    spec = spec_from_file_location(name, path, 
        submodule_search_locations=[]
    )
    mod = module_from_spec(spec)
    mod.__package__ = name
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path.remove(pkg_root)
    return mod

def initialize_models(training, network, data_config, model_path=None):
    
    network_module = import_module(network, name='_network_module')
    data_config = SimpleIterDataset({}, data_config, for_training=training).config
    model, model_info = network_module.get_model(data_config)

    if model_path is not None:
        wts = torch.load(model_path, map_location = 'cpu', weights_only = True)
        model.load_state_dict(wts)
    
    model_metadata = {
        'model': model,
        'info': model_info
    }

    return model_metadata

class BaseLoader(ABC):
    def __init__(self, model_name, teacher_name=None):
        self.model_name = model_name
        self.teacher_name = teacher_name

    @abstractmethod
    def load(self):
        ...

    def get_label(self):
        return self.model_name

    def get_teacher(self):
        return self.teacher_name

class TorchLoader(BaseLoader):
    def __init__(self, network_path, data_config, model_path, **kwargs):
        super().__init__(**kwargs)
        self.network = network_path
        self.config = data_config
        self.model_path = model_path

    def load(self):
        model_dict = initialize_models(training=False, network=self.network, data_config=self.config, model_path=self.model_path)
        return model_dict['model']

class SurrogateLoader(BaseLoader):
    def __init__(self, sr_path, vae_network, vae_path, data_config, equations: list, **kwargs):
        super().__init__(**kwargs)
        self.sr_path = sr_path
        self.vae_network = vae_network
        self.config = data_config
        self.vae_path = vae_path
        self.equations = equations

    def load(self):
        regressor = PySRRegressor.from_file(run_directory = self.sr_path)
        vae = initialize_models(training=False, network=self.vae_network, data_config=self.config, model_path=self.vae_path)['model']
        modules = regressor.pytorch(self.equations)
        surrogate = surrogates.FullSurrogate(modules, dr=vae)
        return surrogate
    def fetch_equations(self):
        regressor = PySRRegressor.from_file(run_directory = self.sr_path)
        return regressor.sympy(self.equations)

class HingeLoader(BaseLoader):
    def __init__(self, sr_path, vae_network, vae_path, data_config, equation, **kwargs):
        super().__init__(**kwargs)
        self.sr_path = sr_path
        self.vae_network = vae_network
        self.config = data_config
        self.vae_path = vae_path
        self.equation = equation

    def load(self):
        regressor = PySRRegressor.from_file(run_directory = self.sr_path)
        vae = initialize_models(training=False, network=self.vae_network, data_config=self.config, model_path=self.vae_path)['model']
        module = regressor.pytorch(self.equation)
        surrogate = surrogates.HingeSurrogate(module, dr=vae)
        return surrogate