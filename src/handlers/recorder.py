import torch
import copy

import numpy as np
import tqdm

from collections import defaultdict
from abc import ABC, abstractmethod

class DataRecorder(ABC):
    def __init__(self, schema: dict):
        '''
        The schema tells the recorder the indexes of any additional arguments passed during record:

        schema = {
            int: (name: str, location: str)
        }
        
        '''

        self.schema = schema
        self.names = [name for name, _ in schema.values()]
        self.locations = [loc for _, loc in schema.values()]

        self.data_dict = {
            loc: defaultdict(list) 
            for loc in set(self.locations)
        }

        self.inputs_name = 'inputs'
        self.data_dict[self.inputs_name] = defaultdict(list)

    def run(self):
        try:
            self.concat_dict()
            return self.data_dict
        finally:
            self.refresh()

    def concat_dict(self):
        self.pre_concat()
        for loc, values in self.data_dict.items():
            for name, arr in values.items():
                if not arr:
                    continue
                if isinstance(arr[0], torch.Tensor):
                    values[name] = torch.cat(arr).cpu().numpy()
                elif isinstance(arr[0], np.ndarray):
                    values[name] = np.concatenate(arr)
                else:
                    raise TypeError(f"Unexpected type in {loc}['{name}']: {type(arr[0])}")
        self.post_concat()

    def pre_concat(self):
        pass

    def post_concat(self):
        pass

    def refresh(self):
        self.data_dict = {
            loc: defaultdict(list) 
            for loc in set(self.locations)
        }
        self.data_dict[self.inputs_name] = defaultdict(list)
        
    @abstractmethod
    def record_data(self, *args):
        ...

class StandardRecorder(DataRecorder):
    def __init__(self, data_config, **kwargs):
        super().__init__(**kwargs)
        
        self.data_config = data_config
        self.feature_names = data_config.input_dicts['pf_features']
        self.four_vectors = data_config.input_dicts['pf_vectors']

    def record_data(self, inputs, label, mask, observers, *args):  
        self.data_dict[self.inputs_name]['features'].append(inputs[1])
        self.data_dict[self.inputs_name]['p4'].append(inputs[2])
        self.data_dict[self.inputs_name]['label'].append(label)
        self.data_dict[self.inputs_name]['mask'].append(inputs[3])
        for k, v in observers.items():
            self.data_dict[self.inputs_name][k].append(v)

        for idx, (arg_name, arg_loc) in self.schema.items():
            self.data_dict[arg_loc][arg_name].append(args[idx])

    def post_concat(self):
        input_dir = self.data_dict[self.inputs_name]
        inputs = input_dir.pop('features')

        if inputs.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"got {inputs.shape[1]}"
            )
        
        for idx, col in enumerate(list(inputs.swapaxes(0, 1))):
            feature_name = self.feature_names[idx]
            input_dir[feature_name] = col

        p4 = input_dir.pop('p4')
        for idx, col in enumerate(list(p4.swapaxes(0, 1))):
            feature_name = self.four_vectors[idx]
            input_dir[feature_name] = col

class SurrogateRecorder(DataRecorder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def record_data(self, *args):
        for idx, (arg_name, arg_loc) in self.schema.items():
            self.data_dict[arg_loc][arg_name].append(args[idx])