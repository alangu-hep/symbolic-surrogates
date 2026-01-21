'''
Basic MLP for SR testing
'''

import torch
import torch.nn as nn
from weaver.utils.logger import _logger

class BasicMLP(nn.Module):
    def __init__(self,
                 input_dim = 17,
                 num_classes = 10,
                 num_layers = 3,
                scale_factor = 1) -> None:
        super().__init__()

        self.neuron_count = 64 * scale_factor
        
        self.layers = [
            nn.Linear(input_dim, self.neuron_count), 
            nn.ReLU()
        ]

        for i in range(num_layers):
            self.layers.extend([
            nn.Linear(self.neuron_count, self.neuron_count), 
            nn.ReLU()
        ])

        self.layers += [nn.Linear(self.neuron_count, num_classes)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        logits = self.layers(x)
        return logits

def get_model(data_config, **kwargs):

    cfg = dict(input_dim = len(data_config.input_dicts['pf_features']), 
               num_classes = len(data_config.label_value),
               num_layers = 6, 
               scale_factor = 1)
    
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = BasicMLP(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()