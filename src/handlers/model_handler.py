import sys
import os

from pysr import PySRRegressor
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from preprocessing.datasets import SimpleIterDataset
from ml_utils import surrogates

from handlers.model_loader import ModelLoader
from ml_utils.losses import Loss
from ml_utils.tracker import Tracker
from handlers.recorder import DataRecorder
from ml_utils.flattening import _flatten_label, _flatten_preds

from abc import ABC, abstractmethod
from collections import defaultdict

import tqdm
import time
import copy

class ModelHandler(ABC):
    def __init__(
        self,
        model_loader: ModelLoader,
        recorder: DataRecorder = None,
        tracker: Tracker = None,
        loss: Loss = None
    ):
        self.model_loader = model_loader
        self.loss = loss
        self.tracker = tracker
        self.recorder = recorder

    @abstractmethod
    def fit(self, *args):
        ...

    @abstractmethod
    def predict(self, *args):
        ...

class TorchHandler(ModelHandler):
    def __init__(
        self,
        args,
        device,
        dataloader: DataLoader,
        opt_func=None,
        grad_scaler=None,
        clip_norm=None,
        **kwargs
    ):
        super(TorchHandler, self).__init__(**kwargs)
        self.device = device
        self.model = copy.deepcopy(self.model_loader.load()).to(device)
        self.dataloader = dataloader
        self.opt, self.scheduler = opt_func(args, self.model, device)
        self.grad_scaler = grad_scaler
        self.clip_norm = clip_norm
        self.data_config = dataloader.dataset.config

    def unpack_loader(self, X, y):
        inputs = [X[k].to(self.device) for k in self.data_config.input_names]
        label = y[self.data_config.label_names[0]].long().to(self.device)
        try:
            mask = y[self.data_config.label_names[0] + '_mask'].bool().to(self.device)
        except KeyError:
            mask = None
        return inputs, label, mask

    def fit(self):
        self.model.train()
        with tqdm.tqdm(self.dataloader) as tq:
            for X, y, observers in tq:
                inputs, label, mask = self.unpack_loader(X, y)
                self.opt.zero_grad()
                
                with torch.amp.autocast("cuda", enabled=self.grad_scaler is not None):
                    model_output = self.model(*inputs)
                    model_output = model_output if isinstance(model_output, tuple) else (model_output,)
                    loss = self.loss(inputs, label, mask, *model_output)
                    if self.recorder is not None:
                        self.recorder.record_data(inputs, label, mask, observers, *model_output)

                self.backprop(loss)

        if self.recorder:
            return self.recorder.run()

    def predict(self):
        self.model.eval()
        with tqdm.tqdm(self.dataloader) as tq:
            for X, y, observers in tq:
                inputs, label, mask = self.unpack_loader(X, y)
                with torch.no_grad():
                    model_output = self.model(*inputs)
                    model_output = model_output if isinstance(model_output, tuple) else (model_output,)
                    if self.loss is not None:
                        loss = self.loss(inputs, label, mask, *model_output)
                    if self.recorder is not None:
                        self.recorder.record_data(inputs, label, mask, observers, *model_output)
        if self.recorder is not None:
            return self.recorder.run()
        
    def backprop(self, loss):
        if self.grad_scaler is None:
            loss.backward()
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.opt.step()
        else:
            self.grad_scaler.scale(loss).backward()
            if self.clip_norm is not None:
                self.grad_scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
        if self.scheduler and getattr(self.scheduler, '_update_per_step', False):
            self.scheduler.step()

class SurrogateHandler(TorchHandler):
    def __init__(
        self,
        **kwargs
    ):
        '''
        Similar to TorchHandler but requires a DataLoader based off a simple TensorDataset instead
        DataRecorder must also be a SurrogateRecorder
        '''
        super().__init__(**kwargs)

    def fit(self):
        self.model.train()
        with tqdm.tqdm(self.dataloader) as tq:
            for X, _ in tq:
                inputs = X
                with torch.amp.autocast("cuda", enabled=self.grad_scaler is not None):
                    model_output = self.model(*inputs)
                    model_output = model_output if isinstance(model_output, tuple) else (model_output,)
                    if self.recorder:
                        self.recorder.record_data(*model_output)
                self.backprop(loss)

        if self.recorder is not None:
            return self.recorder.run()

    def predict(self):
        self.model.eval()
        with tqdm.tqdm(self.dataloader) as tq:
            for X, _ in tq:
                inputs = X
                with torch.no_grad():
                    model_output = self.model(*inputs)
                    model_output = model_output if isinstance(model_output, tuple) else (model_output,)
                    if self.recorder:
                        self.recorder.record_data(*model_output)
        if self.recorder is not None:
            return self.recorder.run()

class EquationHandler(ModelHandler):
    def __init__(
        self,
        args,
        inputs: np.ndarray,
        targets: np.ndarray,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.args = args
        self.inputs = inputs
        self.targets = targets

    def fit(self, run_id, output_dir):

        args = self.args
        
        regressor = PySRRegressor(
            maxsize=args.max_size,
            niterations=args.n_iterations,
            populations=args.n_populations,
            population_size=args.population_size,
            ncycles_per_iteration = args.iteration_cycles,
            weight_optimize=args.weight_optimize,
            binary_operators=args.binary_operators,
            unary_operators = args.unary_operators,
            constraints = args.constraints,
            nested_constraints = args.nested_constraints, 
            output_directory = output_dir,
            run_id = run_id,
            parsimony = args.parsimony,
            annealing=args.sr_annealing,
            batching=args.sr_batching,
            elementwise_loss = args.sr_loss,
            output_torch_format = True,
            random_state=42
        )

        regressor.fit(self.inputs, self.targets)

        return regressor
        
    def predict():
        ...