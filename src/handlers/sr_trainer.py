import pysr
from pysr import PySRRegressor
import torch
import tqdm
import numpy

class SymbolicTrainer:
    def __init__(self, model, dr, loader, device):

        model.eval()
        dr.eval()

        self.model = model
        self.dr = dr

        self.loader = loader
        self.data_config = loader.dataset.config

        self.device = device

        self.inputs = []
        self.logits = []

    def run(self, args):

        self.construct_dataset()

        run_id = args.sr_prefix[args.sr_prefix.rfind('/') + 1:]
        output_dir = args.sr_prefix[:args.sr_prefix.rfind('/')]
        
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

        regressor.fit(self.inputs, self.logits)

        return regressor
    
    def evaluate_input(self, inputs, mask, label):
        model_output = self.model(*inputs)
        reconstructed, mean, log_var, z = self.dr(*inputs)

        self.inputs.append(torch.cat([mean, log_var], axis=1).detach())
        self.logits.append(model_output.detach())
    
    def construct_dataset(self):
        with tqdm.tqdm(self.loader) as tq:
            for X, y, _ in tq:
                inputs = [X[k].to(self.device) for k in self.data_config.input_names]
                label = y[self.data_config.label_names[0]].long().to(self.device)
                try:
                    mask = y[self.data_config.label_names[0] + '_mask'].bool().to(self.device)
                except KeyError:
                    mask = None

                with torch.no_grad():
                    self.evaluate_input(inputs, label, mask)

        self.inputs = torch.cat(self.inputs).cpu().numpy()
        self.logits = torch.cat(self.logits).cpu().numpy()