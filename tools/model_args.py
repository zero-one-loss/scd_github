from dataclasses import dataclass, field, fields, asdict
from multiprocessing import cpu_count
import json
import sys
import os
import torch

@dataclass
class ModelArgs:

    target: str = 'models.pkl'
    nrows: float = 0.75
    nfeatures: float = 1.0
    w_inc: float = 0.17
    tol: float = 0.00000
    local_iter: int = 100
    num_iters: int = 1000
    interval: int = 20
    rounds: int = 100
    w_inc1: float = 0.17
    updated_features: int = 128
    n_jobs: int = 1
    num_gpus: int = 1
    adv_train: bool = False
    eps: float = 0.1
    w_inc2: float = 0.2
    hidden_nodes: int = 20
    evaluation: bool = True
    verbose: bool = True
    b_ratio: float = 0.2
    cuda: bool = True
    seed: int = 2018
    save: bool = False
    criterion: classmethod = torch.nn.Module
    structure: classmethod = torch.nn.Module
    dataset: str = 'mnist'
    num_classes: int = 2
    c: float = 1.0
    gpu: int = 0
    fp16: bool = False



    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            json.dump(asdict(self), f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)

