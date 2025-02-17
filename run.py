#!/usr/bin/env python3
import itertools
import subprocess
import multiprocessing
import gc
import os

SKYRMION_BASE_PATH = os.environ.get("SKYRMION_BASE_PATH")
ON_LOCAL = SKYRMION_BASE_PATH == "/home/rothals/dev/school/diploma_thesis/dev"
N_THREADS = 4

# Define the argument combinations
args_combinations = {
    "--activation": [
        "celu", "elu", "exponential", "gelu", "glu", "hard_shrink", "hard_sigmoid", 
        "hard_silu", "hard_swish", "hard_tanh", "leaky_relu", "linear", "log_sigmoid", 
        "log_softmax", "mish", "relu", "relu6", "selu", "sigmoid", "silu", "swish", 
        "soft_shrink", "softmax", "softplus", "softsign", "squareplus", "tanh", "tanh_shrink"
        ],
    "--augment": [("cutmix", "mixup")],
    # "--batch_size": 16,
    # "--bias_regularizer": 1e-5,
    "--conv_type": ["standard"],
    # "--dataloader_workers": 0,
    "--decay": ["cosine"],
    "--depth": [4],
    # "--dropout": 0.1,
    "--epochs": [6],
    "--filters": [32],
    # "--ffm": False,
    # "--kernel_regularizer": 1e-4,
    # "--kernel_size": [3],
    # "--label_smoothing": 0.0,
    # "--learning_rate": 0.1,
    # "--learning_rate_final": 0.001,
    "--logdir_suffix": ["activations"],
    "--model": ["model5"],
    "--optimizer": ["SGD"],
    # "--padding": "same",
    # "--pooling": ["max", "average"],
    # "--seed": [42],
    # "--save_model": False,
    # "--stochastic_depth": 0.0,
    # "--stride": [1, 2],
    # "--threads": 1,
    # "--weight_decay": 0.004,
    # "--width": 1
}

def run_command(config):
    """Run skyrmion.py with given configuration"""
    command = ["python3", os.path.join(os.environ["SKYRMION_BASE_PATH"], "skyrmion.py")]
    
    for arg, value in config.items():
        command.append(arg)
        if isinstance(value, tuple):
            command.extend(map(str, value))
        else:
            command.append(str(value))

    subprocess.run(command)
    gc.collect()
    
# generate all combinations of arguments
all_combinations = [dict(zip(args_combinations.keys(), values)) 
                        for values in itertools.product(*args_combinations.values())]

# gun skyrmion.py with each combination localy (serially)
if ON_LOCAL:
    for config in all_combinations:
        run_command(config)
else: 
    # run in parallel
    with multiprocessing.Pool(processes=N_THREADS) as pool:
        pool.map(run_command, all_combinations)