import subprocess
import itertools

# Define the argument combinations
args_combinations = {
    # "--activation": ["relu", "tanh", "sigmoid"],
    "--augment": ["cutmix", "mixup"],
    # "--batch_size": [16, 32],
    # "--dataloader_workers": [0, 2],
    # "--decay": ["linear", "exponential", "cosine", "piecewise"],
    # "--depth": [32, 56],
    "--dropout": [0.0, 0.1],
    # "--epochs": [30, 50],
    # "--label_smoothing": [0.0, 0.1],
    "--learning_rate": [0.1, 0.01],
    # "--learning_rate_final": [0.001, 0.0001],
    # "--model": ["model5", "resnet"],
    # "--optimizer": ["SGD", "Adam"],
    # "--seed": [42, 123],
    # "--stochastic_depth": [0.0, 0.1],
    # "--threads": [1, 4],
    # "--weight_decay": [0.004, 0.002],
    # "--width": [1, 2]
}

# Generate all combinations of arguments
all_combinations = [dict(zip(args_combinations.keys(), values)) for values in itertools.product(*args_combinations.values())]

# Run skyrmion.py with each combination
for combination in all_combinations:
    args_list = ["python3", "skyrmion.py"]
    for arg, value in combination.items():
        args_list.append(arg)
        args_list.append(str(value))
    print(args_list)
    # subprocess.run(args_list)
