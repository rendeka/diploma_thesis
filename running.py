#!/usr/bin/env python3

import itertools
import subprocess
import multiprocessing
import gc
import os

class RUN:
    SKYRMION_BASE_PATH: str = os.environ.get("SKYRMION_BASE_PATH")
    ON_LOCAL: bool = SKYRMION_BASE_PATH == "/home/rothals/dev/school/diploma_thesis/dev"
    
    def __init__(self, name, args_combinations: dict[str, list], n_threads: int = 4):
        """
        Initialize an experiment with a given name and hyperparameter combinations.

        :param name: A descriptive name for the experiment.
        :param args_combinations: A dictionary of hyperparameter lists.
        """
        self.name = name
        self.all_combinations = [
            dict(zip(args_combinations.keys(), values))
            for values in itertools.product(*args_combinations.values())
        ]
        self.N_THREADS = n_threads
    
    def run_command(self, config):
        """Run skyrmion.py with the given configuration."""
        command = ["python3", os.path.join(self.SKYRMION_BASE_PATH, "skyrmion.py")]
        
        for arg, value in config.items():
            command.append(arg)
            if isinstance(value, tuple):
                command.extend(map(str, value))
            else:
                command.append(str(value))

        subprocess.run(command)
        gc.collect()
    
    def run(self):
        """Run all configurations either serially (local) or in parallel."""
        print(f"Running experiment: {self.name}")
        if self.ON_LOCAL:
            for config in self.all_combinations:
                self.run_command(config)
        else:
            with multiprocessing.Pool(processes=self.N_THREADS) as pool:
                pool.map(self.run_command, self.all_combinations)