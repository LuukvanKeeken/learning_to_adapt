import joblib
import tensorflow as tf
import argparse
import os.path as osp
from learning_to_adapt.samplers.utils import rollout
import json

from learning_to_adapt.samplers.vectorized_env_executor import ParallelEnvExecutor
from .CartPoleEval import evaluate_agent_pole_length_range, evaluate_agent_pole_mass_range, evaluate_agent_force_mag_range
import numpy as np
import csv
from experiment_utils.one_evaluation import one_evaluation

def find_smallest_postloss_index(path):
    data = {}
    with open(path + 'progress.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                data.setdefault(key, []).append(float(value))

    post_losses = data['Post-Loss']

    return post_losses.index(min(post_losses))


if __name__ == "__main__":

    
    paths = ['data/grbal_cartpole_15itr_run1', 'data/grbal_cartpole_15itr_run2', 'data/grbal_cartpole_15itr_run3/', 'data/grbal_cartpole_15itr_run4/', 'data/grbal_cartpole_15itr_run5/']

    for path in paths:
        print(f"Now evaluating {path}")
        smallest_postloss_idx = find_smallest_postloss_index(path)
        indices = [14]
        if smallest_postloss_idx != 14:
            indices.append(smallest_postloss_idx)
        
        for i in indices:
            pole_length, pole_mass, force_mag = one_evaluation(path, i)
