import joblib
import tensorflow as tf
import argparse
import os.path as osp
from learning_to_adapt.samplers.utils import rollout
import json

from learning_to_adapt.samplers.vectorized_env_executor import ParallelEnvExecutor
from .CartPoleEval import evaluate_agent_pole_length_range, evaluate_agent_pole_mass_range, evaluate_agent_force_mag_range
import numpy as np


def one_evaluation(path, i):
    tf.reset_default_graph()
    with tf.Session() as sess:

        print(f"After {i} iterations:")
        pkl_path = osp.join(path, f'itr_{i}.pkl')
        json_path = osp.join(path, 'params.json')
        json_params = json.load(open(json_path, 'r'))
        data = joblib.load(pkl_path)
        policy = data['policy']
        env = data['env']
        eval_seeds = np.load('./seeds/evaluation_seeds.npy')
        eval_envs = ParallelEnvExecutor(env, 5, 5, 200)

        print("Evaluating pole length ...")
        all_rewards_pole_length = evaluate_agent_pole_length_range(policy, eval_envs, 1, eval_seeds, 200, 16, [0.05])#, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        np.save(f'{path}/adapt_eval/pole_length_itr{i}.npy', all_rewards_pole_length)
        print(len(eval_seeds))
        print(all_rewards_pole_length)
        print("Evaluating pole mass ...")
        all_rewards_pole_mass = evaluate_agent_pole_mass_range(policy, eval_envs, 1, eval_seeds, 200, 16, [0.5])#, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
        np.save(f'{path}/adapt_eval/pole_mass_itr{i}.npy', all_rewards_pole_mass)
        print(len(eval_seeds))
        print("Evaluating force mag ...")
        all_rewards_force_mag = evaluate_agent_force_mag_range(policy, eval_envs, 1, eval_seeds, 200, 16, [2.0])#, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0])
        np.save(f'{path}/adapt_eval/force_mag_itr{i}.npy', all_rewards_force_mag)
        print(len(eval_seeds))

    for process in eval_envs.ps:
        process.terminate()

    return all_rewards_pole_length, all_rewards_pole_mass, all_rewards_force_mag