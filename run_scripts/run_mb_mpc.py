from learning_to_adapt.dynamics.mlp_dynamics import MLPDynamicsModel
from learning_to_adapt.trainers.mb_trainer import Trainer
from learning_to_adapt.policies.mpc_controller import MPCController
from learning_to_adapt.samplers.sampler import Sampler
from learning_to_adapt.logger import logger
from learning_to_adapt.envs.normalized_env import normalize
from learning_to_adapt.utils.utils import ClassEncoder
from learning_to_adapt.samplers.model_sample_processor import ModelSampleProcessor
from learning_to_adapt.envs import *
import json
import os
import tensorflow as tf

EXP_NAME = 'mb_mpc'


def run_experiment(config):
    tf.reset_default_graph()
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode=config['snapshot_mode'])
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    env = normalize(config['env'](reset_every_episode=True, task=config['task']))

    dynamics_model = MLPDynamicsModel(
        name="dyn_model",
        env=env,
        learning_rate=config['learning_rate'],
        hidden_sizes=config['hidden_sizes'],
        valid_split_ratio=config['valid_split_ratio'],
        rolling_average_persitency=config['rolling_average_persitency'],
        hidden_nonlinearity=config['hidden_nonlinearity'],
        batch_size=config['batch_size'],
    )

    policy = MPCController(
        name="policy",
        env=env,
        dynamics_model=dynamics_model,
        discount=config['discount'],
        n_candidates=config['n_candidates'],
        horizon=config['horizon'],
        use_cem=config['use_cem'],
        num_cem_iters=config['num_cem_iters'],
    )

    sampler = Sampler(
        env=env,
        policy=policy,
        num_rollouts=config['num_rollouts'],
        max_path_length=config['max_path_length'],
        n_parallel=config['n_parallel'],
    )

    sample_processor = ModelSampleProcessor(recurrent=False)

    algo = Trainer(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        initial_random_samples=config['initial_random_samples'],
        dynamics_model_max_epochs=config['dynamic_model_epochs'],
        evaluate_agent = config['evaluate_agent'],
        num_eval_episodes=config['num_eval_episodes'],
    )
    algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
            # Environment
            'env': CartPoleEnv,
            'task': 'None',
            'normalize': True,

            # Policy
            'n_candidates': 500,
            'horizon': 10,
            'use_cem': False,
            'num_cem_iters': 5,
            'discount': 1.,

            # Sampling
            'max_path_length': 200,
            'num_rollouts': 10,
            'initial_random_samples': True,

            # Training
            'n_itr': 15,
            'learning_rate': 1e-3,
            'batch_size': 256,
            'dynamic_model_epochs': 100,
            'valid_split_ratio': 0.1,
            'rolling_average_persitency': 0.99,

            # Dynamics Model
            'hidden_sizes': (512, 512, 512),
            'hidden_nonlinearity': 'relu',


            #  Other
            'n_parallel': 5,
            'evaluate_agent': True,
            'num_eval_episodes': 25,
            'snapshot_mode': 'all',
            }

    for i in range(1, 6):
        config['exp_name'] = f'MBMPC_cartpole_{config["n_itr"]}itr_task{config["task"]}_run{i}'
        run_experiment(config)