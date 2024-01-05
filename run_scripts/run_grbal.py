from learning_to_adapt.dynamics.meta_mlp_dynamics import MetaMLPDynamicsModel
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

EXP_NAME = 'grbal'


def run_experiment(config):
    tf.reset_default_graph()
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode=config['snapshot_mode'])
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    env = normalize(config['env'](reset_every_episode=True, task=config['task'], task_args=config['task_args']))

    dynamics_model = MetaMLPDynamicsModel(
        name="dyn_model",
        env=env,
        meta_batch_size=config['meta_batch_size'],
        inner_learning_rate=config['inner_learning_rate'],
        learning_rate=config['learning_rate'],
        hidden_sizes=config['hidden_sizes_model'],
        valid_split_ratio=config['valid_split_ratio'],
        rolling_average_persitency=config['rolling_average_persitency'],
        hidden_nonlinearity=config['hidden_nonlinearity_model'],
        batch_size=config['adapt_batch_size'],
        num_rollouts=config['num_rollouts'],
        max_path_length=config['max_path_length'],
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
        n_parallel=config['n_parallel'],
        max_path_length=config['max_path_length'],
        num_rollouts=config['num_rollouts'],
        adapt_batch_size=config['adapt_batch_size'],  # Comment this out and it won't adapt during rollout
    )

    sample_processor = ModelSampleProcessor(recurrent=True)

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
        adapt_batch_size=config['adapt_batch_size'],
        num_eval_episodes=config['num_eval_episodes'],
    )
    algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
                # Environment
                'env': CartPoleEnv,
                'max_path_length': 200,
                'task': 'range',
                'task_args': {'pole_length_range': (0.5, 2.0), 'pole_mass_range': (0.1, 0.1), 'force_mag_range': (10.0, 10.0)},
                'normalize': True,
                 'n_itr': 15,
                'discount': 1.,

                # Policy
                'n_candidates': 500,
                'horizon': 10,
                'use_cem': False,
                'num_cem_iters': 5,

                # Training
                'num_rollouts': 5,
                'valid_split_ratio': 0.1,
                'rolling_average_persitency': 0.99,
                'initial_random_samples': True,

                # Dynamics Model
                'meta_batch_size': 20,
                'hidden_nonlinearity_model': 'relu',
                'learning_rate': 1e-3,
                'inner_learning_rate': 0.001,
                'hidden_sizes_model': (512, 512, 512),
                'dynamic_model_epochs': 100,
                'adapt_batch_size': 16,

                #  Other
                'n_parallel': 5,
                'evaluate_agent': True,
                'num_eval_episodes': 25,
                'snapshot_mode': 'all',

    }

    for i in range(1, 6):
        config['exp_name'] = f'grbal_cartpole__{config["n_itr"]}itr_task{config["task"]}_polelengthrange_0.5_2.0_run{i}'
        run_experiment(config)
