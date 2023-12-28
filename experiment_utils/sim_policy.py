import joblib
import tensorflow as tf
import argparse
import os.path as osp
from learning_to_adapt.samplers.utils import rollout
import json

from learning_to_adapt.samplers.vectorized_env_executor import ParallelEnvExecutor
from .CartPoleEval import evaluate_agent, evaluate_agent_vectorized
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str, help='Directory with the pkl and json file')
    parser.add_argument('--max_path_length', '-l', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', '-n', type=int, default=1,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--video_filename', type=str,
                        help='path to the out video file')
    parser.add_argument('--prompt', type=bool, default=False,
                        help='Whether or not to prompt for more sim')
    parser.add_argument('--ignore_done', action='store_true',
                        help='Whether stop animation when environment done or continue anyway')
    args = parser.parse_args()

    with tf.Session() as sess:
        pkl_path = osp.join(args.param, 'params.pkl')
        json_path = osp.join(args.param, 'params.json')
        print("Testing policy %s" % pkl_path)
        json_params = json.load(open(json_path, 'r'))
        data = joblib.load(pkl_path)
        policy = data['policy']
        env = data['env']
        eval_seeds = np.load('./seeds/evaluation_seeds.npy')
        eval_envs = ParallelEnvExecutor(env, 5, 5, 200)
        for i in range(args.num_rollouts):
            print("Performing evaluation ...")
            # eval_rewards = evaluate_agent(policy, env, 100, eval_seeds, args.max_path_length, json_params.get('adapt_batch_size', None))
            eval_rewards = evaluate_agent_vectorized(policy, eval_envs, 100, eval_seeds, 200, 16)
            print(np.mean(eval_rewards))
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, ignore_done=args.ignore_done,
                           adapt_batch_size=json_params.get('adapt_batch_size', None))
            # print(sum(path['rewards']))
