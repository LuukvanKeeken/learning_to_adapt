import itertools
import time
import gym
from learning_to_adapt.envs.cartpole_env import CartPoleEnv
from learning_to_adapt.samplers.vectorized_env_executor import ParallelEnvExecutor
import numpy as np


def evaluate_agent(policy, env, num_episodes, evaluation_seeds, max_path_length, adapt_batch_size, pole_length = 0.5, pole_mass = 0.1, force_mag = 10):
    env.env.unwrapped.length = pole_length
    env.env.unwrapped.masspole = pole_mass
    env.env.unwrapped.force_mag = force_mag


    a_bs = adapt_batch_size
    eval_rewards = []

    for i_episode in range(num_episodes):
        print(f'Eval episode {i_episode+1}')
        env.env.seed(int(evaluation_seeds[i_episode]))
        
        observations = []
        actions = []
        rewards = []

        o = env.reset()
        policy.reset()
        path_length = 0

        while path_length < max_path_length:
            if a_bs is not None and len(observations) > a_bs + 1:
                adapt_obs = observations[-a_bs - 1:-1]
                adapt_act = actions[-a_bs - 1:-1]
                adapt_next_obs = observations[-a_bs:]
                policy.dynamics_model.switch_to_pre_adapt()
                policy.dynamics_model.adapt([np.array(adapt_obs)], [np.array(adapt_act)],
                                            [np.array(adapt_next_obs)])
            a, agent_info = policy.get_action(o)
            next_o, r, d, env_info = env.step(a)
            observations.append(o)
            rewards.append(r)
            actions.append(a[0])

            path_length += 1
            if d:
                break
            o = next_o

        eval_rewards.append(sum(rewards))

    env.env.unwrapped.length = 0.5
    env.env.unwrapped.masspole = 0.1
    env.env.unwrapped.force_mag = 10

    return eval_rewards


def evaluate_agent_2(policy, num_episodes, max_path_length, evaluation_seeds, adapt_batch_size, pole_length = 0.5, pole_mass = 0.1, force_mag = 10):
    evaluation_seeds = list(evaluation_seeds)
    env = gym.make('CartPole-v0')
    env.unwrapped.length = pole_length
    env.unwrapped.masspole = pole_mass
    env.unwrapped.force_mag = force_mag

    a_bs = adapt_batch_size
    eval_rewards = []

    for i_episode in range(num_episodes):
        print(f'Eval episode {i_episode+1}')
        env.seed(int(evaluation_seeds[i_episode]))
        
        observations = []
        actions = []
        rewards = []

        o = env.reset()
        policy.reset()
        path_length = 0

        while path_length < max_path_length:
            action_selection_start = time.time()
            if a_bs is not None and len(observations) > a_bs + 1:
                adapt_obs = observations[-a_bs - 1:-1]
                adapt_act = actions[-a_bs - 1:-1]
                adapt_next_obs = observations[-a_bs:]
                policy.dynamics_model.switch_to_pre_adapt()
                policy.dynamics_model.adapt([np.array(adapt_obs)], [np.array(adapt_act)],
                                            [np.array(adapt_next_obs)])
            a, agent_info = policy.get_action(o)
            action_selection_time = time.time() - action_selection_start
            step_time = time.time()
            next_o, r, d, env_info = env.step(a[0][0])
            step_time = time.time() - step_time
            print(f'Action selection time: {action_selection_time}')
            print(f'Step time: {step_time}')
            observations.append(o)
            rewards.append(r)
            actions.append(a[0])

            path_length += 1
            if d:
                break
            o = next_o

        eval_rewards.append(sum(rewards))

    return eval_rewards



def evaluate_agent_vectorized(policy, eval_envs, num_episodes, evaluation_seeds, max_path_length, adapt_batch_size, pole_length = 0.5, pole_mass = 0.1, force_mag = 10):
    evaluation_seeds = list(evaluation_seeds)
    

    for idx in range(eval_envs.num_envs):
        # env.env.unwrapped.length = pole_length
        # env.env.unwrapped.masspole = pole_mass
        # env.env.unwrapped.force_mag = force_mag
        eval_envs.seed_individual(idx, int(evaluation_seeds.pop(0)))

    num_envs = eval_envs.num_envs
    running_paths = [_get_empty_running_paths_dict() for _ in range(num_envs)]

    policy.reset(dones = [True] * num_envs)

    obses = np.asarray(eval_envs.reset())

    finished_episodes = 0

    a_bs = adapt_batch_size

    eval_rewards = []

    while finished_episodes < num_episodes:
        
        running_path_lengths = np.array([len(running_paths[i]['observations']) for i in range(len(running_paths))])
        if a_bs is not None and (running_path_lengths > a_bs + 1).all():
            adapt_obs = [np.stack(running_paths[idx]['observations'][-a_bs - 1:-1])
                            for idx in range(num_envs)]
            adapt_act = [np.stack(running_paths[idx]['actions'][-a_bs-1:-1])
                            for idx in range(num_envs)]
            adapt_next_obs = [np.stack(running_paths[idx]['observations'][-a_bs:])
                                for idx in range(num_envs)]
            policy.dynamics_model.switch_to_pre_adapt()
            policy.dynamics_model.adapt(adapt_obs, adapt_act, adapt_next_obs)
        actions, agent_infos = policy.get_actions(obses)

        next_obses, rewards, dones, env_infos = eval_envs.step(actions)

        # agent_infos, env_infos = _handle_info_dicts(agent_infos, env_infos)

        for idx, observation, action, reward, done in zip(itertools.count(), obses, actions, rewards, dones):
            if isinstance(reward, np.ndarray):
                reward = reward[0]

            running_paths[idx]["observations"].append(observation)
            running_paths[idx]["actions"].append(action)
            running_paths[idx]["rewards"].append(reward)
            running_paths[idx]["dones"].append(done)

            if done:
                eval_rewards.append(sum(running_paths[idx]["rewards"]))
                finished_episodes += 1
                print(finished_episodes)
                if evaluation_seeds:
                    eval_envs.seed_individual(idx, int(evaluation_seeds.pop(0)))
                next_obses[idx] = eval_envs.reset_individual(idx)
                running_paths[idx] = _get_empty_running_paths_dict()
                # I think this works?
                reset_vals = [False] * num_envs
                reset_vals[idx] = True
                policy.reset(dones = reset_vals)

        obses = next_obses


    return eval_rewards






def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])
