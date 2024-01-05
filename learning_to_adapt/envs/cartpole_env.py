from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.spaces import Discrete
from learning_to_adapt.envs.base import Env
from learning_to_adapt.logger import logger
import gym
import numpy as np




class CartPoleEnv(Env, Serializable):


    def __init__(self, task='range', reset_every_episode=True, task_args = {'pole_length_range': (0.5, 2.0), 'pole_mass_range': (0.1, 0.1), 'force_mag_range': (10.0, 10.0)}):
        # Not sure if Serializable is actually needed
        Serializable.quick_init(self, locals())


        self.env = gym.make("CartPole-v0")
        self.reset_every_episode = reset_every_episode
        task = None if task == 'None' else task
        self.task_args = task_args

        
        assert task in [None, 'range']
        self.task = task

        super(CartPoleEnv, self).__init__()


    # Something flat_dim
    @property
    def action_space(self):
        space = self.env.action_space
        space.shape = (1,)
        return space


    def log_diagnostics(self, paths, prefix):
        pass
        


    @property
    def observation_space(self):
        return self.env.observation_space

    

    def render(self):
        self.env.render()


    def reset(self):
        self.reset_task()
        init_obs = self.env.reset()
        return init_obs
    

    def reset_task(self):
        if self.task == 'range':
            pole_length_range = self.task_args['pole_length_range']
            pole_mass_range = self.task_args['pole_mass_range']
            force_mag_range = self.task_args['force_mag_range']

            random_pole_length = np.random.uniform(pole_length_range[0], pole_length_range[1])
            random_pole_mass = np.random.uniform(pole_mass_range[0], pole_mass_range[1])
            random_force_mag = np.random.uniform(force_mag_range[0], force_mag_range[1])

            params = {'pole_length': random_pole_length, 
                      'pole_mass': random_pole_mass, 
                      'force_mag': random_force_mag}
            
            self.set_params(params)
        elif self.task is None:
            pass

        else:
            raise NotImplementedError




    # BASE DOESN'T HAVE THIS?
    # In the CartPole environment, the agent receives a reward
    # of +1 for every timestep that the pole remains upright,
    # including the timestep of failure. So for a triplet of
    # observation, action, and next_observation, where the
    # observation would have led to failure, the reward is 0.
    def reward(self, observation, action, next_observation):
        assert observation.ndim == 2

        # Get the x and theta values from the observations.
        x_values = observation[:, 0]
        theta_values = observation[:, 2]

        # Check whether the x and theta values are within the bounds.
        x_lower_than_threshold = x_values < -self.env.x_threshold
        x_higher_than_threshold = x_values > self.env.x_threshold
        theta_lower_than_threshold = theta_values < -self.env.theta_threshold_radians
        theta_higher_than_threshold = theta_values > self.env.theta_threshold_radians

        # Aggregate the checks.
        already_terminated = x_lower_than_threshold | x_higher_than_threshold | theta_lower_than_threshold | theta_higher_than_threshold

        # Transform False to 1.0 and True to 0.0.
        return 1 - already_terminated.astype(float)
    
    
    def seed(self, seed):
        self.env.seed(seed)

    
    def set_params(self, params):
        self.env.unwrapped.length = params['pole_length']
        self.env.unwrapped.masspole = params['pole_mass']
        self.env.unwrapped.force_mag = params['force_mag']

    def step(self, action):
        if action.shape == (1,):
            action = action[0]
        elif action.ndim == 2:
            action = action[0][0]
            
        next_obs, reward, done, info = self.env.step(action)
        # done = False
        return next_obs, reward, done, info



if __name__ == '__main__':
    env = CartPoleEnv(task='original')
    
    while True:
        env.reset()
        # env.reset_task()
        for i in range(200):
            next_state, reward, done, _= env.step(env.action_space.sample())
            env.render()
            print(env.env.unwrapped.steps_beyond_done, end='\r')

    

    
            
        



