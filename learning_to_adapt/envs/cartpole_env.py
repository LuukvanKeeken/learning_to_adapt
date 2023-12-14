from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.spaces import Discrete
import gym




class CartPoleEnv(Serializable):


    def __init__(self, task='original', reset_every_episode=False, task_args = None):
        # Not sure if Serializable is actually needed
        Serializable.quick_init(self, locals())


        self.env = gym.make("CartPole-v0")
        self.reset_every_episode = reset_every_episode
        self.task_args = task_args

        # Maybe 'original' should be 'None'?
        assert task in ['original']

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        space = self.env.action_space
        space.shape = (1,)
        return space
    
    # REPLACE IN SOME WAY
    def reset(self):
        test = 1
        return test

    # REPLACE IN SOME WAY
    def reward(self):
        test = 1
        return test
    
    
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)

        return next_obs, reward, done, info



if __name__ == '__main__':
    env = CartPoleEnv(task='original')
    action_space1 = env.action_space
    action_space2 = env.get_action_space()
    print(env.action_space.shape)
    print(env.action_space)
    print(env.get_action_space().shape)
    print(env.get_action_space())
    while True:
        # env.reset()
        # env.reset_task()
        env.env.reset()
        for i in range(200):
            action_space = env.get_action_space()
            next_state, reward, done, _= env.step(action_space.sample())
            env.env.render()
            
        



