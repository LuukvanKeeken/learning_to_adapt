import tensorflow as tf
import time
from learning_to_adapt.envs.cartpole_env import CartPoleEnv
from learning_to_adapt.logger import logger
from experiment_utils.CartPoleEval import evaluate_agent, evaluate_agent_vectorized
import numpy as np

from learning_to_adapt.samplers.vectorized_env_executor import ParallelEnvExecutor
from learning_to_adapt.envs.normalized_env import normalize

class Trainer(object):
    """
    Training script for Learning to Adapt

    Args:
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        initial_random_sampled (bool) : Whether or not to collect random samples in the first iteration
        dynamics_model_max_epochs (int): Number of epochs to train the dynamics model
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            env,
            sampler,
            sample_processor,
            policy,
            dynamics_model,
            n_itr,
            start_itr=0,
            initial_random_samples=True,
            dynamics_model_max_epochs=200,
            sess=None,
            evaluate_agent=False,
            adapt_batch_size=None,
            num_eval_episodes=100,
            ):

        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.dynamics_model = dynamics_model
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.dynamics_model_max_epochs = dynamics_model_max_epochs

        self.initial_random_samples = initial_random_samples

        self.adapt_batch_size = adapt_batch_size
        self.num_eval_episodes = num_eval_episodes

        if sess is None:
            sess = tf.Session()
        self.sess = sess

        if evaluate_agent:
            self.evaluate_agent = True
            self.eval_seeds = np.load('./seeds/evaluation_seeds.npy')
            eval_env = normalize(CartPoleEnv(reset_every_episode=False, task=None))
            self.eval_envs = ParallelEnvExecutor(eval_env, 5, 5, 200)
        else:
            self.evaluate_agent = False

    def train(self):
        """
        Collects data and trains the dynamics model
        """
        with self.sess.as_default() as sess:

            # Initialize uninitialized vars  (only initialize vars that were not loaded)
            # uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.initializers.global_variables())

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                time_env_sampling_start = time.time()

                if self.initial_random_samples and itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    env_paths = self.sampler.obtain_samples(log=True, random=True, log_prefix='')

                else:
                    logger.log("Obtaining samples from the environment using the policy...")
                    env_paths = self.sampler.obtain_samples(log=True, log_prefix='')

                logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)

                ''' -------------- Process the samples ----------------'''
                logger.log("Processing environment samples...")

                time_env_samp_proc = time.time()
                samples_data = self.sample_processor.process_samples(env_paths, log=True)
                logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)

                ''' --------------- Fit the dynamics model --------------- '''

                time_fit_start = time.time()

                logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
                self.dynamics_model.fit(samples_data['observations'],
                                        samples_data['actions'],
                                        samples_data['next_observations'],
                                        epochs=self.dynamics_model_max_epochs,
                                        verbose=True,
                                        log_tabular=True)

                logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

                """ -------------- Agent Evaluation ---------------------"""

                if self.evaluate_agent:
                    logger.log("Evaluating agent...")
                    time_agent_eval = time.time()
                    eval_rewards = evaluate_agent_vectorized(self.policy, self.eval_envs, self.num_eval_episodes, self.eval_seeds, self.adapt_batch_size)
                    # eval_rewards = evaluate_agent_3(self.policy, 5, 100, 200, self.eval_seeds, 16)
                    logger.logkv('Eval-AverageReturn', np.mean(eval_rewards))
                    logger.record_tabular('Time-AgentEval', time.time() - time_agent_eval)

                """ ------------------- Logging --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                self.log_diagnostics(env_paths, '')
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()
                if itr == 1:
                    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy, env, and dynamics model for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, dynamics_model=self.dynamics_model)

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
