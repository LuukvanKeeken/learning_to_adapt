from copy import deepcopy
from itertools import chain
import random
import gym
from learning_to_adapt.dynamics.core.layers import MLP
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.utils import tensor_utils
from learning_to_adapt.logger import logger
import time
from learning_to_adapt.dynamics.core.utils import create_keras_mlp


class MetaMLPDynamicsModel(Serializable):
    """
    Class for MLP continous dynamics model
    """

    _activations = {
        None: None,
        "relu": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self,
                 name,
                 env,
                 hidden_sizes=(512, 512),
                 meta_batch_size=10,
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 inner_learning_rate=0.1,
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 num_rollouts=5,
                 max_path_length=200,
                 ):

        Serializable.quick_init(self, locals())

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None
        self.meta_batch_size = meta_batch_size

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.inner_learning_rate = inner_learning_rate
        self.name = name
        self._dataset_train = None
        self._dataset_test = None
        self._prev_params = None
        self._adapted_param_values = None
        self.num_rollouts = num_rollouts
        self.max_path_length = max_path_length
        self.env = env

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]

        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]


        self.inference_model = create_keras_mlp(input_dim=obs_space_dims + action_space_dims,
                                                output_dim=obs_space_dims,
                                                hidden_sizes=hidden_sizes,
                                                hidden_nonlinearity=hidden_nonlinearity,
                                                output_nonlinearity=output_nonlinearity)



        """ ------------------ Pre-Update Graph + Adaptation ----------------------- """
        with tf.variable_scope(name):
            # Placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            # Concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            # Create MLP
            mlp = MLP(name,
                      output_dim=obs_space_dims,
                      hidden_sizes=hidden_sizes,
                      hidden_nonlinearity=hidden_nonlinearity,
                      output_nonlinearity=output_nonlinearity,
                      input_var=self.nn_input,
                      input_dim=obs_space_dims+action_space_dims)

            self.delta_pred = mlp.output_var  # shape: (batch_size, ndim_obs, n_models)

            self.loss = tf.reduce_mean(tf.square(self.delta_ph - self.delta_pred))
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.adaptation_sym = tf.train.GradientDescentOptimizer(self.inner_learning_rate).minimize(self.loss)

            # Tensor_utils
            self.f_delta_pred = tensor_utils.compile_function([self.obs_ph, self.act_ph], self.delta_pred)

        """ --------------------------- Meta-training Graph ---------------------------------- """
        nn_input_per_task = tf.split(self.nn_input, self.meta_batch_size, axis=0)
        delta_per_task = tf.split(self.delta_ph, self.meta_batch_size, axis=0)

        pre_input_per_task, post_input_per_task = zip(*[tf.split(nn_input, 2, axis=0) for nn_input in nn_input_per_task])
        pre_delta_per_task, post_delta_per_task = zip(*[tf.split(delta, 2, axis=0) for delta in delta_per_task])
        
        pre_input_per_task_batched = tf.stack(pre_input_per_task)
        post_input_per_task_batched = tf.stack(post_input_per_task)
        pre_delta_per_task_batched = tf.stack(pre_delta_per_task)
        post_delta_per_task_batched = tf.stack(post_delta_per_task)


        self._adapted_params = []

        
        with tf.variable_scope(name + '/pre_model', reuse=tf.AUTO_REUSE):
            pre_mlp = MLP(name,
                            output_dim=obs_space_dims,
                            hidden_sizes=hidden_sizes,
                            hidden_nonlinearity=hidden_nonlinearity,
                            output_nonlinearity=output_nonlinearity,
                            input_var=pre_input_per_task_batched,
                            input_dim=obs_space_dims + action_space_dims,
                            params=mlp.get_params())

            pre_delta_pred_batched = pre_mlp.output_var
            
            self.delta_diff = pre_delta_per_task_batched - pre_delta_pred_batched
            self.squared_diff = tf.square(self.delta_diff)
            pre_losses = tf.reduce_mean(self.squared_diff, axis=[1,2])
            adapted_params = self._adapt_sym(pre_losses, pre_mlp.get_params())
            self._adapted_params = adapted_params

        with tf.variable_scope(name + '/post_model', reuse=tf.AUTO_REUSE):
            post_mlp = MLP(name,
                            output_dim=obs_space_dims,
                            hidden_sizes=hidden_sizes,
                            hidden_nonlinearity=hidden_nonlinearity,
                            output_nonlinearity=output_nonlinearity,
                            input_var=post_input_per_task_batched,
                            params=adapted_params,
                            input_dim=obs_space_dims + action_space_dims)
            post_delta_pred_batched = post_mlp.output_var

            
            post_losses = tf.reduce_mean(tf.square(post_delta_per_task_batched - post_delta_pred_batched), axis=[1,2])


            self.pre_loss = tf.reduce_mean(pre_losses)
            
            self.post_loss = tf.reduce_mean(post_losses)
            self.train_op = optimizer(self.learning_rate).minimize(self.post_loss)

        """ --------------------------- Post-update Inference Graph --------------------------- """
        with tf.variable_scope(name + '_ph_graph'):
            self.post_update_delta = []
            self.network_phs_meta_batch = []
            
            # nn_input_per_task = tf.split(self.nn_input, self._adapted_models_num, axis=0)
            # nn_input_per_task_batched = tf.stack(nn_input_per_task)

            self.post_update_obs_ph = tf.placeholder(tf.float32, shape = (self.num_rollouts, None, obs_space_dims))
            self.post_update_act_ph = tf.placeholder(tf.float32, shape = (self.num_rollouts, None, action_space_dims))

            self.post_update_nn_input = tf.concat([self.post_update_obs_ph, self.post_update_act_ph], axis=2)


            with tf.variable_scope('task'):
                self.network_phs_meta_batch = self._create_placeholders_for_vars(mlp.get_params())

                mlp_meta_batch = MLP(name,
                                        output_dim=obs_space_dims,
                                        hidden_sizes=hidden_sizes,
                                        hidden_nonlinearity=hidden_nonlinearity,
                                        output_nonlinearity=output_nonlinearity,
                                        params=self.network_phs_meta_batch,
                                        input_var=self.post_update_nn_input,
                                        input_dim=obs_space_dims + action_space_dims,
                                        )

                self.post_update_delta = mlp_meta_batch.output_var

        self._networks = [mlp]

    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False):

        # Account for environments that have action spaces that are one-dimensional, such as CartPole.
        if act.ndim == 2 and act.shape[0] == self.num_rollouts and act.shape[1] == (self.max_path_length - 1):
            act = np.reshape(act, (act.shape[0], act.shape[1], 1))
        elif act.ndim == 1 and (isinstance(act[0].shape, tuple) and len(act[0].shape) == 1):
            for i in range(len(act)):
                act[i] = np.reshape(act[i], (act[i].shape[0], 1))

            
        assert (obs.ndim == 3 and obs.shape[2] == self.obs_space_dims) or (obs.ndim == 1 and obs[0].shape[1] == self.obs_space_dims)
        assert (obs_next.ndim == 3 and obs_next.shape[2] == self.obs_space_dims) or (obs_next.ndim == 1 and obs_next[0].shape[1] == self.obs_space_dims)
        assert (act.ndim == 3 and act.shape[2] == self.action_space_dims) or (act.ndim == 1 and act[0].shape[1] == self.action_space_dims)

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.get_default_session()

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(obs, act, obs_next)

        if self.normalize_input:
            # Normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert (obs.ndim == act.ndim == obs_next.ndim == 3) or (obs.ndim == act.ndim == delta.ndim == 1 and obs[0].shape[0] == act[0].shape[0] == delta[0].shape[0])
        else:
            delta = obs_next - obs

        # Split into valid and test set
        obs_train, act_train, delta_train, obs_test, act_test, delta_test = train_test_split(obs, act, delta,
                                                                                             test_split_ratio=valid_split_ratio)
        if self._dataset_test is None:
            self._dataset_test = dict(obs=obs_test, act=act_test, delta=delta_test)
            self._dataset_train = dict(obs=obs_train, act=act_train, delta=delta_train)
        else:
            # If earlier rollouts did not all have the same lengths,
            # but the newest data does.
            if (len(self._dataset_test['obs'].shape) == 1 and len(obs_test.shape) == 3):
                self._dataset_test['obs'] = np.array(list(self._dataset_test['obs']) + list(obs_test))
                self._dataset_test['act'] = np.array(list(self._dataset_test['act']) + list(act_test))
                self._dataset_test['delta'] = np.array(list(self._dataset_test['delta']) + list(delta_test))

                self._dataset_train['obs'] = np.array(list(self._dataset_train['obs']) + list(obs_train))
                self._dataset_train['act'] = np.array(list(self._dataset_train['act']) + list(act_train))
                self._dataset_train['delta'] = np.array(list(self._dataset_train['delta']) + list(delta_train))
            else:
                self._dataset_test['obs'] = np.concatenate([self._dataset_test['obs'], obs_test])
                self._dataset_test['act'] = np.concatenate([self._dataset_test['act'], act_test])
                self._dataset_test['delta'] = np.concatenate([self._dataset_test['delta'], delta_test])

                self._dataset_train['obs'] = np.concatenate([self._dataset_train['obs'], obs_train])
                self._dataset_train['act'] = np.concatenate([self._dataset_train['act'], act_train])
                self._dataset_train['delta'] = np.concatenate([self._dataset_train['delta'], delta_train])

        valid_loss_rolling_average = None
        epoch_times = []

        """ ------- Looping over training epochs ------- """
        if len(self._dataset_train['obs'].shape) == 1:
            num_timesteps_train = sum([len(o) for o in self._dataset_train['obs']])
            num_steps_per_epoch = max(int(num_timesteps_train/(self.meta_batch_size * self.batch_size * 2)), 1)

            num_timesteps_test = sum([len(o) for o in self._dataset_test['obs']])
            num_steps_test = max(int(num_timesteps_test/(self.meta_batch_size * self.batch_size * 2)), 1)
        else:
            num_steps_per_epoch = max(int(np.prod(self._dataset_train['obs'].shape[:2])
                                    / (self.meta_batch_size * self.batch_size * 2)), 1)
            num_steps_test = max(int(np.prod(self._dataset_test['obs'].shape[:2])
                                 / (self.meta_batch_size * self.batch_size * 2)), 1)

        for epoch in range(epochs):

            # preparations for recording training stats
            pre_batch_losses = []
            post_batch_losses = []
            t0 = time.time()

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            for _ in range(num_steps_per_epoch):
                obs_batch, act_batch, delta_batch = self._get_batch(train=True)

                start_time = time.time()
                pre_batch_loss, post_batch_loss, _ = sess.run([self.pre_loss, self.post_loss, self.train_op],
                                                               feed_dict={self.obs_ph: obs_batch,
                                                               self.act_ph: act_batch,
                                                               self.delta_ph: delta_batch})

                print(f"ONE STEP: {time.time() - start_time}")

                pre_batch_losses.append(pre_batch_loss)
                post_batch_losses.append(post_batch_loss)

            valid_losses = []
            for _ in range(num_steps_test):
                obs_test, act_test, delta_test = self._get_batch(train=False)

                # compute validation loss
                feed_dict = {self.obs_ph: obs_test,
                             self.act_ph: act_test,
                             self.delta_ph: delta_test}
                valid_loss = sess.run(self.loss, feed_dict=feed_dict)
                valid_losses.append(valid_loss)

            valid_loss = np.mean(valid_losses)
            if valid_loss_rolling_average is None:
                valid_loss_rolling_average = 1.5 * valid_loss  # set initial rolling to a higher value avoid too early stopping
                valid_loss_rolling_average_prev = 2 * valid_loss
                if valid_loss < 0:
                    valid_loss_rolling_average = valid_loss/1.5  # set initial rolling to a higher value avoid too early stopping
                    valid_loss_rolling_average_prev = valid_loss/2

            valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average \
                                         + (1.0-rolling_average_persitency)*valid_loss

            epoch_times.append(time.time() - t0)

            if verbose:
                logger.log("Training DynamicsModel - finished epoch %i - "
                           "train loss: %.4f   valid loss: %.4f   valid_loss_mov_avg: %.4f   epoch time: %.2f"
                           % (epoch, np.mean(post_batch_losses), valid_loss, valid_loss_rolling_average,
                              time.time() - t0))

            if valid_loss_rolling_average_prev < valid_loss_rolling_average:
                logger.log('Stopping Training of Model since its valid_loss_rolling_average increased')
                break

            if epoch == epochs - 1:
                logger.log('Stopping Training of Model since it reached max epochs')
                break

            valid_loss_rolling_average_prev = valid_loss_rolling_average

        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.logkv('AvgModelEpochTime', np.mean(epoch_times))
            logger.logkv('Post-Loss', np.mean(post_batch_losses))
            logger.logkv('Pre-Loss', np.mean(pre_batch_losses))
            logger.logkv('Epochs', epoch)

    def predict(self, obs, act):

        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            delta = np.array(self._predict(obs, act))
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta = np.array(self._predict(obs, act))

        assert delta.ndim == 2
        pred_obs = obs_original + delta

        return pred_obs

    
    def set_inference_model_weights(self):
        # it = iter(self._adapted_param_values)
        # weights = []

        # for k, b in zip(it, it):
        #     kernels = self._adapted_param_values[k]
        #     biases = self._adapted_param_values[b]

        #     for i in range(len(kernels)):
        #         weights.append(kernels[i])
        #         weights.append(biases[i])

        # self.inference_model.set_weights(weights)
        it = iter(self._adapted_param_values)

        weights = list(chain.from_iterable((self._adapted_param_values[k][i], self._adapted_param_values[b][i])
                                            for k, b in zip(it, it) for i in range(len(self._adapted_param_values[k]))))

        self.inference_model.set_weights(weights)




    def _predict(self, obs, act):
        if self._adapted_param_values is not None:
            
            all_inputs = np.concatenate([obs, act], axis=1)
            all_inputs = np.split(all_inputs, self._num_adapted_models, axis=0)
            # start_time = time.time()
            delta_new = self.inference_model.predict(all_inputs)
            delta_new = np.concatenate(delta_new, axis=0)
            delta = delta_new
            # print(f"Inference model: {time.time() - start_time}")
            # print('ADAPTED')
            # sess = tf.get_default_session()
            # # obs, act = self._pad_inputs(obs, act)
            # obs = np.reshape(obs, (self._num_adapted_models, -1, obs.shape[-1]))
            # act = np.reshape(act, (self._num_adapted_models, -1, act.shape[-1]))
            # feed_dict = {self.post_update_obs_ph: obs, self.post_update_act_ph: act}
            # feed_dict.update(self.network_params_feed_dict)
            # start_time = time.time()
            # delta = sess.run(self.post_update_delta[:self._num_adapted_models], feed_dict=feed_dict)
            # print(time.time() - start_time)
            # delta = np.concatenate(delta, axis=0)
        else:
            # print('NON-ADAPTED')
            # start_time = time.time()
            delta = self.f_delta_pred(obs, act)
            # print(time.time() - start_time)
        return delta

    def _pad_inputs(self, obs, act, obs_next=None):
        if self._num_adapted_models < self.meta_batch_size:
            pad = int(obs.shape[0] / self._num_adapted_models * (self.meta_batch_size - self._num_adapted_models))
            obs = np.concatenate([obs, np.zeros((pad,) + obs.shape[1:])], axis=0)
            act = np.concatenate([act, np.zeros((pad,) + act.shape[1:])], axis=0)
            if obs_next is not None:
                obs_next = np.concatenate([obs_next, np.zeros((pad,) + obs_next.shape[1:])], axis=0)

        if obs_next is not None:
            return obs, act, obs_next
        else:
            return obs, act

    def adapt(self, obs, act, obs_next):
        self._num_adapted_models = len(obs)
        assert len(obs) == len(act) == len(obs_next)
        obs = np.concatenate([np.concatenate([ob, np.zeros_like(ob)], axis=0) for ob in obs], axis=0)
        act = np.concatenate([np.concatenate([a, np.zeros_like(a)], axis=0) for a in act], axis=0)
        obs_next = np.concatenate([np.concatenate([ob, np.zeros_like(ob)], axis=0) for ob in obs_next], axis=0)

        obs, act, obs_next = self._pad_inputs(obs, act, obs_next)
        assert obs.shape[0] == act.shape[0] == obs_next.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims

        if self.normalize_input:
            # Normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert obs.ndim == act.ndim == obs_next.ndim == 2
        else:
            delta = obs_next - obs

        self._prev_params = [nn.get_param_values() for nn in self._networks]

        sess = tf.get_default_session()
        
        
        # Get all adapted networks.
        # start_time = time.time()
        adapted_param_values_all = sess.run(self._adapted_params,
                                              feed_dict={self.obs_ph: obs, self.act_ph: act, self.delta_ph: delta})
        # print(f"GETTING ADAPTED PARAMS: {time.time() - start_time}")

        # For each layer type, get only the first num_adapted_models of layers.
        self._adapted_param_values = OrderedDict()
        for key, value in adapted_param_values_all.items():
            self._adapted_param_values[key] = value[:self._num_adapted_models]

        # start_time = time.time()
        self.set_inference_model_weights()
        # print(f"ADAPTTIME: {time.time() - start_time}")
        

    def switch_to_pre_adapt(self):
        if self._prev_params is not None:
            [nn.set_params(params) for nn, params in zip(self._networks, self._prev_params)]
            self._prev_params = None
            self._adapted_param_values = None

    def _get_batch(self, train=True):
        if train:
            # If not all rollouts have the same length
            if len(self._dataset_train['obs'].shape) == 1:
                # Select only the paths with a sufficient number of timesteps,
                # and place only those back in the training set.
                path_lengths = np.array([len(o) for o in self._dataset_train['obs']])
                sufficiently_long_paths_indices = np.where(path_lengths >= self.batch_size * 2)
                sufficiently_long_obs = self._dataset_train['obs'][sufficiently_long_paths_indices]
                sufficiently_longs_acts = self._dataset_train['act'][sufficiently_long_paths_indices]
                sufficiently_long_deltas = self._dataset_train['delta'][sufficiently_long_paths_indices]
                
                assert sufficiently_long_paths_indices[0].size > 0, "No paths are longer than 2*batch_size"

                self._dataset_train = dict(obs=sufficiently_long_obs, 
                                           act=sufficiently_longs_acts,
                                           delta=sufficiently_long_deltas)
                
                num_paths = len(sufficiently_long_paths_indices[0])
                path_lengths = path_lengths[sufficiently_long_paths_indices]
                idx_path = np.random.randint(0, num_paths, size=self.meta_batch_size)
                idx_batch = [np.random.randint(self.batch_size, len_path - self.batch_size) if len_path > self.batch_size * 2 else self.batch_size for len_path in path_lengths[idx_path]]

                obs_batch = np.concatenate([self._dataset_train['obs'][ip][ib - self.batch_size:ib + self.batch_size] for ip, ib in zip(idx_path, idx_batch)], axis=0)
                act_batch = np.concatenate([self._dataset_train['act'][ip][ib - self.batch_size:ib + self.batch_size] for ip, ib in zip(idx_path, idx_batch)], axis=0)
                delta_batch = np.concatenate([self._dataset_train['delta'][ip][ib - self.batch_size:ib + self.batch_size] for ip, ib in zip(idx_path, idx_batch)], axis=0)





            else: # If all rollouts have the same length
                num_paths, len_path = self._dataset_train['obs'].shape[:2]
                idx_path = np.random.randint(0, num_paths, size=self.meta_batch_size)
                idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size)

                obs_batch = np.concatenate([self._dataset_train['obs'][ip,
                                            ib - self.batch_size:ib + self.batch_size, :]
                                            for ip, ib in zip(idx_path, idx_batch)], axis=0)
                act_batch = np.concatenate([self._dataset_train['act'][ip,
                                            ib - self.batch_size:ib + self.batch_size, :]
                                            for ip, ib in zip(idx_path, idx_batch)], axis=0)
                delta_batch = np.concatenate([self._dataset_train['delta'][ip,
                                            ib - self.batch_size:ib + self.batch_size, :]
                                            for ip, ib in zip(idx_path, idx_batch)], axis=0)

        else:
            # If not all rollouts have the same length
            if len(self._dataset_test['obs'].shape) == 1:
                # Select only the paths with a sufficient number of timesteps,
                # and place only those back in the training set.
                dataset_test = deepcopy(self._dataset_test)
                path_lengths = np.array([len(o) for o in dataset_test['obs']])
                sufficiently_long_paths_indices = np.where(path_lengths >= self.batch_size * 2)
                sufficiently_long_obs = dataset_test['obs'][sufficiently_long_paths_indices]
                sufficiently_longs_acts = dataset_test['act'][sufficiently_long_paths_indices]
                sufficiently_long_deltas = dataset_test['delta'][sufficiently_long_paths_indices]
                
                # If there are no rollouts with sufficient lengths,
                # then pad them with zeros until they are long enough.
                # This is not a perfect solution, but it will likely
                # only affect the first epoch.
                while sufficiently_long_paths_indices[0].size == 0:
                    for i in range(len(dataset_test['obs'])):
                        obs = dataset_test['obs'][i]
                        act = dataset_test['act'][i]
                        delta = dataset_test['delta'][i]

                        dataset_test['obs'][i] = np.concatenate([obs, np.zeros_like(obs)], axis=0)
                        dataset_test['act'][i] = np.concatenate([act, np.zeros_like(act)], axis=0)
                        dataset_test['delta'][i] = np.concatenate([delta, np.zeros_like(delta)], axis=0)
                    
                    path_lengths = np.array([len(o) for o in dataset_test['obs']])
                    sufficiently_long_paths_indices = np.where(path_lengths >= self.batch_size * 2)
                    sufficiently_long_obs = dataset_test['obs'][sufficiently_long_paths_indices]
                    sufficiently_longs_acts = dataset_test['act'][sufficiently_long_paths_indices]
                    sufficiently_long_deltas = dataset_test['delta'][sufficiently_long_paths_indices]

                assert sufficiently_long_paths_indices[0].size > 0, "No paths are longer than 2*batch_size"

                
                
                idx_path = random.choices(sufficiently_long_paths_indices[0], k=self.meta_batch_size)
                idx_batch = [np.random.randint(self.batch_size, len_path - self.batch_size) if len_path > self.batch_size * 2 else self.batch_size for len_path in path_lengths[idx_path]]

                obs_batch = np.concatenate([dataset_test['obs'][ip][ib - self.batch_size:ib + self.batch_size] for ip, ib in zip(idx_path, idx_batch)], axis=0)
                act_batch = np.concatenate([dataset_test['act'][ip][ib - self.batch_size:ib + self.batch_size] for ip, ib in zip(idx_path, idx_batch)], axis=0)
                delta_batch = np.concatenate([dataset_test['delta'][ip][ib - self.batch_size:ib + self.batch_size] for ip, ib in zip(idx_path, idx_batch)], axis=0)

            
            else: # If all rollouts have the same length
                num_paths, len_path = self._dataset_test['obs'].shape[:2]
                idx_path = np.random.randint(0, num_paths, size=self.meta_batch_size)
                idx_batch = np.random.randint(self.batch_size, len_path - self.batch_size, size=self.meta_batch_size)

                obs_batch = np.concatenate([self._dataset_test['obs'][ip,
                                            ib - self.batch_size:ib + self.batch_size, :]
                                            for ip, ib in zip(idx_path, idx_batch)], axis=0)
                act_batch = np.concatenate([self._dataset_test['act'][ip,
                                            ib - self.batch_size:ib + self.batch_size, :]
                                            for ip, ib in zip(idx_path, idx_batch)], axis=0)
                delta_batch = np.concatenate([self._dataset_test['delta'][ip,
                                            ib - self.batch_size:ib + self.batch_size, :]
                                            for ip, ib in zip(idx_path, idx_batch)], axis=0)
                
        return obs_batch, act_batch, delta_batch

    def _normalize_data(self, obs, act, obs_next=None):
        obs_normalized = normalize(obs, self.normalization['obs'][0], self.normalization['obs'][1])

        # If the actions are discrete, they should not be normalized.
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            actions_normalized = act
        else:
            actions_normalized = normalize(act, self.normalization['act'][0], self.normalization['act'][1])

        if obs_next is not None:
            delta = obs_next - obs
            deltas_normalized = normalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
            return obs_normalized, actions_normalized, deltas_normalized
        else:
            return obs_normalized, actions_normalized
        

    def compute_normalization(self, obs, act, obs_next):
        assert obs.shape[0] == obs_next.shape[0] == act.shape[0]

        # If not all rollouts have the same lengths.
        if len(obs.shape) == 1:
            all_the_same = True
            # Check if the obs, act, and obs_next arrays of a
            # rollout have the same length.
            for o, a, o_n in zip(obs, act, obs_next):
                if not(o.shape[0] == a.shape[0] == o_n.shape[0]):
                    all_the_same = False
                    break

            assert all_the_same

            # Concatenate all timesteps.
            obs_concat = np.concatenate(obs, axis = 0)
            act_concat = np.concatenate(act, axis = 0)
            obs_next_concat = np.concatenate(obs_next, axis = 0)

            delta_concat = obs_next_concat - obs_concat

            # store means and std in dict
            self.normalization = OrderedDict()
            self.normalization['obs'] = (np.mean(obs_concat, axis=0), np.std(obs_concat, axis=0))
            self.normalization['delta'] = (np.mean(delta_concat, axis=0), np.std(delta_concat, axis=0))
            self.normalization['act'] = (np.mean(act_concat, axis=0), np.std(act_concat, axis=0))
        else:
            assert obs.shape[1] == obs_next.shape[1] == act.shape[1]
            delta = obs_next - obs

            assert delta.ndim == 3 and delta.shape[2] == obs_next.shape[2] == obs.shape[2]

            # store means and std in dict
            self.normalization = OrderedDict()
            self.normalization['obs'] = (np.mean(obs, axis=(0, 1)), np.std(obs, axis=(0, 1)))
            self.normalization['delta'] = (np.mean(delta, axis=(0, 1)), np.std(delta, axis=(0, 1)))
            self.normalization['act'] = (np.mean(act, axis=(0, 1)), np.std(act, axis=(0, 1)))

    def _adapt_sym(self, loss, params_var):
        update_param_keys = list(params_var.keys())
        
        def compute_gradients(loss):
            return tf.gradients(loss, [params_var[key] for key in update_param_keys])
        
        grads = tf.vectorized_map(compute_gradients, loss)
        gradients = dict(zip(update_param_keys, grads))

        # Gradient descent
        adapted_policy_params = [params_var[key] - tf.multiply(self.inner_learning_rate, gradients[key])
                          for key in update_param_keys]

        adapted_policy_params_dict = OrderedDict(zip(update_param_keys, adapted_policy_params))

        return adapted_policy_params_dict


    def _create_placeholders_for_vars(self, vars):
        placeholders = OrderedDict()
        for key, var in vars.items():
            placeholders[key] = tf.placeholder(tf.float32, shape=(self.num_rollouts,) + tuple(var.shape), name=key + '_ph')
        return OrderedDict(placeholders)

    @property
    def network_params_feed_dict(self):
        return dict(list((self.network_phs_meta_batch[key], self._adapted_param_values[key])
                         for key in self._adapted_param_values.keys()))

    
    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        state['normalization'] = self.normalization
        state['networks'] = [nn.__getstate__() for nn in self._networks]
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.normalization = state['normalization']
        for i in range(len(self._networks)):
            self._networks[i].__setstate__(state['networks'][i])


def normalize(data_array, mean, std):
    # If not all rollouts have the same lengths.
    if len(data_array.shape) == 1:
        data_array = np.array([(data_array[i] - mean) / (std + 1e-10) for i in range(len(data_array))])
    else:    
        data_array = (data_array - mean) / (std + 1e-10)

    return data_array


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean


def train_test_split(obs, act, delta, test_split_ratio=0.2):
    assert obs.shape[0] == act.shape[0] == delta.shape[0]
    dataset_size = obs.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1-test_split_ratio))

    idx_train = indices[:split_idx]
    idx_test = indices[split_idx:]
    assert len(idx_train) + len(idx_test) == dataset_size

    # If not all rollouts have the same lengths.
    if len(obs.shape) == 1:
        return obs[idx_train], act[idx_train], delta[idx_train], \
               obs[idx_test], act[idx_test], delta[idx_test]

    return obs[idx_train, :], act[idx_train, :], delta[idx_train, :], \
           obs[idx_test, :], act[idx_test, :], delta[idx_test, :]
