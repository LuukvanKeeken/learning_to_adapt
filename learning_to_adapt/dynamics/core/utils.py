import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.initializers import Zeros, glorot_normal


def create_dnn(name,
               output_dim,
               kernel_sizes,
               strides,
               num_filters,
               hidden_nonlinearity,
               output_nonlinearity,
               hidden_dim,
               n_channels,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (tuple): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None
    assert len(kernel_sizes) == len(strides) == len(num_filters)

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    with tf.variable_scope(name):
        x = input_var

        x = tf.reshape(x, [-1, hidden_dim, hidden_dim, n_channels])

        for idx, (kernel_size, stride, filter) in enumerate(zip(kernel_sizes, strides, num_filters)):
            x = tf.layers.conv2d_transpose(x,
                                           filters=filter,
                                           kernel_size=kernel_size,
                                           strides=stride,
                                           name='conv_t_%d' % idx,
                                           activation=hidden_nonlinearity,
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init,
                                           )

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x,
                            np.prod(output_dim),
                            name='output',
                            activation=output_nonlinearity,
                            kernel_initializer=w_init,
                            bias_initializer=b_init,
                            # reuse=reuse,
                            )

        output_var = tf.reshape(x, (-1,) + output_dim)

    return input_var, output_var


def create_keras_mlp(output_dim,
                     hidden_sizes,
                     hidden_nonlinearity,
                     output_nonlinearity,
                     input_dim=None,
                     batch_normalization=False,
                     ):
    """
    Creates a MLP network using tf.keras.Model
    Args:
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        batch_normalization (bool): whether to use batch normalization

    Returns:
        model (tf.keras.Model): A Keras model instance
    """

    assert input_dim is not None

    def structure(input, model_idx):
        x = input

        for idx, hidden_size in enumerate(hidden_sizes):
            if batch_normalization:
                x = BatchNormalization()(x)
            x = Dense(hidden_size,
                    activation=hidden_nonlinearity,
                    kernel_initializer=glorot_normal(),
                    bias_initializer=Zeros(),
                    name=f'model_{model_idx}_hidden_{idx}')(x)

        if batch_normalization:
            x = BatchNormalization()(x)


        output_var = Dense(output_dim,
                       activation=output_nonlinearity,
                       kernel_initializer=glorot_normal(),
                       bias_initializer=Zeros(),
                       name=f'model_{model_idx}_output')(x)

        return output_var

    inputs = []
    outputs = []

    for i in range(5):
        input = Input(shape=input_dim, dtype='float32')
        inputs.append(input)
        output = structure(input, i)
        outputs.append(output)

    return Model(inputs=inputs, outputs=outputs)
    
    
    # input_var = Input(shape=input_dim, dtype='float32')
    

    # output_var = Dense(output_dim,
    #                    activation=output_nonlinearity,
    #                    kernel_initializer=glorot_normal(),
    #                    bias_initializer=Zeros(),
    #                    name='output')(x)

    # # model = Model(inputs=input_var, outputs=output_var)
    # input_vars = [input_var for _ in range(5)]
    # output_vars = [output_var for _ in range(5)]
    # model = Model(inputs=input_vars, outputs=output_vars)

    # return model



def create_mlp(output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               batch_normalization=False,
               reuse=False,
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')

    x = input_var

    for idx, hidden_size in enumerate(hidden_sizes):
        if batch_normalization == 'traning':
            x = tf.layers.batch_normalization(x, training=True)
        elif batch_normalization == 'testing':
            x = tf.layers.batch_normalization(x, training=False)

        x = tf.layers.dense(x,
                            hidden_size,
                            name='hidden_%d' % idx,
                            activation=hidden_nonlinearity,
                            kernel_initializer=w_init,
                            bias_initializer=b_init,
                            reuse=reuse,
                            )

    if batch_normalization == 'traning':
        x = tf.layers.batch_normalization(x, training=True)
    elif batch_normalization == 'testing':
        x = tf.layers.batch_normalization(x, training=False)

    output_var = tf.layers.dense(x,
                                 output_dim,
                                 name='output',
                                 activation=output_nonlinearity,
                                 kernel_initializer=w_init,
                                 bias_initializer=b_init,
                                 reuse=reuse,
                                 )

    return input_var, output_var


def create_rnn(name,
               cell_type,
               output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               state_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               reuse=False,
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None
    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')

    if state_var is None:
        create_hidden = True
    else:
        create_hidden = False

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        cell = []
        if state_var is None:
            state_var = []

        for idx, hidden_size in enumerate(hidden_sizes):
            if cell_type == 'lstm':
                cell.append(tf.nn.rnn_cell.LSTMCell(hidden_size, activation=hidden_nonlinearity))
                if create_hidden:
                    c = tf.placeholder(tf.float32, (None, hidden_size), name='cell_state_%d' % idx)
                    h = tf.placeholder(tf.float32, (None, hidden_size), name='hidden_state_%d' % idx)
                    state_var.append(tf.contrib.rnn.LSTMStateTuple(c, h))
            elif cell_type == 'gru':
                cell.append(tf.nn.rnn_cell.GRUCell(hidden_size, activation=hidden_nonlinearity))
                if create_hidden:
                    h = tf.placeholder(tf.float32, (None, hidden_size), name='hidden_state_%d' % idx)
                    state_var.append(h)
            elif cell_type == 'rnn':
                cell.append(tf.nn.rnn_cell.RNNCell(hidden_size, activation=hidden_nonlinearity))
                if create_hidden:
                    h = tf.placeholder(tf.float32, (None, hidden_size), name='hidden_state_%d' % idx)
                    state_var.append(h)
            else:
                raise NotImplementedError

        if len(hidden_sizes) > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            if create_hidden:
                state_var = tuple(state_var)
        else:
            cell = cell[0]
            if create_hidden:
                state_var = state_var[0]

        outputs, next_state_var = tf.nn.dynamic_rnn(cell,
                                                    input_var,
                                                    initial_state=state_var,
                                                    time_major=False,
                                                    )

        if output_dim is None:
            output_var = outputs
        else:
            output_var = tf.layers.dense(outputs,
                                         output_dim,
                                         name='output',
                                         activation=output_nonlinearity,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init,
                                         )

    return input_var, state_var,  output_var, next_state_var, cell

def tensordot_fn(args):
    x, param = args
    return tf.tensordot(x, param, axes=1)

def forward_mlp(output_dim,
                hidden_sizes,
                hidden_nonlinearity,
                output_nonlinearity,
                input_var,
                mlp_params,
                ):
    """
    Creates the forward pass of an mlp given the input vars and the mlp params. Assumes that the params are passed in
    order i.e. [hidden_0/kernel, hidden_0/bias, hidden_1/kernel, hidden_1/bias, ..., output/kernel, output/bias]
    Args:
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        mlp_params (OrderedDict): OrderedDict of the params of the neural network.

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """
    x = input_var
    idx = 0
    bias_added = False
    sizes = tuple(hidden_sizes) + (output_dim,)
    assert len(mlp_params) > 0

    if output_nonlinearity is None:
        output_nonlinearity = tf.identity

    for name, param in mlp_params.items():
        assert str(idx) in name or (idx == len(hidden_sizes) and "output" in name)

        if "kernel" in name:
            # Distinguish between a singular network and a batch of networks.
            if len(param.shape) == 2:
                assert param.shape == (x.shape[-1], sizes[idx])
                x = tf.tensordot(x, param, axes=[[2], [0]])
                # x_reshaped = tf.reshape(x, [-1, int(x.shape[-1])])
                # result = tf.matmul(x_reshaped, param)
                # x = tf.reshape(result, [int(x.shape[0]), -1, sizes[idx]])
            else:
                param_shape_list = param.shape.as_list()
                x_shape_list = [x.shape[0].value, x.shape[-1].value, sizes[idx]]
                assert param_shape_list == x_shape_list or (param_shape_list[0] is None and x_shape_list[0] is None and param_shape_list[1:] == x_shape_list[1:]), "param has to be a batch of layers, representing multiple networks"
                # assert param.shape == (x.shape[0], x.shape[-1], sizes[idx]) or (param.shape[0] == None and x.shape[0] == None and param.shape[1:] == (x.shape[-1], sizes[idx])), "param has to be a batch of layers, representing multiple networks"
                # x = tf.einsum('abc,acd->abd', x, param) 
                # x = tf.vectorized_map(tensordot_fn, (x, param))
                # x.set_shape([param.shape[0], None, sizes[idx]])
                # x = tf.tensordot(x, param, axes=[[1,2], [1,2]])
                # x_reshaped = tf.reshape(x, [-1, x.shape[-1]])
                # param_reshaped = tf.reshape(param, [param.shape[0]*param.shape[1], -1])
                # result = tf.matmul(x_reshaped, param_reshaped)
                # x = tf.reshape(result, [int(x.shape[0]), int(x.shape[1]), -1])
                # x = tf.matmul(x, param)
        elif "bias" in name:
            if len(param.shape) == 1:
                assert param.shape == (sizes[idx],)
                x = tf.add(x, param)
            else:
                param_shape_list = param.shape.as_list()
                x_shape_list = [x.shape[0].value, sizes[idx]]
                assert param_shape_list == x_shape_list or (param_shape_list[0] is None and x_shape_list[0] is None and param_shape_list[1:] == x_shape_list[1:]), "param has to be a batch of layers, representing multiple networks"
                # assert param.shape == (x.shape[0], sizes[idx]), "param has to be a batch of layers, representing multiple networks"
                x = tf.add(x, tf.expand_dims(param, axis=1))
            bias_added = True
        else:
            raise NameError

        if bias_added:
            if "hidden" in name:
                x = hidden_nonlinearity(x)
            elif "output" in name:
                x = output_nonlinearity(x)
            else:
                raise NameError
            idx += 1
            bias_added = False
    output_var = x
    return input_var, output_var




