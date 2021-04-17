from nfvmaddpg.model.gcn.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment



def get_layer_uid(_LAYER_UIDS, layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    # if sparse:
    #     # res = tf.matmul(x, y, a_is_sparse=True)
    #     # x = tf.sparse_reshape()
    #     res = tf.matmul(x, y)
    #     # res = tf.sparse_tensor_dense_matmul(x, y)
    # else:
    res = tf.matmul(x, y)
    return res

def weight_dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    # if sparse:
    #     # res = tf.matmul(x, y, a_is_sparse=True)
    #     # x = tf.sparse_reshape()
    #     res = tf.matmul(x, y)
    #     # res = tf.sparse_tensor_dense_matmul(x, y)
    # else:
    x_shape = x.get_shape().as_list()
    x = tf.reshape(x, (-1, x_shape[2]))
    res = tf.matmul(x, y)
    res_shape = res.get_shape().as_list()
    res = tf.reshape(res, (-1, x_shape[1], res_shape[1]))
    
    return res


def graph_aggregation_layer(scope, inputs, units, training, activation=None, dropout_rate=0.):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        i = tf.layers.dense(inputs, units=units, activation=tf.nn.sigmoid, name='first_layer')  # 9 128
        j = tf.layers.dense(inputs, units=units, activation=activation, name='sec_layer')  # 9 128
        output = tf.reduce_sum(i * j, 1)  # 128
        output = activation(output) if activation is not None else output
        output = tf.layers.dropout(output, dropout_rate, training=training)

        return output


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        _LAYER_UIDS = {}
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(_LAYER_UIDS, layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        # output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs # True
        self.featureless = featureless # False
        self.bias = bias # False

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # 下面是定义变量，主要是通过调用utils.py中的glorot函数实现
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                                    name='weights')
            # for i in range(len(self.support)):
            #     self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
            #                                             name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # convolve 卷积的实现。主要是根据论文中公式Z = \tilde{D}^{-1/2}\tilde{A}^{-1/2}X\theta实现
        supports = list()  # 多个图的情况
        if not self.featureless:
            pre_sup = weight_dot(x, self.vars['weights'],
                            sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights']
        support = dot(self.support, pre_sup, sparse=True)
        supports.append(support)
        # for i in range(len(self.support)):
        #     if not self.featureless:
        #         pre_sup = dot(x, self.vars['weights_' + str(i)],
        #                       sparse=self.sparse_inputs)
        #     else:
        #         pre_sup = self.vars['weights_' + str(i)]
        #     support = dot(self.support[i], pre_sup, sparse=True)
        #     supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class EdgeGraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(EdgeGraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['line_support']
        self.sparse_inputs = sparse_inputs  # True
        self.featureless = featureless  # False
        self.bias = bias  # False

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_edgefeatures_nonzero']

        # 下面是定义变量，主要是通过调用utils.py中的glorot函数实现
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                                        name='weights')
            # for i in range(len(self.support)):
            #     self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
            #                                             name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        # if self.sparse_inputs:
        #     x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        # else:
        x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # convolve 卷积的实现。主要是根据论文中公式Z = \tilde{D}^{-1/2}\tilde{A}^{-1/2}X\theta实现
        supports = list()  # 多个图的情况
        if not self.featureless:
            pre_sup = weight_dot(x, self.vars['weights'],
                            sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights']
        support = dot(self.support, pre_sup, sparse=True)
        supports.append(support)
        # for i in range(len(self.support)):
        #     if not self.featureless:
        #         pre_sup = dot(x, self.vars['weights_' + str(i)],
        #                       sparse=self.sparse_inputs)
        #     else:
        #         pre_sup = self.vars['weights_' + str(i)]
        #     support = dot(self.support[i], pre_sup, sparse=True)
        #     supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)



