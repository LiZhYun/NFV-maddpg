from nfvmaddpg.model.gcn.layers import *
from nfvmaddpg.model.gcn.metrics import *
from nfvmaddpg.model.gcn.utils import *
from tensorflow.python.framework import sparse_tensor

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        # self._loss()
        # self._accuracy()

        # self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, edge_input_dim=1, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, edge_input_dim=1, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']  # (12,1)
        self.input_dim = input_dim  # 1
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.hidden1  # 16
        # self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,  # 1
                                            output_dim=FLAGS.hidden1,  # 16
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim, # 16
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class BGCN(Model):
    def __init__(self, placeholders, input_dim, edge_input_dim=1, **kwargs):
        super(BGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']  # (12,24)
        self.edge_inputs = placeholders['edge_features']
        self.pm = placeholders['pm']
        self.input_dim = input_dim  # 1
        self.edge_inputs_dim = edge_input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.hidden1  # 16
        # self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.line_layers = []
        self.line_activations = []

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim + FLAGS.hidden1,  # 1 + 8
                                            output_dim=FLAGS.hidden1,  # 8
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1 + self.output_dim, # 8 + 8
                                            output_dim=self.output_dim,  # 8
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            logging=self.logging))

        self.line_layers.append(EdgeGraphConvolution(input_dim=self.edge_inputs_dim,  # 2
                                            output_dim=FLAGS.hidden1,  # 8
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.line_layers.append(EdgeGraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,  # 8
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            logging=self.logging))

    
    def build(self):         
        """ Wrapper for _build() """ 
        with tf.variable_scope(self.name):             
            self._build()          # Build sequential layer model         
            self.activations.append(self.inputs)
            self.line_activations.append(self.edge_inputs)         
            for layer, line_layer in zip(self.layers, self.line_layers):
                line_hidden = line_layer(self.line_activations[-1])
                self.line_activations.append(line_hidden)
                tran_tensor = dot(self.pm, line_hidden, sparse=True)
                if layer.sparse_inputs:
                    line_idx = tf.where(tf.not_equal(tran_tensor, 0))
                    # if not isinstance(self.activations[-1], sparse_tensor.SparseTensor):
                    #     idx = tf.where(tf.not_equal(self.activations[-1], 0))
                    #     self.activations[-1] = tf.SparseTensor(
                    #         idx, tf.gather_nd(self.activations[-1], idx), tf.shape(self.activations[-1],out_type=tf.int64))
                    hidden = layer(
                        # tf.concat(axis=1, values=[self.activations[-1], tf.SparseTensor(sparse_to_tuple(coo_matrix(dot(self.pm, line_hidden, sparse=True))))]))
                        tf.sparse_concat(axis=1, sp_inputs=[self.activations[-1], tf.SparseTensor(line_idx, tf.gather_nd(tran_tensor, line_idx), tf.shape(tran_tensor, out_type=tf.int64))]))
                else:
                    hidden = layer(
                        tf.concat(axis=-1, values=[self.activations[-1], tran_tensor]))
                self.activations.append(hidden)
            self.outputs = self.activations[-1]          # Store model variables for easy access  
            self.outputs = graph_aggregation_layer('GAL', self.outputs, 4, True, activation=tf.nn.tanh)
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)         
            self.vars = {var.name: var for var in variables}

    def predict(self):
        return tf.nn.softmax(self.outputs)


