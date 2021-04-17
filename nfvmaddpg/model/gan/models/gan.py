import numpy as np
import tensorflow as tf

from nfvmaddpg.model.gan.models import postprocess_logits
from nfvmaddpg.model.gan.utils.layers import multi_dense_layers
from nfvmaddpg.model.transformer.model import Transformer
from nfvmaddpg.model.transformer.data_load import input_fn, calc_num_batches
from nfvmaddpg.model.gcn.models import BGCN

class GraphGANModel(object):

    def __init__(self, arglist, features, edge_features, vertexes, edges, nodes, embedding_dim, decoder_units, discriminator_units,
                 decoder, discriminator, scope, soft_gumbel_softmax=False, hard_gumbel_softmax=False,
                 batch_discriminator=True):
        self.vertexes, self.edges, self.nodes, self.embedding_dim, self.decoder_units, self.discriminator_units, \
        self.decoder, self.discriminator, self.batch_discriminator = vertexes, edges, nodes, embedding_dim, decoder_units, \
                                                                     discriminator_units, decoder, discriminator, batch_discriminator

        self.training = tf.placeholder_with_default(
            False, shape=(), name="training")  # 是否训练placeholder
        self.dropout_rate = tf.placeholder_with_default(0., shape=(), name="dropout_rate") # 丢弃率 placeholder
        self.soft_gumbel_softmax = tf.placeholder_with_default(soft_gumbel_softmax, shape=()) # gumbel softmax  placeholder
        self.hard_gumbel_softmax = tf.placeholder_with_default(hard_gumbel_softmax, shape=()) # gumbel softmax  placeholder
        self.temperature = tf.placeholder_with_default(1., shape=()) # 温度 placeholder

        self.edges_labels = tf.placeholder(dtype=tf.int64, shape=(None, vertexes, vertexes), name="edges_labels")  # 链路带宽 placeholder 12*12
        self.nodes_labels = tf.placeholder(dtype=tf.int64, shape=(None, vertexes, nodes), name="nodes_labels") # 节点包含的usage placeholder 12*12
        # self.embeddings = tf.placeholder(dtype=tf.float32, shape=(None, embedding_dim), name="embeddings") # 嵌入向量 placeholder 8

        self.rewardR = tf.placeholder(dtype=tf.float32, shape=(None, 1))   # 真实RL奖励 placeholder 1
        self.rewardF = tf.placeholder(dtype=tf.float32, shape=(None, 1))   # 假RL奖励 placeholder 1
        self.adjacency_tensor = tf.cast(self.edges_labels, dtype=tf.float32)
        self.node_tensor = tf.cast(self.nodes_labels, dtype=tf.float32)
        # self.adjacency_tensor = tf.one_hot(self.edges_labels, depth=edges, dtype=tf.float32)  # 键类型的onehot向量 shape (12,12,1)
        # self.node_tensor = tf.one_hot(self.nodes_labels, depth=nodes, dtype=tf.float32)   # 原子类型的onehot向量  shape (12,12,12)

        self.noise = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="noise")
        # self.gcn_out = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="gcn_out")
        # self.trans_out = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="transformer_out")

        self.gcn_placeholders = {
        # 1
            'support': tf.placeholder(tf.float32, shape=(None, vertexes, vertexes), name="support"),
            # 'support': tf.sparse_placeholder(tf.float32, shape=(None, vertexes, vertexes), name="support"),
            # 'support': [tf.sparse_placeholder(tf.float32, name="support") for _ in range(arglist.batch_size)],
            # (12,1)
            'pm': tf.placeholder(tf.float32, shape=(None, vertexes, edge_features.shape[0]), name="pm"),
            'features': tf.placeholder(tf.float32, shape=(None,) + features.shape, name="features"),
            'line_support': tf.placeholder(tf.float32, shape=(None, edge_features.shape[0], edge_features.shape[0]), name="line_support"),
            # 'line_support': [tf.sparse_placeholder(tf.float32, name="line_support") for _ in range(arglist.batch_size)],
            # (12,1)
            'edge_features': tf.placeholder(tf.float32, shape=(None,) + edge_features.shape, name="edge_features"),

            # 'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            # 'labels_mask': tf.placeholder(tf.int32),
            'dropout': self.dropout_rate,
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32, name="num_features_nonzero"),
            'num_edgefeatures_nonzero': tf.placeholder(tf.int32, name="num_edgefeatures_nonzero"),
        }

        with tf.variable_scope(scope):

            with tf.variable_scope('gcn', reuse=tf.AUTO_REUSE):
                self.gcn_model = BGCN(self.gcn_placeholders, input_dim=features.shape[1],
                                      edge_input_dim=edge_features.shape[1], logging=True, name='gcn')
                # self.gcn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gcn')
            
            with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
                self.trans_model = Transformer(arglist)
                # self.trans_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transformer')
            mem = self.trans_model.train()
            self.embeddings = tf.concat([self.noise, self.gcn_model.outputs, mem], -1)

            with tf.variable_scope('generator'):
                self.edges_logits, self.nodes_logits = self.decoder(self.embeddings, decoder_units, vertexes, edges, nodes,
                                                                    training=self.training, dropout_rate=self.dropout_rate)   # 返回邻接矩阵（有多个 因为键类型多）以及节点类型
                # shape 9 9 5    shape 9 5
            with tf.name_scope('outputs'):
                (self.edges_softmax, self.nodes_softmax), \
                (self.edges_argmax, self.nodes_argmax), \
                (self.edges_gumbel_logits, self.nodes_gumbel_logits), \
                (self.edges_gumbel_softmax, self.nodes_gumbel_softmax), \
                (self.edges_gumbel_argmax, self.nodes_gumbel_argmax) = postprocess_logits(
                    (self.edges_logits, self.nodes_logits), temperature=self.temperature)   # 得到softmax 和 gumbel softmax

                self.edges_hat = tf.case({self.soft_gumbel_softmax: lambda: self.edges_gumbel_logits,  # 如果为true则使用gumbel softmax
                                        self.hard_gumbel_softmax: lambda: self.edges_gumbel_logits},
                                        default=lambda: self.edges_logits,    # 默认为直接softmax
                                        exclusive=True)
                # self.edges_hat = tf.case({self.soft_gumbel_softmax: lambda: self.edges_gumbel_softmax,  # 如果为true则使用gumbel softmax
                #                           self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                #                               self.edges_gumbel_argmax - self.edges_gumbel_softmax) + self.edges_gumbel_softmax},
                #                          default=lambda: self.edges_softmax,    # 默认为直接softmax
                #                          exclusive=True)

                self.nodes_hat = tf.case({self.soft_gumbel_softmax: lambda: self.nodes_gumbel_softmax,
                                        self.hard_gumbel_softmax: lambda: tf.stop_gradient(
                                            self.nodes_gumbel_argmax - self.nodes_gumbel_softmax) + self.nodes_gumbel_softmax},
                                        default=lambda: self.nodes_gumbel_softmax,
                                        exclusive=True)

            with tf.name_scope('D_x_real'):
                self.logits_real, self.features_real = self.D_x((self.adjacency_tensor, None, self.node_tensor), # 12121 none 1212  
                                                                units=discriminator_units) # output shape 1    64
            with tf.name_scope('D_x_fake'):
                self.logits_fake, self.features_fake = self.D_x((self.edges_hat, None, self.nodes_hat),
                                                                units=discriminator_units)

            with tf.name_scope('V_x_real'):
                self.value_logits_real = self.V_x((self.adjacency_tensor, None, self.node_tensor),
                                                units=discriminator_units) # 1
            with tf.name_scope('V_x_fake'):
                self.value_logits_fake = self.V_x((self.edges_hat, None, self.nodes_hat), units=discriminator_units)

    def D_x(self, inputs, units):  # 判别器
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            outputs0 = self.discriminator(inputs, units=units[:-1], training=self.training,
                                          dropout_rate=self.dropout_rate) # shape 128

            outputs1 = multi_dense_layers(outputs0, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                          dropout_rate=self.dropout_rate) # shape 64
            

            if self.batch_discriminator:
                outputs_batch = tf.layers.dense(outputs0, units[-2] // 8, activation=tf.tanh)
                outputs_batch = tf.layers.dense(tf.reduce_mean(outputs_batch, 0, keep_dims=True), units[-2] // 8,
                                                activation=tf.nn.tanh)
                outputs_batch = tf.tile(outputs_batch, (tf.shape(outputs0)[0], 1))

                outputs1 = tf.concat((outputs1, outputs_batch), -1)

            outputs = tf.layers.dense(outputs1, units=1) # shape 1

        return outputs, outputs1

    def V_x(self, inputs, units):  # 奖励网络 用于近似样本的奖赏函数
        with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
            outputs = self.discriminator(inputs, units=units[:-1], training=self.training,
                                         dropout_rate=self.dropout_rate) # 128

            outputs = multi_dense_layers(outputs, units=units[-1], activation=tf.nn.tanh, training=self.training,
                                         dropout_rate=self.dropout_rate) # 64

            outputs = tf.layers.dense(outputs, units=1, activation=tf.nn.sigmoid) # 1

        return outputs

    def sample_z(self, batch_dim):
        return np.random.normal(0, 1, size=(batch_dim, self.embedding_dim))
