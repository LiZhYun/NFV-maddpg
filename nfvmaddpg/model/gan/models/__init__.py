import tensorflow as tf
from nfvmaddpg.model.gan.utils.layers import multi_graph_convolution_layers, graph_aggregation_layer, multi_dense_layers


def encoder_rgcn(inputs, units, training, dropout_rate=0.):
    graph_convolution_units, auxiliary_units = units #  128 64   64

    with tf.variable_scope('graph_convolutions'):
        output = multi_graph_convolution_layers(inputs, graph_convolution_units, activation=tf.nn.tanh,
                                                dropout_rate=dropout_rate, training=training) # shape 12 64
    output = tf.layers.batch_normalization(
        output, training=training)
    with tf.variable_scope('graph_aggregation'):
        _, hidden_tensor, node_tensor = inputs
        annotations = tf.concat(
            (output, hidden_tensor, node_tensor) if hidden_tensor is not None else (output, node_tensor), -1) # shape 12 76

        output = graph_aggregation_layer(annotations, auxiliary_units, activation=tf.nn.tanh,
                                         dropout_rate=dropout_rate, training=training) # 64

    return output


def decoder_adj(inputs, units, vertexes, edges, nodes, training, dropout_rate=0.): 
    #           8 (128, 256, 512) 12        1      12     false  
    output = multi_dense_layers(inputs, units, activation=tf.nn.relu, dropout_rate=dropout_rate, training=training)  # 全连接层 返回512维
    
    with tf.variable_scope('edges_logits'):
        edges_logits = tf.reshape(multi_dense_layers(inputs=output, units=(512, 256, vertexes * vertexes),  # edges * vertexes * vertexes 代表每种键类型中 那些原子与原子之间的连接为该类键
                                                     activation=tf.nn.relu, dropout_rate=dropout_rate, training=training), (-1, vertexes, vertexes))
        edges_logits = tf.transpose((edges_logits + tf.matrix_transpose(edges_logits)) / 2, (0, 1, 2)) # 成为对称 shape 12,12
        edges_logits = tf.layers.dropout(edges_logits, dropout_rate, training=training)

    with tf.variable_scope('nodes_logits'):
        nodes_logits = multi_dense_layers(
            # 每个节点的类型
            inputs=output, units=(512, 256, vertexes * nodes), activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)
        nodes_logits = tf.reshape(nodes_logits, (-1, vertexes, nodes)) # shape 12,13
        nodes_logits = tf.layers.dropout(nodes_logits, dropout_rate, training=training)

    return edges_logits, nodes_logits


def decoder_dot(inputs, units, vertexes, edges, nodes, training, dropout_rate=0.):
    output = multi_dense_layers(inputs, units[:-1], activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('edges_logits'):
        edges_logits = tf.reshape(tf.layers.dense(inputs=output, units=edges * vertexes * units[-1],
                                                  activation=None), (-1, edges, vertexes, units[-1]))
        edges_logits = tf.transpose(tf.matmul(edges_logits, tf.matrix_transpose(edges_logits)), (0, 2, 3, 1))
        edges_logits = tf.layers.dropout(edges_logits, dropout_rate, training=training)

    with tf.variable_scope('nodes_logits'):
        nodes_logits = tf.layers.dense(inputs=output, units=vertexes * nodes, activation=None)
        nodes_logits = tf.reshape(nodes_logits, (-1, vertexes, nodes))
        nodes_logits = tf.layers.dropout(nodes_logits, dropout_rate, training=training)

    return edges_logits, nodes_logits


def decoder_rnn(inputs, units, vertexes, edges, nodes, training, dropout_rate=0.):
    output = multi_dense_layers(inputs, units[:-1], activation=tf.nn.tanh, dropout_rate=dropout_rate, training=training)

    with tf.variable_scope('edges_logits'):
        edges_logits, _ = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(units[-1] * 4),
                                            inputs=tf.tile(tf.expand_dims(output, axis=1),
                                                           (1, vertexes, 1)), dtype=output.dtype)

        edges_logits = tf.layers.dense(edges_logits, edges * units[-1])
        edges_logits = tf.transpose(tf.reshape(edges_logits, (-1, vertexes, edges, units[-1])), (0, 2, 1, 3))
        edges_logits = tf.transpose(tf.matmul(edges_logits, tf.matrix_transpose(edges_logits)), (0, 2, 3, 1))
        edges_logits = tf.layers.dropout(edges_logits, dropout_rate, training=training)

    with tf.variable_scope('nodes_logits'):
        nodes_logits, _ = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(units[-1] * 4),
                                            inputs=tf.tile(tf.expand_dims(output, axis=1),
                                                           (1, vertexes, 1)), dtype=output.dtype)
        nodes_logits = tf.layers.dense(nodes_logits, nodes)
        nodes_logits = tf.layers.dropout(nodes_logits, dropout_rate, training=training)

    return edges_logits, nodes_logits


def postprocess_logits(inputs, temperature=1.):

    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    softmax = [tf.transpose(tf.nn.softmax(tf.transpose(e_logits, (0,2,1)) / temperature), (0,2,1))
               for e_logits in listify(inputs)]  # shape 1212
    argmax = [tf.one_hot(tf.argmax(e_logits, axis=-1), depth=e_logits.shape[-1], dtype=e_logits.dtype)
              for e_logits in listify(inputs)]    # 转为argmax shape 995
    gumbel_logits = [e_logits - tf.log(- tf.log(tf.random_uniform(tf.shape(e_logits), dtype=e_logits.dtype)))
                     for e_logits in listify(inputs)]   # 加上gumbel noise shape 995
    gumbel_softmax = [tf.transpose(tf.nn.softmax(tf.transpose(e_gumbel_logits, (0, 2, 1)) / temperature), (0, 2, 1))
                      for e_gumbel_logits in gumbel_logits] # shape 995
    gumbel_argmax = [
        tf.one_hot(tf.argmax(e_gumbel_logits, axis=-1), depth=e_gumbel_logits.shape[-1], dtype=e_gumbel_logits.dtype)
        for e_gumbel_logits in gumbel_logits]  # shape 995

    return [delistify(e) for e in (softmax, argmax, gumbel_logits, gumbel_softmax, gumbel_argmax)]
