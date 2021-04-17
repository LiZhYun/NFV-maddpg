import tensorflow as tf


def graph_convolution_layer(inputs, units, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs
    adj = tf.transpose(adjacency_tensor[:, :, :], (0, 1, 2)) # shape 1212 去掉了类型为zero的

    annotations = tf.concat((hidden_tensor, node_tensor), -1) if hidden_tensor is not None else node_tensor # 12 12

    # output = tf.stack([tf.layers.dense(inputs=annotations, units=units) for _ in range(adj.shape[1])], 1) # shape 1 12 128
    output = tf.layers.dense(inputs=annotations, units=units) # shape 12 128

    output = tf.matmul(adj, output) # 12 128
    output = output + tf.layers.dense(inputs=annotations, units=units)  # 作为层之间的自连接 shape 12 128
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


# 在通过图卷积进行几层传播之后，Li等人（2016）将节点嵌入聚合到图级表示向量中
def graph_aggregation_layer(inputs, units, training, activation=None, dropout_rate=0.):
    i = tf.layers.dense(inputs, units=units, activation=tf.nn.sigmoid) # 12 64
    j = tf.layers.dense(inputs, units=units, activation=activation) # 12 64
    output = tf.reduce_sum(i * j, 1) # 64
    output = activation(output) if activation is not None else output
    output = tf.layers.dropout(output, dropout_rate, training=training)

    return output


def multi_dense_layers(inputs, units, training, activation=None, dropout_rate=0.):
    hidden_tensor = inputs
    shortcut = inputs
    for u in units:
        hidden_tensor = tf.layers.dense(hidden_tensor, units=u, activation=activation)
        hidden_tensor = tf.layers.batch_normalization(
            hidden_tensor, training=training)
        hidden_tensor = tf.layers.dropout(hidden_tensor, dropout_rate, training=training)
    
    shortcut = tf.layers.dense(shortcut, units=units[-1], activation=None)
    hidden_tensor = tf.add(shortcut, hidden_tensor)
    return tf.nn.relu(hidden_tensor)


def multi_graph_convolution_layers(inputs, units, training, activation=None, dropout_rate=0.):
    adjacency_tensor, hidden_tensor, node_tensor = inputs  # 1212  none 1212
    for u in units:# 128 64
        hidden_tensor = graph_convolution_layer(inputs=(adjacency_tensor, hidden_tensor, node_tensor),
                                                units=u, activation=activation, dropout_rate=dropout_rate,
                                                training=training)  # 上一层的gcn结果concat到下一层的node特征中12 128

    return hidden_tensor  # shape 12 64
