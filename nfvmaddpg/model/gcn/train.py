from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import sys
import os
parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)
from gcn.utils import *
from gcn.models import GCN, MLP, BGCN

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags  # 用于接受从终端传入的命令行参数
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'bgcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
adj, features, line_adj, edge_features, pm = load_data(FLAGS.dataset) # shape (12,12) (12,24) (27,27) (27,2)
# Some preprocessing
features = preprocess_features(features) # coords, values, shape ((12,2),(12,),(12,24))
edge_features = preprocess_features(edge_features) # 27 2
pm = process_pm(pm)
# print(features[2])
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)] # coords, values, shape ((42,2),(42,),(12,12))
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
elif FLAGS.model == 'bgcn':
    # coords, values, shape ((42,2),(42,),(12,12))
    support = [preprocess_adj(adj)]
    line_support = [preprocess_adj(line_adj)]
    num_supports = 1
    model_func = BGCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
if FLAGS.model == 'bgcn':
    placeholders = {
        # 1
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        # (12,1)
        'pm': tf.sparse_placeholder(tf.float32),
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'line_support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        # (12,1)
        'edge_features': tf.sparse_placeholder(tf.float32, shape=tf.constant(edge_features[2], dtype=tf.int64)),

        # 'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        # 'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        # helper variable for sparse dropout
        'num_features_nonzero': tf.placeholder(tf.int32),
        'num_edgefeatures_nonzero': tf.placeholder(tf.int32),
    }
else:
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # 1
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)), # (12,1)
        # 'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        # 'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

# Create model
model = model_func(
    placeholders, input_dim=features[2][1], edge_input_dim=edge_features[2][1], logging=True)  # 1

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    if FLAGS.model == 'bgcn':
        feed_dict.update({placeholders['edge_features']: edge_features})
        feed_dict.update({placeholders['pm']: pm})
        feed_dict.update(
            {placeholders['num_edgefeatures_nonzero']: edge_features[1].shape})
        feed_dict.update(
            {placeholders['line_support'][i]: line_support[i] for i in range(len(line_support))})

    # Training step
    # outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    outs, shape = sess.run(
        [model.outputs, tf.shape(model.outputs)], feed_dict=feed_dict)

    # # Validation
    # cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    # cost_val.append(cost)

    # Print results
    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
    #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
    #       "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    print("Epoch:", '%04d' % (epoch + 1), "out=", "{}".format(outs), "shape=", "({})".format(shape))  # shape 12 16

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")

# # Testing
# test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
