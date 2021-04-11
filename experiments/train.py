import gc
import os, sys
import psutil

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import pickle
import threading
import logging


import nfvmaddpg.common.tf_util as U
from nfvmaddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from nfvmaddpg.model.gan.models.gan import GraphGANModel
from nfvmaddpg.model.gan.models import encoder_rgcn, decoder_adj
from nfvmaddpg.model.gan.utils.sparse_topo_dataset import SparseToPoDataset
from nfvmaddpg.model.gcn.utils import process_pm, preprocess_features, preprocess_adj
from nfvmaddpg.model.gcn.models import BGCN
from nfvmaddpg.model.parser.getrequestlist import requestlist
from nfvmaddpg.model.parser.optimization.parser import Parser
from nfvmaddpg.model.parser.test import posEncode
from nfvmaddpg.model.transformer.model import Transformer
from nfvmaddpg.model.transformer.data_load import input_fn, calc_num_batches, load_vocab, encode


from multiagentnfv.core_nfv import Request
from copy import deepcopy
from itertools import permutations, islice, product, chain

REQ_NUM = 0
Z_DIM = 12

flags = tf.app.flags  # 用于接受从终端传入的命令行参数
FLAGS = flags.FLAGS


flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="nfvtopo_abilene", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=32, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--policy", type=str, default="maddpg", help="policy for agents")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    # Transformer
    parser.add_argument('--vocab_size', default=17, type=int)
    parser.add_argument('--train1', default='src/train_vnf', help="german training segmented data")     
    parser.add_argument('--train2', default='src/train_pos', help="english training segmented data")
    parser.add_argument('--vocab', default='nfvmaddpg/model/transformer/src/vnf.vocab', help="vocabulary file path")
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--d_model', default=4, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=32, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=4, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=50, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=50, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)     

    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def gan_model(scope, arglist, features, edge_features):
    data = SparseToPoDataset() 
    data.load('nfvmaddpg/model/gan/data/vnftopo.sparsedataset')
    model = GraphGANModel(arglist, features, edge_features, data.vertexes,  # 拓扑中最大节点数 9
                          1,  # 键类型数 5
                          data.vnf_num_types,  # 原子种类数目 5
                          Z_DIM,   # 嵌入后的维度 8
                          decoder_units=(32, 64, 128, 256, 512),
                          #   discriminator_units=((128, 64), 128, (128, 64)),
                          discriminator_units=((32,), 16, (16,)),
                          decoder=decoder_adj,  # 解码器
                          discriminator=encoder_rgcn,  # 判别器
                          scope=scope,
                          soft_gumbel_softmax=False,
                          hard_gumbel_softmax=False,
                          batch_discriminator=False)
    return model, data

def transformer_model(scope, hp):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        model = Transformer(hp)
        return model


def gcn_model(scope, placeholders, input_dim, edge_input_dim):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        model = BGCN(placeholders, input_dim=input_dim,
                    edge_input_dim=edge_input_dim, logging=True)
        return model

def gcn_placeholder(features, edge_features):
    placeholders = {
        # 1
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
        # (12,1)
        'pm': tf.sparse_placeholder(tf.float32),
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'line_support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
        # (12,1)
        'edge_features': tf.sparse_placeholder(tf.float32, shape=tf.constant(edge_features[2], dtype=tf.int64)),

        # 'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        # 'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        # helper variable for sparse dropout
        'num_features_nonzero': tf.placeholder(tf.int32),
        'num_edgefeatures_nonzero': tf.placeholder(tf.int32),
    }
    return placeholders


def construct_feed_dict(features, support, placeholders, edge_features, pm, line_support):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['dropout']: 0.1})
    feed_dict.update({placeholders['edge_features']: edge_features})
    feed_dict.update({placeholders['pm']: pm})
    feed_dict.update({placeholders['num_edgefeatures_nonzero']: edge_features[1].shape})
    feed_dict.update({placeholders['line_support'][i]: line_support[i] for i in range(len(line_support))})

    return feed_dict


def make_env(scenario_name, arglist, benchmark=False):
    from multiagentnfv.environment import MultiAgentEnv
    import multiagentnfv.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world("nfvmaddpg/topo/netGraph_abilene.pickle", "nfvmaddpg/request/vnf.vocab", "nfvmaddpg/request/usage.vocab")
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, obs_shape_n, arglist, features, edge_features):
    trainers = []
    model = mlp_model
    
    trainer = MADDPGAgentTrainer
    for i in range(env.n):
        gan, data = gan_model('p_func_%d' % i, arglist, features, edge_features)
        target_gan, target_data = gan_model('target_p_func_%d' % i, arglist, features, edge_features)
        trainers.append(trainer(name="agent_%d" % i, model=model, 
        obs_shape_n=obs_shape_n, act_space_n=env.action_space, agent_index=i, args=arglist, gan=[gan, target_gan], data=[data, target_data], local_q_func=(arglist.policy == 'ddpg')))
    return trainers


def preprocess(obs_n):  # 每个agent获得拓扑的信息 adj, features, line_adj, edge_features, world.topo.pm
    new_obs_n = []
    for obs in obs_n:
        adj, features, line_adj, edge_features, pm = obs
        support = preprocess_adj(adj)
        line_support = preprocess_adj(line_adj)
        features = preprocess_features(features)
        edge_features = preprocess_features(edge_features)
        pm = process_pm(pm)
        new_obs_n.append((support, features, line_support, edge_features, pm))
    return new_obs_n


def process_request(request_n):
    global REQ_NUM
    train_vnf_n = []
    tran_pos_n = []
    for request in request_n:
        if request == 0:
            train_vnf_n.append(0)
            tran_pos_n.append(0)
            continue
        request.add_prefix(REQ_NUM)
        REQ_NUM += 1
        request.forceOrder = dict()
        prsr = Parser(request)
        prsr.preparse()
        parseDict = prsr.parseDict.copy()     
        optords = prsr.optorderdict.copy()
        request.optords = optords
        parallel_num = prsr.parallel_num.copy()
        perms = dict()
        for v in optords.values():
            perms[",".join(v)] = []
            for x in permutations(v):
                perms[",".join(v)].append(x)


        prod = list(product(*perms.values()))[0]
        reqPlacementInputList = []   # 根据所有optorder可能的顺序组合  构造每个确定顺序的排列
        for i in range(len(prod)):
            for k in optords.keys():
                if prod[i] in perms[k]:
                    request.forceOrder[k] = prod[i]
        prsr = Parser(request)
        prsr.parse()
        reqPlacementInput = prsr.create_pairs()
        posencode = posEncode(reqPlacementInput, parseDict, optords)
        transformer_input = {}
        for k, v in posencode.items():
            for m, n in reqPlacementInput['UF'].items():
                if k == m:
                    transformer_input[k] = []
                    transformer_input[k].append(posencode[k])
                    transformer_input[k].append(reqPlacementInput['UF'][k])
                    posencode[k] = transformer_input[k]
                    reqPlacementInput['UF'][k] = transformer_input[k]
                else:
                    transformer_input[k] = posencode[k]
                    transformer_input[m] = reqPlacementInput['UF'][m]
        for k in list(transformer_input.keys()):
            if type(transformer_input[k]) != list:
                del transformer_input[k]
        train_vnf = '' 
        tran_pos = '' 
        for k, v in transformer_input.items():     
            train_vnf += v[1] + ' '     
            tran_pos += str(v[0]) + ' '
        train_vnf_n.append(train_vnf)
        tran_pos_n.append(tran_pos)

    del transformer_input, prsr, posencode, reqPlacementInput, prod, perms, optords, parallel_num, parseDict
    gc.collect()
    return train_vnf_n, tran_pos_n


def create_request(request_n, agents):
    for index, request in enumerate(request_n):
        if request == 0:
            agents[index].request = None
            agents[index].processing = False
        else:
            agents[index].processing = True
            new_request = Request()
            new_request.ID = request.prefix
            new_request.optord_vnfs = request.optords
            for key, opts in new_request.optord_vnfs.items():
                for opt_index in range(len(opts)):
                    opts[opt_index] = ''.join(opts[opt_index].split('_')[1:])
            new_request.request_class = request
            new_request.arrival_time = datetime.now()
            new_request.reliability_requirements = np.random.random() * 0.15 + 0.8
            agents[index].request = new_request


def get_batch(sent1, sent2, vocab_fpath, batch_size, maxlen1, maxlen2, shuffle=False):
    sents1 = sent1.strip()
    sents2 = sent2.strip()
    token2idx, idx2token = load_vocab(vocab_fpath)
    x = encode(sents1, "x", token2idx)
    inp_str = sents2.encode("utf-8").decode("utf-8")
    tokens = inp_str.split()
    y = [int(token) for token in tokens]
    if len(x) < maxlen1:
        x.extend([0] * (maxlen1 - len(x)))
    if len(y) < maxlen2:
        y.extend([0] * (maxlen2 - len(y)))


    # batches = input_fn(sents1, sents2, vocab_fpath,
    #                    batch_size, maxlen1, maxlen2, shuffle=shuffle)
    # num_batches = calc_num_batches(len(sents1), batch_size)
    return np.expand_dims(x, 0), np.expand_dims(y, 0)


def sample_z(): 
    return np.random.normal(0, 1, size=4)

def release_resources(lock, world):
    while True:
        lock.acquire()
        world.time = datetime.now()
        request_run = world.requests_running[:]
        for request in request_run:
            if (world.time - request.schedule_time).seconds >= request.time_to_finish:
                logging.info("# Releasing Resource!")
                world.requests_running.remove(request)
                world.requests_finished += 1
                for vl in request.virtual_links:
                    for link in vl.links:
                        link.bandwidth_occupied -= vl.datarate_required
                        link.bandwidth_available = link.bandwidth_total - link.bandwidth_occupied
                for vnf in request.vnfs:
                    for instance in vnf.instances_belong:
                        instance.traffic_available += vnf.traffic_required
                        ratio = np.random.randint(2)
                        alpha = 0.8
                        origin = instance.traffic_available
                        instance.traffic_available = instance.traffic_available * alpha * ratio + instance.traffic_available * (1 - ratio)
                        instance.traffic_total -= (origin - instance.traffic_available)
                
                del request
        del request_run
        gc.collect()

        time.sleep(0.01)
        lock.release()


def train(arglist):
    with U.multi_threaded_session():  # 只使用一个cpu的会话
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers 
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]  # gcn 4 + selfattention 4 + 随机 4
        obs_n = env.reset()  # 每个agent获得拓扑的信息 adj, features, line_adj, edge_features, world.topo.pm          
        obs_n = preprocess(obs_n)
        features = obs_n[0][1]
        edge_features = obs_n[0][-2]
        trainers = get_trainers(env, obs_shape_n, arglist, features, edge_features)
        print('Using policy {} '.format(arglist.policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()

        request_list = requestlist()
        request_n = []
        for i in range(env.n):
            request_num = np.random.randint(len(request_list) + 1)
            if request_num == len(request_list):
                request_n.append(0)
            else:
                request_n.append(deepcopy(request_list[request_num]))
        

        train_vnf_n, tran_pos_n = process_request(request_n)
        create_request(request_n, env.agents) # 为每个agent创建Request类

        for i in range(env.n):
            if train_vnf_n[i] != 0:
                # train_vnf_n[i].strip()
                # tran_pos_n[i].strip()


                xs, ys = get_batch(train_vnf_n[i], tran_pos_n[i],
                        'nfvmaddpg/model/transformer/src/vnf.vocab', 1, 50, 50)
                # train_batches, num_train_batches, num_train_samples = get_batch(train_vnf_n[i], tran_pos_n[i],
                #         'nfvmaddpg/model/transformer/src/vnf.vocab', 1, 50, 50)

                # iter = train_batches.make_one_shot_iterator()
                # # iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
                # xs, ys = iter.get_next()  
                # train_init_op = iter.make_initializer(train_batches)
                # tf.get_default_session().run(train_init_op)
                # obs_n[i].append((xs[0], ys[0]))
                # xs, ys = tf.get_default_session().run([xs,ys])
                obs_n[i] = obs_n[i] + (deepcopy(xs), deepcopy(ys))
            else:
                obs_n[i] = obs_n[i] + (np.zeros((1, 50)), np.zeros((1, 50)))
        del train_vnf_n, tran_pos_n, xs, ys
        gc.collect()

        episode_step = 0
        train_step = 0
        t_start = time.time()

        lock = threading.RLock()
        t = threading.Thread(target=release_resources, args=(lock, env.world))
        t.setDaemon(True)
        t.start()

        logging.info('# Starting iterations...')
        # print('Starting iterations...')

        while True:
            # info = psutil.virtual_memory()
            # logging.info('内存使用：%s' % str(psutil.Process(os.getpid()).memory_info().rss))
            logging.info("# Request Num %s" % str(REQ_NUM))
            # get action
            # prev = psutil.Process(os.getpid()).memory_info().rss
            action_n = [agent.action(obs) if env_agent.processing==True else [np.zeros((1,12,12)), np.zeros((1,12,12))] for agent, obs, env_agent in zip(trainers, obs_n, env.agents)]
            # after = psutil.Process(os.getpid()).memory_info().rss
            # logging.info('action后内存变化：%s' % str(after - prev))
            # action_tmp = []
            # for action in action_n:
            #     action_tmp.append(np.concatenate((action[0], action[1]), -1))
            # 要更新gcn和transformer 需要在replay buffer中放入最初的adj等
            # environment step
            # prev = psutil.Process(os.getpid()).memory_info().rss
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)  # new_obs_n改
            # after = psutil.Process(os.getpid()).memory_info().rss
            # logging.info('step后内存变化：%s' % str(after - prev))
            if np.any(rew_n):
                logging.info("Reward is {}".format(rew_n))
            new_obs_n = preprocess(new_obs_n)         

            # prev = psutil.Process(os.getpid()).memory_info().rss
            request_n = [] 
            for i in range(env.n):             
                request_num = np.random.randint(len(request_list) + 1) 
                if request_num == len(request_list):                 
                    request_n.append(0) 
                else:                 
                    request_n.append(deepcopy(request_list[request_num]))
            # 为每个agent创建Request类
            train_vnf_n, tran_pos_n = process_request(request_n)         
            create_request(request_n, env.agents)
            # after = psutil.Process(os.getpid()).memory_info().rss
            # logging.info('处理1后内存变化：%s' % str(after - prev))

            # prev = psutil.Process(os.getpid()).memory_info().rss
            # trans_out = []
            for i in range(env.n):
                if train_vnf_n[i] != 0:
                    # train_batches, num_train_batches, num_train_samples = get_batch(train_vnf_n[i], tran_pos_n[i],
                    #                                                                 'nfvmaddpg/model/transformer/src/vnf.vocab', 1, 50, 50)

                    # iter = train_batches.make_one_shot_iterator()
                    # # iter = tf.data.Iterator.from_structure(
                    # #     train_batches.output_types, train_batches.output_shapes)
                    # xs, ys = iter.get_next()
                    # # train_init_op = iter.make_initializer(train_batches)
                    # # tf.get_default_session().run(train_init_op)
                    # # new_obs_n[i].append((xs[0], ys[0]))
                    # xs, ys = tf.get_default_session().run([xs,ys])
                    xs, ys = get_batch(train_vnf_n[i], tran_pos_n[i],
                                       'nfvmaddpg/model/transformer/src/vnf.vocab', 1, 50, 50)
                    new_obs_n[i] = new_obs_n[i] + (deepcopy(xs), deepcopy(ys))
                    # new_obs_n[i] = new_obs_n[i] + (tf.get_default_session().run(xs[0]), tf.get_default_session().run(ys[0]))
                    # memory, y = tran_model.train(xs, ys)
                    # # U.initialize()
                    # tf.get_default_session().run(train_init_op)
                    # trans_out.append(tf.get_default_session().run(memory))
                    # # trans_out.append(memory)
                    del xs, ys
                    gc.collect()
                else:
                    new_obs_n[i] = new_obs_n[i] + (np.zeros((1, 50)), np.zeros((1, 50)))
            # after = psutil.Process(os.getpid()).memory_info().rss
            # logging.info('处理2后内存变化：%s' % str(after - prev))

            del train_vnf_n, tran_pos_n
            gc.collect()
                    # new_obs_n[i].append((np.zeros((50,)), np.zeros((50,))))
            # for i in range(env.n):
            #     noise = sample_z() 
            #     if np.all(trans_out[i] == 0):                 
            #         new_obs_n[i] = np.zeros((12,))
            #         continue 
            #     new_obs_n[i] = np.hstack((noise, trans_out[i], gcn_out_n[i]))
            

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            # prev = psutil.Process(os.getpid()).memory_info().rss
            for i, agent in enumerate(trainers):
                action_n[i][0] = action_n[i][0].reshape(action_n[i][0].shape[1], action_n[i][0].shape[2])
                action_n[i][1] = action_n[i][1].reshape(action_n[i][1].shape[1], action_n[i][1].shape[2])
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                # logging.info("Agent_%d's Replay buffer size: %s" % (i,str(agent.replay_buffer.sizeofstorage())))
            # after = psutil.Process(os.getpid()).memory_info().rss
            # logging.info('存入经验后内存变化：%s' % str(after - prev))
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                obs_n = preprocess(obs_n)         
                
                request_n = []
                for i in range(env.n):
                    request_num = np.random.randint(len(request_list) + 1)
                    if request_num == len(request_list):
                        request_n.append(0)
                    else:
                        request_n.append(deepcopy(request_list[request_num]))

                train_vnf_n, tran_pos_n = process_request(request_n)
                create_request(request_n, env.agents)  # 为每个agent创建Request类
                # trans_out = []
                for i in range(env.n):
                    if train_vnf_n[i] != 0:
                        # train_batches, num_train_batches, num_train_samples = get_batch(train_vnf_n[i], tran_pos_n[i],
                        #                                                                 'nfvmaddpg/model/transformer/src/vnf.vocab', 1, 50, 50)

                        # iter = train_batches.make_one_shot_iterator()
                        # # iter = tf.data.Iterator.from_structure(
                        # #     train_batches.output_types, train_batches.output_shapes)
                        # xs, ys = iter.get_next()
                        # # train_init_op = iter.make_initializer(train_batches)
                        # # tf.get_default_session().run(train_init_op)
                        # # obs_n[i].append((xs[0], ys[0]))
                        # xs, ys = tf.get_default_session().run([xs,ys])
                        xs, ys = get_batch(train_vnf_n[i], tran_pos_n[i],
                                           'nfvmaddpg/model/transformer/src/vnf.vocab', 1, 50, 50)
                        obs_n[i] = obs_n[i] + (deepcopy(xs), deepcopy(ys))
                        # obs_n[i] = obs_n[i] + (tf.get_default_session().run(xs[0]), tf.get_default_session().run(ys[0]))
                        # memory, y = tran_model.train(xs, ys)
                        # # U.initialize()
                        # tf.get_default_session().run(train_init_op)
                        # trans_out.append(tf.get_default_session().run(memory))
                        # trans_out.append(memory)
                    else:
                        obs_n[i] = obs_n[i] + (np.zeros((1, 50)), np.zeros((1, 50)))
                del train_vnf_n, tran_pos_n, xs, ys
                gc.collect()
                        # obs_n[i].append((np.zeros((50,)), np.zeros((50,))))

                # for i in range(env.n):
                #     noise = sample_z()
                #     if np.all(trans_out[i] == 0):
                #         obs_n[i] = np.zeros((12,))
                #         continue
                #     obs_n[i] = np.hstack((noise, trans_out[i], gcn_out_n[i]))

                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            # prev = psutil.Process(os.getpid()).memory_info().rss
            for index, agent in enumerate(trainers):
                loss = agent.update(trainers, train_step, index)
            # after = psutil.Process(os.getpid()).memory_info().rss
            # logging.info('更新后内存变化：%s' % str(after - prev))

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                # if num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                # else:
                #     print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                #         train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                #         [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                    agrew_file_name= str(arglist.plots_dir) + str(arglist.exp_name) + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
