import numpy as np
import random
import tensorflow as tf
import gc
# import common.tf_util as U
# import maddpg.common.tf_util as U
from nfvmaddpg.common import tf_util as U
from nfvmaddpg.common.distributions import make_pdtype
from nfvmaddpg import AgentTrainer
from nfvmaddpg.trainer.replay_buffer import ReplayBuffer
import logging


def construct_feed_dict(features, support, placeholders, edge_features, pm, line_support):
    feed_dict = dict()
    # feed_dict.update({placeholders['labels']: labels})
    # feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    # feed_dict.update({placeholders['support'][i]: support[i]
    #                   for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[0].shape[1]})
    feed_dict.update({placeholders['dropout']: 0.1})
    feed_dict.update({placeholders['edge_features']: edge_features})
    feed_dict.update({placeholders['pm']: pm})
    feed_dict.update({placeholders['num_edgefeatures_nonzero']: edge_features[0].shape[1]})
    feed_dict.update({placeholders['line_support']: line_support})
    # feed_dict.update({placeholders['line_support'][i]: line_support[i] for i in range(len(line_support))})

    return feed_dict

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(agents_num, edges_labels, nodes_labels, training, dropout_rate, make_obs_ph_n, act_space_n, p_index, p_func, data, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        # act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        la = 0.2
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        # act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        act_ph_n = [tf.placeholder(dtype=tf.float32, shape=[None]+[act_space_n[i][0].shape[0], act_space_n[i][0].shape[1] +
                                                                   act_space_n[i][1].shape[1]], name="agent_%d_action" % i) for i in range(len(act_space_n))]  # act的placeholder
        # act_ph_n = [[tf.placeholder(dtype=tf.float32, shape=[None]+list(act_space_n[i][0].shape), name="agent_%d_node" % i),
        #              tf.placeholder(dtype=tf.float32, shape=[None]+list(act_space_n[i][1].shape), name="agent_%d_edge" % i)] for i in range(len(act_space_n))]  # act的placeholder

        p_input = obs_ph_n[p_index]   # 状态
        
        # topos, a, x, _, _, _, _ = data.next_train_batch(1)
        # embeddings = p_func.sample_z(1)
        # Actor当前网络基于状态S得到动作A
        p = p_func[0] # TODO
        # p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units) 
        generator_vars = U.scope_vars("p_func_%d/generator" % p_index)
        discriminator_vars = U.scope_vars("p_func_%d/discriminator" % p_index)
        gcn_vars = U.scope_vars("p_func_%d/gcn" % p_index)
        transformer_vars = U.scope_vars("p_func_%d/transformer" % p_index)
        transformer_vars.extend(U.scope_vars("p_func_%d/encoder" % p_index))
        # generator_vars = U.scope_vars(U.absolute_scope_name("p_func/generator"))
        # discriminator_vars = U.scope_vars(U.absolute_scope_name("p_func/discriminator"))
        # p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # generator_vars, discriminator_vars, gcn_vars, transformer_vars
        # input_generator_vars_params, input_discriminator_vars_params, input_gcn_vars_params, input_transformer_vars_params
        # set_generator_vars_params_op, set_discriminator_vars_params_op, set_gcn_vars_params_op, set_transformer_vars_params_op

        input_generator_vars_params = []
        for param in generator_vars:
            input_generator_vars_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_generator_vars_params_op = []
        for idx, param in enumerate(input_generator_vars_params):
            set_generator_vars_params_op.append(
                generator_vars[idx].assign(param))

        input_discriminator_vars_params = []
        for param in discriminator_vars:
            input_discriminator_vars_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_discriminator_vars_params_op = []
        for idx, param in enumerate(input_discriminator_vars_params):
            set_discriminator_vars_params_op.append(
                discriminator_vars[idx].assign(param))

        input_gcn_vars_params = []
        for param in gcn_vars:
            input_gcn_vars_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_gcn_vars_params_op = []
        for idx, param in enumerate(input_gcn_vars_params):
            set_gcn_vars_params_op.append(
                gcn_vars[idx].assign(param))

        input_transformer_vars_params = []
        for param in transformer_vars:
            input_transformer_vars_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_transformer_vars_params_op = []
        for idx, param in enumerate(input_transformer_vars_params):
            set_transformer_vars_params_op.append(
                transformer_vars[idx].assign(param))
        # wrap parameters in distribution
        # act_pd = act_pdtype_n[p_index].pdfromflat(p)
        act_sample = [p.nodes_hat + np.random.randn(
            act_space_n[p_index][0].shape[0], act_space_n[p_index][0].shape[1]), p.edges_hat + np.random.randn(act_space_n[p_index][1].shape[0], act_space_n[p_index][1].shape[1])]
        
        # act_sample = act_pd.sample()  # Actor当前网络基于状态S得到动作A后加入了噪声
        p_reg = tf.reduce_mean(tf.square(p.nodes_hat)) + \
            tf.reduce_mean(tf.square(p.edges_hat))  # L2正则

        act_input_n = act_ph_n
        # act_input_n = act_ph_n + []
        # act_input_n[p_index] = act_pd.sample()
        obs_input_n = [] 
        for obs in obs_ph_n:             
            obs_input_n.append(tf.expand_dims(obs, -1))
        
        embedding_gen = p_func[0].embeddings
        target_embedding_gen = p_func[1].embeddings
        obs_input_n = []
        for num in range(agents_num):
            obs_input_n.append(tf.expand_dims(embedding_gen, -1))
        q_input = tf.concat(obs_input_n + act_input_n, -1)

        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)  # 更新Actor当前网络的所有参数

        eps = tf.random_uniform(tf.shape(p.logits_real)[
                                :1], dtype=p.logits_real.dtype)  # 表示从真实数据和生成数据中采样的随机值

        x_int0 = p.adjacency_tensor * tf.expand_dims(tf.expand_dims(
            eps, -1), -1) + p.edges_softmax * (1 - tf.expand_dims(tf.expand_dims(eps, -1), -1))
        x_int1 = p.node_tensor * tf.expand_dims(tf.expand_dims(eps, -1), -1) + p.nodes_softmax * (
            1 - tf.expand_dims(tf.expand_dims(eps, -1), -1))

        grad0, grad1 = tf.gradients(p.D_x(
            (x_int0, None, x_int1), p.discriminator_units), (x_int0, x_int1))  # 995   95

        grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), -1) + tf.reduce_mean(
            ((1 - tf.norm(grad1, axis=-1)) ** 2), -1, keepdims=True)  # shape 批次样本数,单个值

        rl_loss = pg_loss + p_reg * 1e-3
        loss_G = tf.reduce_mean(- p.logits_fake)
        loss_D = tf.reduce_mean(- p.logits_real + p.logits_fake)
        grad_penalty = tf.reduce_mean(grad_penalty)

        optimize_generator = U.minimize_and_clip(optimizer, la * loss_G + (1 - la) * rl_loss, generator_vars, grad_norm_clipping)
        optimize_discriminator = U.minimize_and_clip(optimizer, loss_D + 10 * grad_penalty, discriminator_vars, grad_norm_clipping)

        # embeddings = obs_input_n[p_index]
        # Create callable functions
        train_G = [la * loss_G + (1 - la) * rl_loss, optimize_generator]
        # train_G = U.function(inputs=obs_ph_n + act_ph_n + [embeddings[p_index]] + [edges_labels[p_index]] + [nodes_labels[p_index]] + [
        #                      training[p_index]] + [dropout_rate[p_index]], outputs=la * loss_G + (1 - la) * rl_loss, updates=[optimize_generator])
        train_D = [loss_D + 10 * grad_penalty, optimize_discriminator]
        # train_D = U.function(inputs=obs_ph_n + act_ph_n + [embeddings[p_index]] + [edges_labels[p_index]] + [nodes_labels[p_index]] + [
        #                      training[p_index]] + [dropout_rate[p_index]], outputs=loss_D + 10 * grad_penalty, updates=[optimize_discriminator])

        
        act = act_sample
        # act = U.function(inputs=[embeddings[p_index]] + [edges_labels[p_index]] + [nodes_labels[p_index]] + [training[p_index]] + [dropout_rate[p_index]], outputs=act_sample)
        # p_values = U.function([obs_ph_n[p_index]], p)
        p_values = [p.nodes_hat, p.edges_hat]
        # p_values = U.function(inputs=[embeddings[p_index]] + [edges_labels[p_index]] + [nodes_labels[p_index]] + [
        #     training[p_index]] + [dropout_rate[p_index]], outputs=p)

        # target network
        target_p = p_func[1]
        # target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_generator_vars = U.scope_vars("target_p_func_%d/generator" % p_index)
        target_discriminator_vars = U.scope_vars("target_p_func_%d/discriminator" % p_index)
        target_gcn_vars = U.scope_vars("target_p_func_%d/gcn" % p_index)
        target_transformer_vars = U.scope_vars("target_p_func_%d/transformer" % p_index)

        update_target_generator = make_update_exp(generator_vars, target_generator_vars)
        update_target_discriminator = make_update_exp(discriminator_vars, target_discriminator_vars)
        update_target_gcn = make_update_exp(gcn_vars, target_gcn_vars)
        update_target_transformer = make_update_exp(transformer_vars, target_transformer_vars)

        # target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act_sample = [target_p.nodes_hat + np.random.randn(act_space_n[p_index][0].shape[0], act_space_n[p_index][0].shape[1]),
                             target_p.edges_hat + np.random.randn(act_space_n[p_index][1].shape[0], act_space_n[p_index][1].shape[1])]

        target_act = target_act_sample
        # target_act = U.function(inputs=[embeddings[p_index]] + [edges_labels[p_index]] + [nodes_labels[p_index]] + [
        #                         training[p_index]] + [dropout_rate[p_index]], outputs=target_act_sample)
        generator_vars, discriminator_vars, gcn_vars, transformer_vars         
        input_generator_vars_params, input_discriminator_vars_params, input_gcn_vars_params, input_transformer_vars_params         
        set_generator_vars_params_op, set_discriminator_vars_params_op, set_gcn_vars_params_op, set_transformer_vars_params_op
        return generator_vars, discriminator_vars, gcn_vars, transformer_vars, input_generator_vars_params, input_discriminator_vars_params, input_gcn_vars_params, input_transformer_vars_params, set_generator_vars_params_op, set_discriminator_vars_params_op, set_gcn_vars_params_op, set_transformer_vars_params_op, obs_ph_n, act_ph_n, p, target_p, act, train_D, train_G, update_target_discriminator, update_target_generator, update_target_gcn, update_target_transformer, {'p_values': p_values, 'target_act': target_act}

def q_train(agents_num, p_func, make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):  # Critic当前网络
        # create distribtuions
        # act_space [node_action_space, edge_action_space]
        # act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        
        embedding_gen = p_func[0].embeddings
        target_embedding_gen = p_func[1].embeddings


        act_ph_n = [tf.placeholder(dtype=tf.float32, shape=[None]+[act_space_n[i][0].shape[0], act_space_n[i][0].shape[1] + act_space_n[i][1].shape[1]], name="agent_%d_action" % i) for i in range(len(act_space_n))]  # act的placeholder
        # act_ph_n = [[tf.placeholder(dtype=tf.float32, shape=[None]+list(act_space_n[i][0].shape), name="agent_%d_node" % i), 
        #                 tf.placeholder(dtype=tf.float32, shape=[None]+list(act_space_n[i][1].shape), name="agent_%d_edge" % i)] for i in range(len(act_space_n))]  # act的placeholder

        target_ph = tf.placeholder(tf.float32, [None], name="target") # target的placeholder

        # act_input_n = []
        # for act in act_ph_n:
        #     act_input_n.append(tf.concat([act[0], act[1]], -1))
        obs_input_n = []
        for num in range(agents_num):
            obs_input_n.append(tf.expand_dims(embedding_gen, -1))
        # 4*3=12 + 12,12 + 12,13
        q_input = tf.concat(obs_input_n + act_ph_n, -1)
        # q_input = tf.concat(obs_input_n + act_ph_n, -1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        # train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        train = [loss, optimize_expr]
        # train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = q
        # q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        obs_input_n = []
        for num in range(agents_num):
            obs_input_n.append(tf.expand_dims(target_embedding_gen, -1))
        # 4*3=12 + 12,12 + 12,13
        q_input = tf.concat(obs_input_n + act_ph_n, -1)
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = target_q
        # target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return target_ph, act_ph_n, train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, gan, data, local_q_func=False):
                        #[(24,), (24,), (24,)]  [node_action_space, edge_action_space]
        self.gan = gan
        self.data = data
        self.name = name
        self.n = len(obs_shape_n) # agent个数
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []   # 为给定形状和数据类型的一批张量创建占位符
        edges_labels_n = []   # 为给定形状和数据类型的一批张量创建占位符
        nodes_labels_n = []   # 为给定形状和数据类型的一批张量创建占位符
        training_n = []   # 为给定形状和数据类型的一批张量创建占位符
        dropout_rate_n = []   # 为给定形状和数据类型的一批张量创建占位符
        # embeddings_n = []
        for i in range(self.n):
            # tf.placeholder(dtype, [None] + list(shape), name=name)
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
            edges_labels_n.append(U.BatchInput([12,12], name="edge_labels"+str(i)).get())
            nodes_labels_n.append(U.BatchInput([12,12], name="nodes_labels"+str(i)).get())
            # training_n.append(tf.placeholder_with_default(False, shape=(), name="training"+str(i)))
            training_n.append(tf.placeholder(tf.bool, (), name="training"+str(i)))
            dropout_rate_n.append(tf.placeholder(tf.float32, shape=(), name="dropout_rate"+str(i)))
            # dropout_rate_n.append(tf.placeholder_with_default(0, shape=(), name="dropout_rate"+str(i)))
            # embeddings_n.append(U.BatchInput(obs_shape_n[i], name="embeddings_n"+str(i)).get())

        # Create all the functions necessary to train the model 
        self.target_ph, self.target_act_placeholder, self.q_train, self.q_update, self.q_debug = q_train(  # 更新critic当前， 更新critic目标  {当前q值；目标q值}
            agents_num=self.n,
            p_func=self.gan,
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,  # [node_action_space, edge_action_space]
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.generator_vars, self.discriminator_vars, self.gcn_vars, self.transformer_vars, self.input_generator_vars_params, self.input_discriminator_vars_params, self.input_gcn_vars_params, self.input_transformer_vars_params, self.set_generator_vars_params_op, self.set_discriminator_vars_params_op, self.set_gcn_vars_params_op, self.set_transformer_vars_params_op, self.obs_placeholder, self.act_placeholder, self.model, self.target_model, self.act, self.p_train_D, self.p_train_G, self.p_update_D, self.p_update_G, self.p_update_gcn, self.p_update_trans, self.p_debug = p_train(  # 建立actor 加入噪声的动作， 更新actor， 更新actor目标， {未加噪声的总做；加入噪声的下一动作}
            agents_num=self.n,
            edges_labels=edges_labels_n,
            nodes_labels=nodes_labels_n,
            training=training_n,
            dropout_rate=dropout_rate_n,
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=self.gan,
            data=self.data,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(6e3)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        
        # gcn_opt = self.model.gcn_model.optimizer.minimize(self.q_train[0], var_list=self.model.gcn_vars)
        # trans_opt = self.model.trans_model.optimizer.minimize(self.q_train[0], var_list=self.model.trans_vars)
        # gcn_vars = U.scope_vars("p_func/gcn")
        self.gcn_opt = self.model.gcn_model.optimizer.minimize(self.q_train[0], var_list=U.scope_vars("p_func_%d/gcn" % self.agent_index))
        self.trans_opt = self.model.trans_model.optimizer.minimize(self.q_train[0], var_list=U.scope_vars("p_func_%d/transformer" % self.agent_index))
        self.target_gcn_opt = self.target_model.gcn_model.optimizer.minimize(self.q_train[0], var_list=U.scope_vars("p_func_%d/gcn" % self.agent_index))
        self.target_trans_opt = self.target_model.trans_model.optimizer.minimize(self.q_train[0], var_list=U.scope_vars("p_func_%d/transformer" % self.agent_index))
        # U.initialize()

    def action(self, obs):
        # act = U.function(inputs=[embeddings] + edges_labels[p_index] + nodes_labels[p_index] + [
        #                  training[p_index]] + [dropout_rate[p_index]], outputs=act_sample)
        noise = np.random.normal(0, 1, size=4)
        topos, a, x, _, _, _, _ = self.data[0].next_train_batch(1)
        training = False
        dropout_rate = 0.1
        # embeddingstf.expand_dims(obs, -1)
        # embeddings = obs[None]
        support, features, line_support, edge_features, pm, trans_input_x, trans_input_y = obs
        gcn_feed_dict = construct_feed_dict(np.expand_dims(features.todense(), 0), np.expand_dims(support.todense(), 0),
                                            self.model.gcn_placeholders, np.expand_dims(edge_features.todense(), 0), np.expand_dims(pm, 0), np.expand_dims(line_support.todense(), 0))
        feed_dict = {self.model.edges_labels: a,
                         self.model.nodes_labels: x,
                         self.model.noise: noise[None],
                         self.model.training: training,
                         self.model.dropout_rate: dropout_rate,
                         self.model.trans_model.x: trans_input_x,
                         self.model.trans_model.y: trans_input_y
                         }
        feed_dict.update(gcn_feed_dict)
        agentaction = tf.get_default_session().run(self.act, feed_dict=feed_dict)

        del noise, gcn_feed_dict, feed_dict
        gc.collect()

        return agentaction
        # gcn.optimizer.minimize(tf.convert_to_tensor(np.mean(action_tmp)), var_list=U.scope_vars("gcn"))
        # tran_model.optimizer.minimize(tf.convert_to_tensor(np.mean(action_tmp)), var_list=U.scope_vars("transformer"))
        # logging.info('# Updating GCN and Self-Attention Networks...')
        # return self.act(*([embeddings] + [a] + [x] + [training] + [dropout_rate]))[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t, agent_index):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        a = []
        x = []
        # support, features, line_support, edge_features, pm, trans_input
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            # act = np.hstack((act[0], act[1]))
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        # target_act = U.function(inputs=[embeddings] + edges_labels[p_index] + nodes_labels[p_index] + [
        #                         training[p_index]] + [dropout_rate[p_index]], outputs=target_act_sample)
        for i in range(num_sample):
            
            for n in range(self.n):
                topos, adj, node_x, _, _, _, _ = self.data[1].next_train_batch(self.args.batch_size)
                a.append(adj)
                x.append(node_x)
            training = False
            dropout_rate = 0.1
            # embeddings = obs_next_n
            

            target_act_next_n = []
            for i in range(self.n):
                target_act_next_n_i = []
                for batch in range(self.args.batch_size):
                    noise = np.random.normal(0, 1, size=4)
                    support, features, line_support, edge_features, pm, trans_input_x, trans_input_y = obs_next_n[i][batch]
                    gcn_feed_dict = construct_feed_dict(np.expand_dims(features.todense(), 0), np.expand_dims(support.todense(), 0), 
                                                        agents[i].target_model.gcn_placeholders, np.expand_dims(edge_features.todense(), 0), np.expand_dims(pm, 0), np.expand_dims(line_support.todense(), 0))
                    feed_dict = {agents[i].target_model.edges_labels: a[i],
                                    agents[i].target_model.nodes_labels: x[i],
                                    agents[i].target_model.noise: noise[None],
                                    agents[i].target_model.training: training,
                                    agents[i].target_model.dropout_rate: dropout_rate,
                                    agents[i].target_model.trans_model.x: trans_input_x,
                                    agents[i].target_model.trans_model.y: trans_input_y
                                    }
                    feed_dict.update(gcn_feed_dict)
                    target_act_next_n_i.append(tf.get_default_session().run(agents[i].p_debug['target_act'], feed_dict=feed_dict))
                target_act_next_n.append(target_act_next_n_i)
            # feed_dict = {self.model.edges_labels: a,
            #              self.model.nodes_labels: x,
            #              self.model.embeddings: embeddings,
            #              self.model.training: training,
            #              self.model.dropout_rate: dropout_rate,
            #              }
            # target_act_next_n = [tf.get_default_session().run(agents[i].p_debug['target_act'], feed_dict={self.target_model.edges_labels: a[i],
            #              self.target_model.nodes_labels: x[i],
            #              self.target_model.embeddings: embeddings[i],
            #              self.target_model.training: training,
            #              self.target_model.dropout_rate: dropout_rate,
            #              }) for i in range(self.n)]
            # target_act_next_n = [agents[i].p_debug['target_act'](
            #     *(embeddings[i] + a[i] + x[i] + training + dropout_rate)) for i in range(self.n)]  # p_debug Actor目标网络 根据经验回放池中采样的下一状态S′选择最优下一动作A′
            act_next_n = []
            for target_act_i in target_act_next_n:
                act_next_n_i = []
                for target_act in target_act_i:
                    act_next_n_i.append(np.concatenate((target_act[0], target_act[1]), -1))
                act_next_n.append(act_next_n_i)
            target_act_next_n = act_next_n

            feed_dict = {}
            # noise = np.random.normal(0, 1, size=4)
            for i in range(self.n):
                noise = np.random.normal(0, 1, size=(self.args.batch_size, 4))
                support_batch = [] 
                features_batch = [] 
                line_support_batch = [] 
                edge_features_batch = [] 
                pm_batch = [] 
                trans_input_x_batch = [] 
                trans_input_y_batch = []
                for batch in range(self.args.batch_size):
                    
                    support, features, line_support, edge_features, pm, trans_input_x, trans_input_y = obs_next_n[i][batch]
                    support_batch.append(np.expand_dims(support.todense(), 0))
                    features_batch.append(np.expand_dims(features.todense(), 0))
                    line_support_batch.append(np.expand_dims(line_support.todense(), 0))
                    edge_features_batch.append(np.expand_dims(edge_features.todense(), 0))
                    pm_batch.append(np.expand_dims(pm, 0))
                    trans_input_x_batch.append(trans_input_x)
                    trans_input_y_batch.append(trans_input_y)
                gcn_feed_dict = construct_feed_dict(
                    np.row_stack(features_batch), np.row_stack(support_batch), agents[i].target_model.gcn_placeholders, 
                    np.row_stack(edge_features_batch), np.row_stack(pm_batch), np.row_stack(line_support_batch))
                feed_dict.update({
                    agents[i].target_model.noise: noise,
                    agents[i].target_model.trans_model.x: np.row_stack(trans_input_x_batch),
                    agents[i].target_model.trans_model.y: np.row_stack(trans_input_y_batch)
                })
                feed_dict.update(gcn_feed_dict)
            target_act_placeholder_tmp = {}
            for act_index, place in enumerate(self.target_act_placeholder):
                target_act_placeholder_tmp[place] = np.row_stack(target_act_next_n[act_index])
            feed_dict.update(target_act_placeholder_tmp)
            target_q_next = tf.get_default_session().run(self.q_debug['target_q_values'], feed_dict=feed_dict)
            # target_q_next = self.q_debug['target_q_values'](
            #     *(embeddings + target_act_next_n))  # q_debug Critic目标网络 计算目标Q值中的Q′(S′,A′,w′)
            # target_q_next = self.q_debug['target_q_values'](
            #     *(embeddings + target_act_next_n))  # q_debug Critic目标网络 计算目标Q值中的Q′(S′,A′,w′)
            attenuation = self.args.gamma * (1.0 - np.expand_dims(done, -1)) * target_q_next
            reward = np.expand_dims(rew, -1)
            target_q += reward + attenuation  # 当前目标Q值yj
        target_q /= num_sample
        target_q = target_q.reshape(self.args.batch_size)

        # noise = np.random.normal(0, 1, size=4)
        q_loss_feed_dict = {self.target_ph: target_q}
        for i in range(self.n):
            noise = np.random.normal(0, 1, size=(self.args.batch_size, 4))                 
            support_batch = []                  
            features_batch = []                  
            line_support_batch = []                  
            edge_features_batch = []                  
            pm_batch = []                  
            trans_input_x_batch = []                  
            trans_input_y_batch = []
            for batch in range(self.args.batch_size):
                # noise = np.random.normal(0, 1, size=4)
                support, features, line_support, edge_features, pm, trans_input_x, trans_input_y = obs_n[i][batch]
                support_batch.append(np.expand_dims(support.todense(), 0))
                features_batch.append(np.expand_dims(features.todense(), 0))
                line_support_batch.append(np.expand_dims(line_support.todense(), 0))
                edge_features_batch.append(np.expand_dims(edge_features.todense(), 0))
                pm_batch.append(np.expand_dims(pm, 0))
                trans_input_x_batch.append(trans_input_x)
                trans_input_y_batch.append(trans_input_y)
            gcn_feed_dict = construct_feed_dict(
                np.row_stack(features_batch), np.row_stack(
                    support_batch), agents[i].model.gcn_placeholders,
                np.row_stack(edge_features_batch), np.row_stack(pm_batch), np.row_stack(line_support_batch))
            q_loss_feed_dict.update({
                agents[i].model.noise: noise,
                agents[i].model.trans_model.x: np.row_stack(trans_input_x_batch),
                agents[i].model.trans_model.y: np.row_stack(trans_input_y_batch)
            })
                # gcn_feed_dict = construct_feed_dict(features, support, agents[i].model.gcn_placeholders, edge_features, pm, line_support)
                # feed_dict_tmp = {
                #     agents[i].model.noise: noise[None],
                #     agents[i].model.trans_model.x: trans_input_x,
                #     agents[i].model.trans_model.y: trans_input_y
                # }
            q_loss_feed_dict.update(gcn_feed_dict)
                # q_loss_feed_dict.update(feed_dict_tmp)
        target_act_placeholder_tmp = {}
        for act_index, place in enumerate(self.target_act_placeholder):
            target_act_placeholder_tmp[place] = np.row_stack(target_act_next_n[act_index])
        q_loss_feed_dict.update(target_act_placeholder_tmp)
        q_loss, _ = tf.get_default_session().run(self.q_train, feed_dict=q_loss_feed_dict)   # 更新Critic当前网络
        # q_loss = self.q_train(*(obs_n + act_n + [target_q]))   # 更新Critic当前网络

        # train p network
        # train_G = U.function(inputs=obs_ph_n + act_ph_n + [embeddings] + edges_labels[p_index] + nodes_labels[p_index] + [training[p_index]] + [dropout_rate[p_index]], outputs=la * loss_G + (1 - la) * rl_loss, updates=[optimize_generator])         
        # train_D = U.function(inputs=obs_ph_n + act_ph_n + [embeddings] + edges_labels[p_index] + nodes_labels[p_index] + [training[p_index]] + [dropout_rate[p_index]], outputs=loss_D + 10 * grad_penalty, updates=[optimize_discriminator])
        # topos, a, x, _, _, _, _ = self.data.next_train_batch(self.n)             training = False             dropout_rate = 0.1
        # p_loss = self.p_train(*(obs_n + act_n))  # 更新Actor当前网络
        feed_dict = {}
        if t % 5 == 0:
            act_placeholder_tmp = {}
            for act_index, place in enumerate(self.act_placeholder):
                act_placeholder_tmp[place] = act_n[act_index]
            feed_dict.update(act_placeholder_tmp)
            # obs_placeholder_tmp = {}
            for i in range(self.n):
                noise = np.random.normal(0, 1, size=(self.args.batch_size, 4))                              
                support_batch = []                               
                features_batch = []                               
                line_support_batch = []                               
                edge_features_batch = []                               
                pm_batch = []                               
                trans_input_x_batch = []                               
                trans_input_y_batch = []
                for batch in range(self.args.batch_size):
                    support, features, line_support, edge_features, pm, trans_input_x, trans_input_y = obs_n[i][batch]
                    support_batch.append(np.expand_dims(support.todense(), 0))                 
                    features_batch.append(np.expand_dims(features.todense(), 0))                 
                    line_support_batch.append(np.expand_dims(line_support.todense(), 0))                 
                    edge_features_batch.append(np.expand_dims(edge_features.todense(), 0))                 
                    pm_batch.append(np.expand_dims(pm, 0))                 
                    trans_input_x_batch.append(trans_input_x)                 
                    trans_input_y_batch.append(trans_input_y)
                    # noise = np.random.normal(0, 1, size=4)
                # gcn_feed_dict = construct_feed_dict(features, support, agents[i].model.gcn_placeholders, edge_features, pm, line_support)
                gcn_feed_dict = construct_feed_dict(np.row_stack(features_batch), np.row_stack(support_batch), agents[i].model.gcn_placeholders, np.row_stack(
                    edge_features_batch), np.row_stack(pm_batch), np.row_stack(line_support_batch))
                feed_dict.update({
                    agents[i].model.noise: noise,
                    agents[i].model.trans_model.x: np.row_stack(trans_input_x_batch),
                    agents[i].model.trans_model.y: np.row_stack(trans_input_y_batch)
                })
                feed_dict.update(gcn_feed_dict)
                    # embeddings = tf.get_default_session().run(self.model.embeddings, feed_dict=feed_dict)
                    # obs_placeholder_tmp[place] = embeddings
            feed_dict.update({self.model.edges_labels: a[agent_index],
                        self.model.nodes_labels: x[agent_index],
                        # self.model.embeddings: embeddings[agent_index],
                        self.model.training: training,
                        self.model.dropout_rate: dropout_rate,
                        #  self.obs_placeholder: obs_n,
                        #  self.act_placeholder: act_n
                        })
                
                # feed_dict.update(obs_placeholder_tmp)
   
            p_loss, _ = tf.get_default_session().run(self.p_train_G, feed_dict=feed_dict)  # 更新Actor当前网络G
            logging.info("# Updating Agent_%s's Actor_G..." % agent_index)
            # p_loss = self.p_train_G(
            #     *(obs_n + act_n + embeddings + a[index] + x[index] + training + dropout_rate))  # 更新Actor当前网络G
        else:
            act_placeholder_tmp = {}
            for act_index, place in enumerate(self.act_placeholder):
                act_placeholder_tmp[place] = act_n[act_index]
            feed_dict.update(act_placeholder_tmp)
            # obs_placeholder_tmp = {}
            for i in range(self.n):
                noise = np.random.normal(0, 1, size=(self.args.batch_size, 4))
                support_batch = []
                features_batch = []
                line_support_batch = []
                edge_features_batch = []
                pm_batch = []
                trans_input_x_batch = []
                trans_input_y_batch = []
                for batch in range(self.args.batch_size):
                    support, features, line_support, edge_features, pm, trans_input_x, trans_input_y = obs_n[
                        i][batch]
                    support_batch.append(np.expand_dims(support.todense(), 0))
                    features_batch.append(
                        np.expand_dims(features.todense(), 0))
                    line_support_batch.append(
                        np.expand_dims(line_support.todense(), 0))
                    edge_features_batch.append(
                        np.expand_dims(edge_features.todense(), 0))
                    pm_batch.append(np.expand_dims(pm, 0))
                    trans_input_x_batch.append(trans_input_x)
                    trans_input_y_batch.append(trans_input_y)
                    # noise = np.random.normal(0, 1, size=4)
                # gcn_feed_dict = construct_feed_dict(features, support, agents[i].model.gcn_placeholders, edge_features, pm, line_support)
                gcn_feed_dict = construct_feed_dict(np.row_stack(features_batch), np.row_stack(support_batch), agents[i].model.gcn_placeholders, np.row_stack(
                    edge_features_batch), np.row_stack(pm_batch), np.row_stack(line_support_batch))
                feed_dict.update({
                    agents[i].model.noise: noise,
                    agents[i].model.trans_model.x: np.row_stack(trans_input_x_batch),
                    agents[i].model.trans_model.y: np.row_stack(
                        trans_input_y_batch)
                })
                feed_dict.update(gcn_feed_dict)
                # embeddings = tf.get_default_session().run(self.model.embeddings, feed_dict=feed_dict)
                # obs_placeholder_tmp[place] = embeddings
            feed_dict.update({self.model.edges_labels: a[agent_index],
                              self.model.nodes_labels: x[agent_index],
                              # self.model.embeddings: embeddings[agent_index],
                              self.model.training: training,
                              self.model.dropout_rate: dropout_rate,
                              #  self.obs_placeholder: obs_n,
                              #  self.act_placeholder: act_n
                              })

            # feed_dict.update(obs_placeholder_tmp)

            p_loss, _ = tf.get_default_session().run(
                self.p_train_D, feed_dict=feed_dict)  # 更新Actor当前网络G
            logging.info("# Updating Agent_%s's Actor_D..." % agent_index)
            # p_loss, _ = self.p_train_D(
            #     *(obs_n + act_n + embeddings + a[index] + x[index] + training + dropout_rate))  # 更新Actor当前网络D
            # p_loss = self.p_train_D(
            #     *(obs_n + act_n + embeddings + a[index] + x[index] + training + dropout_rate))  # 更新Actor当前网络D
        if  t % 30 == 0:  # only update every 100 steps
            # gcn_opt = self.model.gcn_model.optimizer.minimize(self.q_train[0], var_list=U.scope_vars("p_func/gcn"))
            # trans_opt = self.model.trans_model.optimizer.minimize(self.q_train[0], var_list=U.scope_vars("p_func/transformer"))
            tf.get_default_session().run([self.gcn_opt, self.trans_opt], feed_dict=q_loss_feed_dict)
            logging.info("# Updating Agent_%s's GCN and Self-Attention Networks..." % agent_index)
        elif t % 100 == 0 or t % 300 == 0:
            self.p_update_D()
            self.p_update_G()
            self.p_update_gcn()
            self.p_update_trans()
            self.q_update()
            logging.info("# Updating Agent_%s's Target Networks..." % agent_index)

        

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
