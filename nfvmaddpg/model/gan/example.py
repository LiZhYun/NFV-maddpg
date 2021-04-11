import tensorflow as tf

from utils.sparse_topo_dataset import SparseToPoDataset
from utils.trainer import Trainer
from utils.utils import *
from utils.vnf_metrics import VNFMetrics

from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn

from optimizers.gan import GraphGANOptimizer


batch_dim = 1
la = 1
dropout = 0.1
n_critic = 3
# metric = 'validity,sas'
metric = 'type_num'
n_samples = 50
z_dim = 4   # 嵌入后的维度
epochs = 2
save_every = 1 # May lead to errors if left as None

data = SparseToPoDataset()
data.load('data/vnftopo.sparsedataset')

steps = (len(data) // batch_dim)


def train_fetch_dict(i, steps, epoch, epochs, min_epochs, model, optimizer):
    a = [optimizer.train_step_G] if i % n_critic == 0 else [optimizer.train_step_D]
    b = [optimizer.train_step_V] if i % n_critic == 0 and la < 1 else []
    return a + b


def train_feed_dict(i, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
    topos,a, x, _, _, _, _ = data.next_train_batch(batch_dim)
    embeddings = model.sample_z(batch_dim)

    if la < 1:

        if i % n_critic == 0:
            rewardR = reward(topos)

            n, e = session.run([model.nodes_gumbel_softmax, model.edges_logits],
                               feed_dict={model.training: False, model.embeddings: embeddings})
            # n, e = np.argmax(n, axis=-1), e
            n, e = standardization(n), standardization(e)
            topos = [data.matrices2topo(n_, e_, strict=True)
                     for n_, e_ in zip(n, e)]

            rewardF = reward(topos)

            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.rewardR: rewardR,
                         model.rewardF: rewardF,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > 0 else 1.0}

        else:
            feed_dict = {model.edges_labels: a,
                         model.nodes_labels: x,
                         model.embeddings: embeddings,
                         model.training: True,
                         model.dropout_rate: dropout,
                         optimizer.la: la if epoch > 0 else 1.0}
    else:
        feed_dict = {model.edges_labels: a,
                     model.nodes_labels: x,
                     model.embeddings: embeddings,
                     model.training: True,
                     model.dropout_rate: dropout,
                     optimizer.la: 1.0}

    return feed_dict


def eval_fetch_dict(i, epochs, min_epochs, model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def eval_feed_dict(i, epochs, min_epochs, model, optimizer, batch_dim):
    topos, a, x, _, _, _, _ = data.next_validation_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(topos)  # (13317, 1)

    n, e = session.run([model.nodes_gumbel_softmax, model.edges_logits],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    # n, e = np.argmax(n, axis=-1), e
    n, e = standardization(n), standardization(e)
    topos = [data.matrices2topo(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    rewardF = reward(topos)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def test_fetch_dict(model, optimizer):
    return {'loss D': optimizer.loss_D, 'loss G': optimizer.loss_G,
            'loss RL': optimizer.loss_RL, 'loss V': optimizer.loss_V,
            'la': optimizer.la}


def test_feed_dict(model, optimizer, batch_dim):
    topos, a, x, _, _, _, _ = data.next_test_batch()
    embeddings = model.sample_z(a.shape[0])

    rewardR = reward(topos)

    n, e = session.run([model.nodes_gumbel_softmax, model.edges_logits],
                       feed_dict={model.training: False, model.embeddings: embeddings})
    # n, e = np.argmax(n, axis=-1), e
    n, e = standardization(n), standardization(e)
    topos = [data.matrices2topo(n_, e_, strict=True) for n_, e_ in zip(n, e)]
    topos2grid_image(topos, data.vnf_decoder)
    rewardF = reward(topos)

    feed_dict = {model.edges_labels: a,
                 model.nodes_labels: x,
                 model.embeddings: embeddings,
                 model.rewardR: rewardR,
                 model.rewardF: rewardF,
                 model.training: False}
    return feed_dict


def reward(topos):  # 改为vnf奖励
    rr = 1.
    for m in ('type_num' if metric == 'all' else metric).split(','):

        if m == 'type_num':
            rr *= VNFMetrics.vnftype_num(topos, data.vnf_decoder)
        # elif m == 'logp':
        #     rr *= VNFMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
        # elif m == 'sas':
        #     rr *= VNFMetrics.synthetic_accessibility_score_scores(mols, norm=True)
        # elif m == 'qed':
        #     rr *= VNFMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
        # elif m == 'novelty':
        #     rr *= VNFMetrics.novel_scores(mols, data)
        # elif m == 'dc':
        #     rr *= VNFMetrics.drugcandidate_scores(mols, data)
        # elif m == 'unique':
        #     rr *= VNFMetrics.unique_scores(mols)
        # elif m == 'diversity':
        #     rr *= VNFMetrics.diversity_scores(mols, data)
        # elif m == 'validity':
        #     rr *= VNFMetrics.valid_scores(mols)
        else:
            raise RuntimeError('{} is not defined as a metric'.format(m))

    return rr.reshape(-1, 1)


def _eval_update(i, epochs, min_epochs, model, optimizer, batch_dim, eval_batch):
    topos = samples(data, model, session, model.sample_z(n_samples), sample=True)
    m0 = all_scores(topos, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    return m0


def _test_update(model, optimizer, batch_dim, test_batch):
    topos = samples(data, model, session, model.sample_z(n_samples), sample=True)
    m0 = all_scores(topos, data, norm=True)
    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
    return m0


# model
model = GraphGANModel(data.vertexes,  # 拓扑中最大节点数 9
                      1,  # 键类型数 5
                      data.vnf_num_types,  # 原子种类数目 5
                      z_dim,   # 嵌入后的维度 8
                      decoder_units=(32, 64, 128, 256, 512),
                    #   discriminator_units=((128, 64), 128, (128, 64)),
                      discriminator_units=((32,), 16, (16,)),
                      decoder=decoder_adj, # 解码器
                      discriminator=encoder_rgcn,  # 判别器
                      soft_gumbel_softmax=False,
                      hard_gumbel_softmax=False,
                      batch_discriminator=False)

# optimizer
optimizer = GraphGANOptimizer(model, learning_rate=1e-3, feature_matching=False)

# session
session = tf.Session()
session.run(tf.global_variables_initializer())

# trainer
trainer = Trainer(model, optimizer, session)

print('Parameters: {}'.format(np.sum([np.prod(e.shape) for e in session.run(tf.trainable_variables())]))) # 计算参数总量 prod计算所有元素的乘积 575556

def best_fn(result):
    best_model_value = 0
    for k, v in result.items():
        if k == 'typenum score':
            continue
        best_model_value += -v
    return best_model_value

trainer.train(batch_dim=batch_dim, # 128
              epochs=epochs, # 10
              steps=steps, # 1040
              train_fetch_dict=train_fetch_dict,
              train_feed_dict=train_feed_dict,
              eval_fetch_dict=eval_fetch_dict,
              eval_feed_dict=eval_feed_dict,
              test_fetch_dict=test_fetch_dict,
              test_feed_dict=test_feed_dict,
              save_every=save_every, # 1
              directory='result', # here users need to first create and then specify a folder where to save the model
              _eval_update=_eval_update,
              _test_update=_test_update, 
              best_fn=best_fn,
              load_history=False)
