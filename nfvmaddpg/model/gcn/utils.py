import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh  # 稀疏矩阵中查找特征值/特征向量的函数
import sys
from scipy.sparse import lil_matrix, coo_matrix
import functools

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l): # 构造样本掩码
    """Create mask."""
    """
    :param idx: 有标签样本的索引列表
    :param l: 所有样本数量
    :return: 布尔类型数组，其中有标签样本所对应的位置为True，无标签样本所对应的位置为False
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    with open("gcn/data/netGraph_abilene.pickle", 'rb') as f:
        if sys.version_info > (3, 0):
            graph = dict(pkl.load(f, encoding='latin1'))
        else:
            graph = dict(pkl.load(f))
    # self.g['nodes'] = list(self.V)
    # self.g['edges'] = list(self.E)
    # self.g['nodeCap'] = dict(self.c)
    # self.g['edgeDatarate'] = dict(self.d)
    # self.g['edgeLatency'] = dict(self.l)
    # self.g['F'] = list(self.F)
    # self.g['n_ins'] = dict(self.n_ins)
    # self.g['n_use'] = dict(self.n_use)
    # self.g['c_req'] = dict(self.c_req)
    features = dict(graph['nodeCap']) # dict
    edge_features = dict(graph['edgeDatarate'])
    edge_features_latency = dict(graph['edgeLatency'])
    nodes = graph['nodes']
    edges = graph['edges'] # (u,v)
    graph_topo = dict()
    for item in edges:
        if graph_topo.get(item[0]) == None:
            graph_topo[item[0]] = [item[1]]
        else:
            graph_topo[item[0]].append(item[1])
    for item in nodes:
        if graph_topo.get(item) == None:
            graph_topo[item] = []
    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']  # 替换为vnf图的x和graph
    # objects = []
    # for i in range(len(names)):
    #     with open("gcn/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))

    # x, y, tx, ty, allx, ally, graph = tuple(objects) # x为稀疏矩阵cs
    # test_idx_reorder = parse_index_file("gcn/data/ind.{}.test.index".format(dataset_str))
    # test_idx_range = np.sort(test_idx_reorder)

    # if dataset_str == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended
    #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #     ty_extended[test_idx_range-min(test_idx_range), :] = ty
    #     ty = ty_extended

    # features = sp.vstack((allx, tx)).tolil()   # feature稀疏矩阵
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # l = lil_matrix((len(features),1))
    # for k,v in features.items():
    #     l[k,0] = v
    # features = l
    # l = np.array((len(edge_features), len(edge_features), 1))
    # for k, v in edge_features.items():
    #     l[k[0], k[1], 0] = v     
    # edge_features = l
    G = nx.from_dict_of_lists(graph_topo)
    for node in G:
        G.nodes[node]['nodeCap'] = features[node]
        G.nodes[node]['nodeMem'] = np.random.randint(10)
        G.nodes[node]['nodeRel'] = 0.2 * np.random.random() + 0.8
        G.nodes[node]['nodeCostCap'] = np.random.random()
        G.nodes[node]['nodeCostMem'] = np.random.random()
        G.nodes[node]['nodeCostRun'] = np.random.random()
        G.nodes[node]['nodeCostAct'] = np.random.random()
        G.nodes[node]['vnf_instances'] = [0] * 17
    features = np.array([[G.nodes[node]['nodeCap'], G.nodes[node]['nodeMem'], G.nodes[node]['nodeRel'], G.nodes[node]['nodeCostCap'],
                          G.nodes[node]['nodeCostMem'], G.nodes[node]['nodeCostRun'], G.nodes[
        node]['nodeCostAct'], *G.nodes[node]['vnf_instances']
    ] for node in list(G.nodes)], dtype=np.float)
    features = lil_matrix(features)
    # l = lil_matrix((len(G.nodes), 1))
    # for i, (u, wt) in enumerate(G.nodes.data('nodeCap')):
    #     l[i, 0] = wt
    # features = l
    # test_adj = np.matrix(np.zeros((3,3)))
    # test_adj = test_adj + np.eye(test_adj.shape[0])
    # test_G = nx.from_numpy_matrix(test_adj)
    line_G = nx.line_graph(G)
    for node in line_G:
        line_G.nodes[node]['edgeDatarate'] = edge_features[node]
        line_G.nodes[node]['edgeLatency'] = edge_features_latency[node]
    edge_l = lil_matrix((len(line_G.nodes), 2))
    for i, (u, wt) in enumerate(line_G.nodes.data('edgeDatarate')):
        edge_l[i,0] = wt
    for i, (u, wt) in enumerate(line_G.nodes.data('edgeLatency')):
        edge_l[i,1] = wt
    edge_features = edge_l


    line_G.add_edges_from(list(line_G.edges), weight=1)
    # line_adj = nx.adjacency_matrix(line_G)
    # print(line_adj.todense())
    def _mynode_func(G):
        """Returns a function which returns a sorted node for line graphs.

        When constructing a line graph for undirected graphs, we must normalize
        the ordering of nodes as they appear in the edge.

        """
        if G.is_multigraph():
            def sorted_node(u, v, key):
                return (u, v, key) if u <= v else (v, u, key)
        else:
            def sorted_node(u, v):
                return (u, v) if u <= v else (v, u)
        return sorted_node
    degree_array = np.array([G.degree[node] for node in G]) 
    degree_variance = np.var(degree_array)
    mysorted_node = _mynode_func(G)

    for node in G:
        line_edges = [mysorted_node(*x) for x in G.edges(node)]
        def comp(a, b):
            if b[0] == a[0]:
                return a[1] - b[1]
            else:
                return a[0]-b[0]
        line_edges = sorted(line_edges, key=functools.cmp_to_key(comp))
        # line_edges = [x for x in G.edges(node)]
        if len(line_edges) <= 1:
            continue
        for i, line_edge in enumerate(line_edges):
            for ii, inv_line_edge in enumerate(reversed(line_edges)):
                if ii < len(line_edges)-1-i and inv_line_edge[0] == line_edge[1] and inv_line_edge[1] != line_edge[0]:
                    line_G.edges[line_edge, inv_line_edge]['weight'] = np.exp(-np.power(
                        (G.degree(inv_line_edge[0]) - 2), 2) / degree_variance)
                if ii < len(line_edges)-1-i and ((inv_line_edge[0] == line_edge[0] and inv_line_edge[1] != line_edge[1]) or (inv_line_edge[1] == line_edge[1] and inv_line_edge[0] != line_edge[0])):
                    line_G.edges[line_edge, inv_line_edge]['weight'] = np.exp(
                        -np.power(((G.degree(inv_line_edge[1]) + G.degree(line_edge[1])) / 2 - 2), 2) / degree_variance)
    line_adj = nx.adjacency_matrix(line_G) 
    # print(line_adj.todense())
    # graph为字典，键为节点index 值为相连的节点index 先转换为Graph类 再返回邻接矩阵
    adj = nx.adjacency_matrix(G)
    # print(adj.todense())
    # print(list(G.nodes), list(line_G.nodes))
    # print(adj[0,2])
    pm = np.zeros((adj.shape[0], line_adj.shape[0]))

    node_list = list(G.nodes)
    edge_list = list(line_G.nodes)
    for num, prev in enumerate(node_list):
        for post in range(num+1, len(node_list)):
            if adj[num, post] == 1:
                edge_index = edge_list.index((prev, node_list[post]))
                pm[num][edge_index] = 1
                pm[post][edge_index] = 1
    # print(pm)
            
    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)

    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    return adj, features, line_adj, edge_features, pm

def sparse_to_tuple(sparse_mx): # 将矩阵转换成tuple格式并返回
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
            # 将稀疏矩阵转化为coo矩阵形式
            # coo矩阵采用三个数组分别存储行、列和非零元素值的信息
        coords = np.vstack((mx.row, mx.col)).transpose()   # 获取非零元素的位置索引
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features): # 处理特征:将特征进行归一化并返回tuple (coords, values, shape)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    features = r_mat_inv.dot(features)
    return features
    # return sparse_to_tuple(features)

def process_pm(pm):
    return pm
    # return sparse_to_tuple(coo_matrix(pm))


def normalize_adj(adj): # 图归一化并返回
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj): # 处理得到GCN中的归一化矩阵并返回  在邻接矩阵中加入自连接
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))  
    return adj_normalized
    # return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, placeholders):   # 构建输入字典并返回
    """Construct feed dictionary."""
    feed_dict = dict()
    # feed_dict.update({placeholders['labels']: labels})
    # feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict
# def construct_feed_dict(features, support, labels, labels_mask, placeholders):   # 构建输入字典并返回
#     """Construct feed dictionary."""
#     feed_dict = dict()
#     # feed_dict.update({placeholders['labels']: labels})
#     # feed_dict.update({placeholders['labels_mask']: labels_mask})
#     feed_dict.update({placeholders['features']: features})
#     feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
#     feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
#     return feed_dict


def chebyshev_polynomials(adj, k): # 切比雪夫多项式近似:计算K阶的切比雪夫近似矩阵
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)  # D^{-1/2}AD^{1/2}
    laplacian = sp.eye(adj.shape[0]) - adj_normalized  # L = I_N - D^{-1/2}AD^{1/2}
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')  # \lambda_{max}
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])   # 2/\lambda_{max}L-I_N

    # 将切比雪夫多项式的 T_0(x) = 1和 T_1(x) = x 项加入到t_k中
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    # 依据公式 T_n(x) = 2xT_n(x) - T_{n-1}(x) 构造递归程序，计算T_2 -> T_k项目
    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


if __name__ == '__main__':
    load_data('test')     
