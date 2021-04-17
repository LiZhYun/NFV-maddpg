import pickle
import numpy as np
import sys
import pickle as pkl 
import networkx as nx 
import scipy.sparse as sp 
from scipy.sparse.linalg.eigen.arpack import eigsh  # 稀疏矩阵中查找特征值/特征向量的函数
from scipy.sparse import lil_matrix, csr_matrix
import itertools
import matplotlib.pyplot as plt
from rdkit import Chem

if __name__ == '__main__':
    from progress_bar import ProgressBar
else:
    from utils.progress_bar import ProgressBar

from datetime import datetime

# todo
# 可以根据此类 生成vnf网络拓扑数据集  为整个拓扑及其所有子集
# 最终需要生成的是 在对sfc嵌入后的整个网络拓扑（大小与整个网络拓扑一致，但是实质只需要嵌入对应的部分拓扑 即为整个拓扑的子集）


class SparseToPoDataset():

    def load(self, filename, subset=1):

        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        self.train_idx = np.random.choice(self.train_idx, int(
            len(self.train_idx) * subset), replace=False)
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset),
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx, int(
            len(self.test_idx) * subset), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count
    
    def load_data(self):
        """
        根据网络拓扑，生成所有连通子图

        :param 
        :return: 所有连通子图，类型为networkx.Graph的列表
        """

        with open("data/netGraph_abilene.pickle", 'rb') as f:
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
        features = dict(graph['nodeCap'])  # dict
        edge_features = dict(graph['edgeDatarate'])
        nodes = graph['nodes']
        edges = graph['edges']  # (u,v)
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
        l = lil_matrix((len(features), 1))
        for k, v in features.items():
            l[k, 0] = v
        self.features = l
        # l_edges = lil_matrix((len(edge_features), 1))
        # for k, v in edge_features.items():
        #     l_edges[k, 0] = v
        # self.edge_features = l_edges
        # graph为字典，键为节点index 值为相连的节点index 先转换为Graph类 再返回邻接矩阵
        G = nx.from_dict_of_lists(graph_topo)
        self.vocab = [line.split()[0] for line in open(
            'E:/MANFV/NFVGAN/data/vnf.vocab', 'r', encoding='utf-8', errors='ignore').read().splitlines()]
        for node in list(G.nodes):
            G.nodes[node]['nodeCap'] = self.features[node,0]
            vnf_random = np.random.randint(low=0,high=len(self.vocab))
            G.nodes[node]['vnftype'] = self.vocab[vnf_random]
            # print(G.nodes[node])
        # i = 0
        for edge in list(G.edges):
            G.edges[edge[0],edge[1]]['edgeDatarate'] = edge_features[edge]
        #     print(G.edges[edge[1], edge[0]])
        #     i += 1
        # print(i)
        adj = nx.adjacency_matrix(G)
        all_connected_subgraphs = []
        # print(adj)
        # nx.draw(G)
        # plt.show()
        # here we ask for all connected subgraphs that have at least 2 nodes AND have less nodes than the input graph
        for nb_nodes in range(2, G.number_of_nodes()+1):
            for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G, nb_nodes)):
                if nx.is_connected(SG):
                    # print(SG.nodes)
                    # print(SG.edges.data())
                    all_connected_subgraphs.append(SG)
        self.data = all_connected_subgraphs
        # labels = np.vstack((ally, ty))
        # labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # plt.subplot(221)
        # nx.draw(all_connected_subgraphs[0])
        # plt.subplot(222)
        # nx.draw(all_connected_subgraphs[50])
        # plt.subplot(223)
        # nx.draw(all_connected_subgraphs[400])
        # #最后一幅子图转为有向图
        # plt.subplot(224)
        # # H = G.to_directed()
        # nx.draw(all_connected_subgraphs[-1])
        # # plt.savefig("four_grids.png")
        # plt.show()
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
        # return adj, features

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)  # 对象的__dict__保存了对象中的self.属性

    def generate(self, validation=0.1, test=0.1):
        # self.log('Extracting {}..'.format(filename))

        # if filename.endswith('.sdf'):
        #     self.data = list(filter(lambda x: x is not None,
        #                             Chem.SDMolSupplier(filename)))
        # elif filename.endswith('.smi'):
        #     self.data = [Chem.MolFromSmiles(
        #         line) for line in open(filename, 'r').readlines()]

        # self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data
        # self.data = list(filter(filters, self.data))
        # self.data = self.data[:size]

        self.log('Extracing {} Subgraph!'.format(len(self.data)))

        self._generate_encoders_decoders()  # 生成每个节点上处理的vnf实例类型与index的对应
        self._generate_AX()  # 创建特征和邻接矩阵

        # it contains the all the molecules stored as rdkit.Chem objects
        self.data = np.array(self.data)

        # it contains the all the molecules stored as SMILES strings
        # self.smiles = np.array(self.smiles)

        # a (N, L) matrix where N is the length of the dataset and each L-dim vector contains the
        # indices corresponding to a SMILE sequences with padding wrt the max length of the longest
        # SMILES sequence in the dataset (see self._genS)
        # self.data_S = np.stack(self.data_S)

        # a (N, 9, 9) tensor where N is the length of the dataset and each 9x9 matrix contains the
        # indices of the positions of the ones in the one-hot representation of the adjacency tensor
        # (see self._genA)
        self.data_A = np.stack(self.data_A)

        # a (N, 9) matrix where N is the length of the dataset and each 9-dim vector contains the
        # indices of the positions of the ones in the one-hot representation of the annotation matrix
        # (see self._genX)
        self.data_X = np.stack(self.data_X)

        # a (N, 9) matrix where N is the length of the dataset and each  9-dim vector contains the
        # diagonal of the correspondent adjacency matrix
        self.data_D = np.stack(self.data_D)

        # a (N, F) matrix where N is the length of the dataset and each F vector contains features
        # of the correspondent molecule (see self._genF)
        self.data_F = np.stack(self.data_F)

        # a (N, 9) matrix where N is the length of the dataset and each  9-dim vector contains the
        # eigenvalues of the correspondent Laplacian matrix
        self.data_Le = np.stack(self.data_Le)

        # a (N, 9, 9) matrix where N is the length of the dataset and each  9x9 matrix contains the
        # eigenvectors of the correspondent Laplacian matrix
        self.data_Lv = np.stack(self.data_Lv)

        self.vertexes = self.data_F.shape[-2]
        self.features = self.data_F.shape[-1]

        self._generate_train_validation_test(validation, test)

    def _generate_encoders_decoders(self):  # 需要改  每个网络节点处理的vnf实例的index
        self.log('Creating vnf-type encoder and decoder..')
        # vocab = [line.split()[0] for line in open(
        #     'E:/MANFV/NFVGAN/data/vnf.vocab', 'r', encoding='utf-8', errors='ignore').read().splitlines()]
        self.vnf_encoder = {token: idx for idx, token in enumerate(self.vocab)}
        self.vnf_decoder = {idx: token for idx, token in enumerate(self.vocab)}
        # vnf_labels = sorted(set([atom.GetAtomicNum(
        # ) for mol in self.data for atom in mol.GetAtoms()] + [0]))  # 获取原子序号
        # self.atom_encoder_m = {l: i for i,
        #                        l in enumerate(atom_labels)}  # 序号到索引
        # self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.vnf_num_types = len(self.vocab)  # 原子种类数目
        self.log('Created vnf encoder and decoder with {} vnf types!'.format(
            self.vnf_num_types))

        # self.log('Creating bonds encoder and decoder..')
        # bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()  # 获取键的类型
        #                                                             for mol in self.data
        #                                                             for bond in mol.GetBonds())))

        # self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        # self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        # self.bond_num_types = len(bond_labels)
        # self.log('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
        #     self.bond_num_types - 1))

        # self.log('Creating SMILES encoder and decoder..')
        # smiles_labels = [
        #     'E'] + list(set(c for mol in self.data for c in Chem.MolToSmiles(mol)))  # 化合物的表达式
        # self.smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
        # self.smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
        # self.smiles_num_types = len(smiles_labels)
        # self.log('Created SMILES encoder and decoder with {} types and 1 PAD symbol!'.format(
        #     self.smiles_num_types - 1))

    def _generate_AX(self):
        self.log('Creating adjacency matrices..')
        pr = ProgressBar(60, len(self.data))  # 创建进度条 长度为60 相应的最大值为data长度

        data = []     # topo集
        # smiles = []   # 表达式
        # data_S = []   # 表达式的index表示
        data_A = []   # 邻接矩阵
        data_X = []   # 分子中原子的index
        data_D = []   # 度矩阵
        data_F = []   # 特征值
        data_Le = []  # 拉普拉斯矩阵特征值
        data_Lv = []  # 拉普拉斯矩阵特征向量

        max_length = len(self.data[-1].nodes)  # 拓扑最大的节点数
        # max_length_s = max(len(Chem.MolToSmiles(mol))
        #                    for mol in self.data)  # 分子最长的表达式

        for i, topo in enumerate(self.data):  # 对于每个拓扑，获得对应的邻接矩阵和度矩阵以及特征
            A = self._genA(topo, connected=True,
                           max_length=max_length)   # 获取对应邻接稀疏矩阵
            D = np.count_nonzero(A, -1)  # 获取对应原子的度
            if A is not None:
                data.append(topo)
                # smiles.append(Chem.MolToSmiles(mol))
                # # 获取表达式的index表示
                # data_S.append(self._genS(mol, max_length=max_length_s))
                data_A.append(A)
                # 获取分子中原子的index
                data_X.append(self._genX(topo, max_length=max_length))
                data_D.append(D)
                data_F.append(self._genF(
                    topo, max_length=max_length))    # 获取节点的特征值

                L = np.diag(D) - A
                Le, Lv = np.linalg.eigh(L)

                data_Le.append(Le)
                data_Lv.append(Lv)

            pr.update(i + 1)

        self.log(date=False)
        self.log('Created adjacency matrices  out of {} topo!'.format(len(self.data)))

        self.data = data
        # self.smiles = smiles
        # self.data_S = data_S
        self.data_A = data_A
        self.data_X = data_X
        self.data_D = data_D
        self.data_F = data_F
        self.data_Le = data_Le
        self.data_Lv = data_Lv
        self.__len = len(self.data)

    def _genA(self, topo, connected=True, max_length=None):

        # max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)
        # A = nx.adjacency_matrix(topo).todense()
        # A = A.resize(max_length, max_length)
        # begin, end = [edge[0] for edge in list(topo.edges)], [
        #     edge[1] for edge in list(topo.edges)]  # 获取每条链路相连的节点index
        # # bond_type = [self.bond_encoder_m[b.GetBondType()]
        # #              for b in mol.GetBonds()]  # 根据键类型获取对应的index

        # A[begin, end] = topo.edges[begin, end]['edgeDatarate']  # 邻接矩阵值为对应链路的带宽
        # A[end, begin] = topo.edges[end, begin]['edgeDatarate']
        for begin, end in zip([edge[0] for edge in list(topo.edges)], [
            edge[1] for edge in list(topo.edges)]):
            A[begin, end] = topo.edges[begin, end]['edgeDatarate']  # 邻接矩阵值为对应链路的带宽
            A[end, begin] = topo.edges[end, begin]['edgeDatarate']
        degree = A[np.where(A.any(axis=1))[0], :]
        degree = degree[:, np.where(A.any(axis=0))[0]]
        degree = np.sum(
            degree[:len(topo.nodes), :len(topo.nodes)], axis=-1)  # 得到每个原子的度

        return A if connected and (degree > 0).all() else None

    def _genX(self, topo, max_length=None):

        # max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array([self.vnf_encoder[topo.nodes[node]['vnftype']] for node in list(topo.nodes)] + [0] * (
                    max_length - len(list(topo.nodes))), dtype=np.int32)

    # def _genS(self, mol, max_length=None):

    #     max_length = max_length if max_length is not None else len(
    #         Chem.MolToSmiles(mol))

    #     return np.array([self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)] + [self.smiles_encoder_m['E']] * (
    #         max_length - len(Chem.MolToSmiles(mol))), dtype=np.int32)  # 获取表达式的index表示  并用e填充

    def _genF(self, topo, max_length=None):

        # max_length = max_length if max_length is not None else mol.GetNumAtoms()

        features = np.array([[topo.nodes[node]['nodeCap'],
                              ] for node in list(topo.nodes)], dtype=np.int32)

        # 返回分子的特征值 不足填充
        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

    def matrices2topo(self, node_labels, edge_labels, strict=False):
        G = nx.DiGraph()
        # mol = Chem.RWMol()  # 用于分子读写的类
        node_add = []
        for row in zip(*np.nonzero(node_labels)):
            vnftype = {'vnftype': self.vnf_decoder[node_labels[row]]}
            node_add.append((row[0],vnftype))
        G.add_nodes_from(node_add)  # 添加节点

        gnodes = list(G.nodes)
        for start, end, link in zip(*np.nonzero(edge_labels)):

            if start > end and start in gnodes and end in gnodes:
               G.add_edge(int(start), int(end))  # 添加链路
               G.edges[int(start), int(end)]['edgeDatarate'] = edge_labels[start, end, link]

        # if strict:
        #     try:
        #         Chem.SanitizeMol(mol)  # 对分子进行检查
        #     except:
        #         mol = None

        return G

    def seq2mol(self, seq, strict=False):
        pass
        # mol = Chem.MolFromSmiles(
        #     ''.join([self.smiles_decoder_m[e] for e in seq if e != 0]))

        # if strict:
        #     try:
        #         Chem.SanitizeMol(mol)
        #     except:
        #         mol = None

        # return mol

    def _generate_train_validation_test(self, validation, test):

        self.log('Creating train, validation and test sets..')

        validation = int(validation * len(self))
        test = int(test * len(self))
        train = len(self) - validation - test

        self.all_idx = np.random.permutation(len(self))
        self.train_idx = self.all_idx[0:train]
        self.validation_idx = self.all_idx[train:train + validation]
        self.test_idx = self.all_idx[train + validation:]

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = train
        self.validation_count = validation
        self.test_count = test

        self.log('Created train ({} items), validation ({} items) and test ({} items) sets!'.format(
            train, validation, test))

    def _next_batch(self, counter, count, idx, batch_size):
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            output = [obj[idx[counter:counter + batch_size]]
                      for obj in (self.data, self.data_A, self.data_X,
                                  self.data_D, self.data_F, self.data_Le, self.data_Lv)]

            counter += batch_size
        else:
            output = [obj[idx] for obj in (self.data, self.data_A, self.data_X,
                                           self.data_D, self.data_F, self.data_Le, self.data_Lv)]

        return [counter] + output

    def next_train_batch(self, batch_size=None):
        out = self._next_batch(counter=self.train_counter, count=self.train_count,
                               idx=self.train_idx, batch_size=batch_size)
        self.train_counter = out[0]

        return out[1:]

    def next_validation_batch(self, batch_size=None):
        out = self._next_batch(counter=self.validation_counter, count=self.validation_count,
                               idx=self.validation_idx, batch_size=batch_size)
        self.validation_counter = out[0]

        return out[1:]

    def next_test_batch(self, batch_size=None):
        out = self._next_batch(counter=self.test_counter, count=self.test_count,
                               idx=self.test_idx, batch_size=batch_size)
        self.test_counter = out[0]

        return out[1:]

    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) +
              ' ' + str(msg) if date else str(msg))

    def __len__(self):
        return self.__len


if __name__ == '__main__':
    data = SparseToPoDataset()
    data.load_data()
    data.generate()
    data.save('data/vnftopo.sparsedataset')

    # data = SparseMolecularDataset()
    # data.generate('data/qm9_5k.smi', validation=0.00021, test=0.00021)  # , filters=lambda x: x.GetNumAtoms() <= 9)
    # data.save('data/qm9_5k.sparsedataset')
