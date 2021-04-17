import pickle
import numpy as np

from rdkit import Chem

if __name__ == '__main__':
    from progress_bar import ProgressBar
else:
    from utils.progress_bar import ProgressBar

from datetime import datetime

# todo
# 可以根据此类 生成vnf网络拓扑数据集  为整个拓扑及其所有子集
# 最终需要生成的是 在对sfc嵌入后的整个网络拓扑（大小与整个网络拓扑一致，但是实质只需要嵌入对应的部分拓扑 即为整个拓扑的子集）

class SparseMolecularDataset():

    def load(self, filename, subset=1):

        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        self.train_idx = np.random.choice(self.train_idx, int(len(self.train_idx) * subset), replace=False) # size：106537
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset), # size: 13317
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx, int(len(self.test_idx) * subset), replace=False)    # size: 13317

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)  # 对象的__dict__保存了对象中的self.属性

    def generate(self, filename, add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        self.log('Extracting {}..'.format(filename))

        if filename.endswith('.sdf'):
            self.data = list(filter(lambda x: x is not None, Chem.SDMolSupplier(filename)))
        elif filename.endswith('.smi'):
            self.data = [Chem.MolFromSmiles(line) for line in open(filename, 'r').readlines()]

        self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data
        self.data = list(filter(filters, self.data))
        self.data = self.data[:size]

        self.log('Extracted {} out of {} molecules {}adding Hydrogen!'.format(len(self.data),
                                                                              len(Chem.SDMolSupplier(filename)),
                                                                              '' if add_h else 'not '))

        self._generate_encoders_decoders() # 生成原子序号、键类型、分子表达式与index的对应
        self._generate_AX() # 创建特征和邻接矩阵

        # it contains the all the molecules stored as rdkit.Chem objects
        self.data = np.array(self.data)

        # it contains the all the molecules stored as SMILES strings
        self.smiles = np.array(self.smiles)

        # a (N, L) matrix where N is the length of the dataset and each L-dim vector contains the 
        # indices corresponding to a SMILE sequences with padding wrt the max length of the longest 
        # SMILES sequence in the dataset (see self._genS)
        self.data_S = np.stack(self.data_S)

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

        self.vertexes = self.data_F.shape[-2]  # 分子中最大原子数
        self.features = self.data_F.shape[-1]  # 54维

        self._generate_train_validation_test(validation, test)

    def _generate_encoders_decoders(self):
        self.log('Creating atoms encoder and decoder..')
        atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0])) # 获取原子序号
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)} # 序号到索引
        self.atom_decoder_m = {i: l for i, l in enumerate(
            atom_labels)}  # {0: 0, 1: 6, 2: 7, 3: 8, 4: 9}
        self.atom_num_types = len(atom_labels)  # 原子种类数目 5
        self.log('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types - 1))

        self.log('Creating bonds encoder and decoder..')
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()  # 获取键的类型
                                                                    for mol in self.data
                                                                    for bond in mol.GetBonds())))

        # {rdkit.Chem.rdchem.BondType(1): 1, rdkit.Chem.rdchem.BondType(2): 2, rdkit.Chem.rdchem.BondType(3): 3, rdkit.Chem.rdchem.BondType(12): 4, rdkit.Chem.rdchem.BondType(21): 0}
        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)  # 5
        self.log('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))

        self.log('Creating SMILES encoder and decoder..')
        smiles_labels = ['E'] + list(set(c for mol in self.data for c in Chem.MolToSmiles(mol))) # 化合物的表达式
        # {'#': 19, '(': 15, ')': 1, '+': 17, '-': 5, '.': 24, '/': 12, '1': 8, '2': 20, '3': 9, '4': 13, '5': 3, '6': 10, '=': 4, ...}
        self.smiles_encoder_m = {l: i for i, l in enumerate(smiles_labels)}
        self.smiles_decoder_m = {i: l for i, l in enumerate(smiles_labels)}
        self.smiles_num_types = len(smiles_labels) # 27
        self.log('Created SMILES encoder and decoder with {} types and 1 PAD symbol!'.format(
            self.smiles_num_types - 1))

    def _generate_AX(self):
        self.log('Creating features and adjacency matrices..')
        pr = ProgressBar(60, len(self.data)) # 创建进度条 长度为60 相应的最大值为data长度

        data = []     # 分子集
        smiles = []   # 表达式
        data_S = []   # 表达式的index表示
        data_A = []   # 邻接矩阵
        data_X = []   # 分子中原子的index
        data_D = []   # 度矩阵
        data_F = []   # 特征值
        data_Le = []  # 拉普拉斯矩阵特征值
        data_Lv = []  # 拉普拉斯矩阵特征向量

        max_length = max(mol.GetNumAtoms() for mol in self.data)  # 分子最大的原子数
        max_length_s = max(len(Chem.MolToSmiles(mol)) for mol in self.data)  # 分子最长的表达式

        for i, mol in enumerate(self.data):  # 对于每个分子，获得对应的邻接矩阵和度矩阵以及特征
            A = self._genA(mol, connected=True, max_length=max_length)   # 获取对应邻接矩阵
            D = np.count_nonzero(A, -1)  # 获取对应原子的度
            if A is not None:
                data.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                data_S.append(self._genS(mol, max_length=max_length_s))  # 获取表达式的index表示
                data_A.append(A)
                data_X.append(self._genX(mol, max_length=max_length))    # 获取分子中原子的index
                data_D.append(D)
                data_F.append(self._genF(mol, max_length=max_length))    # 获取分子的特征值

                L = np.diag(D) - A
                Le, Lv = np.linalg.eigh(L)

                data_Le.append(Le)
                data_Lv.append(Lv)

            pr.update(i + 1)

        self.log(date=False)
        self.log('Created {} features and adjacency matrices  out of {} molecules!'.format(len(data),
                                                                                           len(self.data)))

        self.data = data  # (133171,)
        self.smiles = smiles # 
        self.data_S = data_S  # (133171, 37)
        self.data_A = data_A  # (133171, 9, 9)
        self.data_X = data_X  # (133171, 9)
        self.data_D = data_D  # (133171, 9)
        self.data_F = data_F  # (133171, 9, 54)
        self.data_Le = data_Le  # (133171, 9)
        self.data_Lv = data_Lv  # (133171, 9, 9)
        self.__len = len(self.data)

    def _genA(self, mol, connected=True, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)

        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()] #获取每条键相连的原子index
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]  # 根据键类型获取对应的index

        A[begin, end] = bond_type  # 邻接矩阵值为对应键类型的index
        A[end, begin] = bond_type

        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1) # 得到每个原子的度

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (
                    max_length - mol.GetNumAtoms()), dtype=np.int32) # 获取分子中原子的index 并以0填充

    def _genS(self, mol, max_length=None):

        max_length = max_length if max_length is not None else len(Chem.MolToSmiles(mol))

        return np.array([self.smiles_encoder_m[c] for c in Chem.MolToSmiles(mol)] + [self.smiles_encoder_m['E']] * (
                    max_length - len(Chem.MolToSmiles(mol))), dtype=np.int32) # 获取表达式的index表示  并用e填充

    def _genF(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        features = np.array([[*[a.GetDegree() == i for i in range(5)],  # true false 长度为5
                              *[a.GetExplicitValence() == i for i in range(9)],  # 获取原子显式化合价
                              *[int(a.GetHybridization()) == i for i in range(1, 7)], # 获取原子杂化方式
                              *[a.GetImplicitValence() == i for i in range(9)],  # 获取原子隐式化合价
                              a.GetIsAromatic(),   # 是否为芳香
                              a.GetNoImplicit(),
                              *[a.GetNumExplicitHs() == i for i in range(5)],
                              *[a.GetNumImplicitHs() == i for i in range(5)],
                              *[a.GetNumRadicalElectrons() == i for i in range(5)],
                              a.IsInRing(),  # 是否在环中
                              *[a.IsInRingSize(i) for i in range(2, 9)]] for a in mol.GetAtoms()], dtype=np.int32)  # 是否在n元环中

        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))  # 返回分子的特征值 不足填充

    def matrices2mol(self, node_labels, edge_labels, strict=False):
        mol = Chem.RWMol()  # 用于分子读写的类
 
        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))  # 添加原子

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(
                    end), self.bond_decoder_m[edge_labels[start, end]])  # 添加键

        if strict:
            try:
                Chem.SanitizeMol(mol)  # 对分子进行检查
            except:
                mol = None

        return mol

    def seq2mol(self, seq, strict=False):
        mol = Chem.MolFromSmiles(''.join([self.smiles_decoder_m[e] for e in seq if e != 0]))

        if strict:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

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
                      for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
                                  self.data_D, self.data_F, self.data_Le, self.data_Lv)]

            counter += batch_size
        else:
            output = [obj[idx] for obj in (self.data, self.smiles, self.data_S, self.data_A, self.data_X,
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
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' + str(msg) if date else str(msg))

    def __len__(self):
        return self.__len


if __name__ == '__main__':
    data = SparseMolecularDataset()
    data.generate('data/gdb9.sdf', filters=lambda x: x.GetNumAtoms() <= 9)
    data.save('data/gdb9_9nodes.sparsedataset')

    # data = SparseMolecularDataset()
    # data.generate('data/qm9_5k.smi', validation=0.00021, test=0.00021)  # , filters=lambda x: x.GetNumAtoms() <= 9)
    # data.save('data/qm9_5k.sparsedataset')
