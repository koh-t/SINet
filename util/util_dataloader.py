# encoding: utf-8
# !/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd

import util.util_globalgraph as util_globalgraph
import util.util_seatgraph as util_seatgraph
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class ShinkokuDataset(Dataset):
    def __init__(self, Nguide=4, mode='train', id='', expid=0, args={}):
        self.args = args
        dirpath = './data/'
        treatpath = f'{dirpath}/experiment/dataset_{expid}/'

        # ---------- ---------- ---------- #
        self.dirpath = dirpath
        self.treatpath = treatpath

        self.mode = mode
        self.id = id

        # ------------------- #
        self.seatname = pd.read_csv(
            f'{dirpath}/source/seatname_loc.csv', index_col=0)

        with open(f'{dirpath}/source/pop_same.pkl', 'rb') as f:
            self.same_pop = pickle.load(f)

        # convert covariate into a graph
        self.getseatgraph = util_seatgraph.Graph(self.seatname, args)
        # ------------------- #

        # ------------------- #
        _type = 'multinomial'
        a = 1.0
        self.facutual_id = np.loadtxt('%sfactual_id_%s_a_%.1f.csv' %
                                      (self.treatpath, _type, a))
        self.facutual_id_inset = np.loadtxt('%sfactual_id_inset_%s_a_%.1f.csv' %
                                            (self.treatpath, _type, a))
        self.valid_id = np.loadtxt('%sfactual_id_%s_a_%.1f.csv' %
                                   (self.treatpath, _type, a))

        # ------------------- #
        # treatments
        self.treatment_unique = np.loadtxt(
            dirpath + 'source/treatment_unique.csv', delimiter=',')
        # ------------------- #
        self.treatment_unique = self.get_unique()

        # ------------------- #
        self.same_pop, self.treatment_id, self.treatment_unique = self.transform_samepop_treatment_unizue(
            self.same_pop, self.treatment_unique, Nguide)

        # covariate name
        self.imgname = ['oh1f', 'oh2f', 'oh3f', 'oh4f', 'ph1f', 'ph2f', 'tf']
        node = pd.read_csv(dirpath + '/source/node_coord.csv')
        edge = pd.read_csv(dirpath + '/source/node_node_distance_edit.csv')
        self.node = node
        self.edge = edge
        self.graph = util_globalgraph.Graph(args)

        self.guide = ['J_TP', 'J_PH_2F_l', 'J_PH_2F_r', 'J_OH_2F_l',
                      'J_OH_2F_r', 'J_OH_3F_l', 'J_OH_3F_r', 'J_OH_4F_l', 'J_OH_4F_r']
        self.guide_node = self.node.iloc[6:15, 1:]
        scaler = MinMaxScaler()
        scaler.fit(self.guide_node)
        self.guide_node = scaler.transform(self.guide_node)
        # ------------------- #

    def __len__(self):
        if type(self.id) == np.array:
            return len(self.facutual_id)
        else:
            return len(self.id)

    def __getitem__(self, idx):
        if self.mode == 'train':
            sample = self.get_train(idx)
        elif self.mode == 'valid':
            sample = self.get_valid(idx)
        else:
            sample = self.get_test(idx)

        return sample

    def set_train(self):
        self.mode = 'train'

    def set_valid(self):
        self.mode = 'valid'

    def set_test(self):
        self.mode = 'test'

    def get_traintest(self):
        return self.mode

    def get_train(self, idx):
        idx = self.id[idx]
        idx = int(idx)
        _idx = int(self.facutual_id[idx])

        fname = self.dirpath + 'sample/xz/xz_' + str(_idx) + '.pkl'
        with open(fname, 'rb') as f:
            xz = pickle.load(f).toarray()[0]
        x = xz[:-9]
        x[x != 0] = 1
        z = xz[-9:]
        z = z.astype(np.float32)

        x = self.getseatgraph.get(x)
        for (i, _img) in enumerate(x):
            x[i] = _img.astype(np.float32)

        m = np.loadtxt(self.dirpath + 'sample/y/outcome_' +
                       str(_idx) + '.csv', delimiter=',').astype(np.float32)

        sample = {'oh1f': x[0], 'oh2f': x[1], 'oh3f': x[2], 'oh4f': x[3],
                  'ph': x[4], 'tf': x[5],
                  'treatment': z, 'outcome': m, 'mean': m}

        return sample

    def get_valid(self, idx):
        idx = self.id[idx]
        idx = int(idx)
        _idx = int(self.valid_id[idx])

        fname = self.dirpath + 'sample/xz/xz_' + str(_idx) + '.pkl'
        with open(fname, 'rb') as f:
            xz = pickle.load(f).toarray()[0]
        x = xz[:-9]
        x[x != 0] = 1
        z = xz[-9:]
        z = z.astype(np.float32)

        x = self.getseatgraph.get(x)
        for (i, _img) in enumerate(x):
            x[i] = _img.astype(np.float32)

        m = np.loadtxt(self.dirpath + 'sample/y/outcome_' +
                       str(_idx) + '.csv', delimiter=',').astype(np.float32)
        sample = {'oh1f': x[0], 'oh2f': x[1], 'oh3f': x[2], 'oh4f': x[3],
                  'ph': x[4], 'tf': x[5],
                  'treatment': z, 'outcome': m, 'mean': m}
        return sample

    def get_test(self, idx):
        idx = self.id[idx]
        idx = int(idx)
        same_pop_id = self.same_pop[idx]
        fname = self.dirpath + 'sample/xz/xz_' + str(same_pop_id[0]) + '.pkl'
        with open(fname, 'rb') as f:
            xz = pickle.load(f).toarray()[0]
        x = xz[:-9]
        x[x != 0] = 1

        z = self.treatment_unique
        z = z.astype(np.float32)
        x = self.getseatgraph.get(x)
        for (i, _img) in enumerate(x):
            x[i] = _img.astype(np.float32)

        mean = []
        outcome = []
        for _idx in same_pop_id:
            _mean = np.loadtxt(self.dirpath + 'sample/y/outcome_' +
                               str(_idx) + '.csv', delimiter=',').astype(np.float32)
            _outcome = np.loadtxt(self.dirpath + 'sample/y/outcome_pois_' +
                                  str(_idx) + '.csv', delimiter=',').astype(np.float32)
            mean.append(_mean)
            outcome.append(_outcome)

        m = np.array(mean).squeeze()

        sample = {'oh1f': x[0], 'oh2f': x[1], 'oh3f': x[2], 'oh4f': x[3],
                  'ph': x[4], 'tf': x[5],
                  'treatment': z, 'outcome': m, 'mean': m}

        return sample

    def get_unique(self):
        _XZ = []
        id = self.same_pop[0]
        for _id in id:
            fname = self.dirpath + 'sample/xz/xz_' + str(_id) + '.pkl'
            with open(fname, 'rb') as f:
                xz = pickle.load(f).toarray()[0]
            _XZ.append(xz)

        XZ = np.c_[_XZ]
        Z = XZ[:, -9:]

        return Z

    def transform_samepop_treatment_unizue(self, _same_pop, _treatment_unique, Nguide):
        treatment_id = _treatment_unique.sum(1) == Nguide
        treatment_unique = _treatment_unique[treatment_id]
        same_pop = [x[treatment_id] for x in _same_pop]

        return same_pop, treatment_id, treatment_unique
