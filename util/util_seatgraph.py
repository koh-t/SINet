# encoding: utf-8
# !/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
from util.util_stgraph import calculate_random_walk_matrix  # noqa


class Graph():
    def __init__(self, seatname, args={}):
        self.args = args
        self.seatname = seatname
        self.oh1f_a = pd.read_csv(
            './data/source/oh1f_A.csv', index_col=0).to_numpy()
        self.oh2f_a = pd.read_csv(
            './data/source/oh2f_A.csv', index_col=0).to_numpy()
        self.oh3f_a = pd.read_csv(
            './data/source/oh3f_A.csv', index_col=0).to_numpy()
        self.oh4f_a = pd.read_csv(
            './data/source/oh4f_A.csv', index_col=0).to_numpy()
        self.ph_a = pd.read_csv(
            './data/source/ph_A.csv', index_col=0).to_numpy()
        self.tf_a = pd.read_csv(
            './data/source/tf_A.csv', index_col=0).to_numpy()

        self.oh1f_loc = pd.read_csv(
            './data/source/oh1f_loc.csv', index_col=0).to_numpy()
        self.oh2f_loc = pd.read_csv(
            './data/source/oh2f_loc.csv', index_col=0).to_numpy()
        self.oh3f_loc = pd.read_csv(
            './data/source/oh3f_loc.csv', index_col=0).to_numpy()
        self.oh4f_loc = pd.read_csv(
            './data/source/oh4f_loc.csv', index_col=0).to_numpy()
        self.ph_loc = pd.read_csv(
            './data/source/ph_loc.csv', index_col=0).to_numpy()
        self.tf_loc = pd.read_csv(
            './data/source/tf_loc.csv', index_col=0).to_numpy()

        self.oh1f_covariate_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == '1F'))[0]
        self.oh2f_covariate_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == '2F'))[0]
        self.oh3f_covariate_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == '3F'))[0]
        self.oh4f_covariate_id = np.where(
            (seatname['hall'] == 'OH') & (seatname['floor'] == '4F'))[0]
        self.ph_covariate_id = np.where((seatname['hall'] == 'PH'))[0]
        self.tf_covariate_id = np.where((seatname['hall'] == 'TP'))[0]

    def get_graph(self):
        oh1f = torch.FloatTensor(self.oh1f_a.astype(np.float32))
        oh2f = torch.FloatTensor(self.oh2f_a.astype(np.float32))
        oh3f = torch.FloatTensor(self.oh3f_a.astype(np.float32))
        oh4f = torch.FloatTensor(self.oh4f_a.astype(np.float32))
        ph = torch.FloatTensor(self.ph_a.astype(np.float32))
        tf = torch.FloatTensor(self.tf_a.astype(np.float32))
        A = {'oh1f': oh1f, 'oh2f': oh2f, 'oh3f': oh3f,
             'oh4f': oh4f, 'ph': ph, 'tf': tf}
        for key in A.keys():
            A[key] = [
                calculate_random_walk_matrix(A[key], self.args),
            ]

        return A

    def get_node(self):
        oh1f = torch.FloatTensor(self.oh1f_node.astype(np.float32))
        oh2f = torch.FloatTensor(self.oh2f_node.astype(np.float32))
        oh3f = torch.FloatTensor(self.oh3f_node.astype(np.float32))
        oh4f = torch.FloatTensor(self.oh4f_node.astype(np.float32))
        ph = torch.FloatTensor(self.ph_node.astype(np.float32))
        tf = torch.FloatTensor(self.tf_node.astype(np.float32))
        ret = {'oh1f': oh1f, 'oh2f': oh2f, 'oh3f': oh3f,
               'oh4f': oh4f, 'ph': ph, 'tf': tf}
        return ret

    def get(self, x):
        if not type(x) == np.ndarray:
            x = x.to_numpy()

        oh1f = x[self.oh1f_covariate_id]
        oh2f = x[self.oh2f_covariate_id]
        oh3f = x[self.oh3f_covariate_id]
        oh4f = x[self.oh4f_covariate_id]
        ph = x[self.ph_covariate_id]
        tf = x[self.tf_covariate_id]

        return [oh1f, oh2f, oh3f, oh4f, ph, tf]
