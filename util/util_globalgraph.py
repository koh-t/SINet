# encoding: utf-8
# !/usr/bin/env python3
import torch
import argparse
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing
from matplotlib import pylab as plt
from sklearn.metrics import pairwise_distances

from util.util_stgraph import calculate_random_walk_matrix  # noqa


class Graph():
    def __init__(self, args):
        self.args = args
        # global graph
        A = pd.read_csv('./data/source/global_graph.csv', index_col=0)
        df = pd.read_csv(
            './data/source/global_graph_node_loc.csv', index_col=0)

        self.A = A

        # to use mean normalization
        # normalized_df=(df-df.mean())/df.std()
        # to use min-max normalization:
        normalized_df = (df-df.min())/(df.max()-df.min())
        self.node = torch.FloatTensor(normalized_df.to_numpy())

        self.node_name = df.index
        '''
        0          S_TP
        1          S_PH
        2       S_OH_5F
        3       S_OH_4F
        4       S_OH_3F
        5       S_OH_2F
        6          J_TP
        7     J_PH_2F_r
        8     J_PH_2F_l
        9     J_OH_4F_r
        10    J_OH_4F_l
        11    J_OH_3F_r
        12    J_OH_3F_l
        13    J_OH_2F_r
        14    J_OH_2F_l
        '''
        # global graph without treatment nodes
        B = pd.read_csv(
            './data/source/global_graph_covariate.csv', index_col=0)
        self.B = B

    def get_graph(self):
        return self.G

    def get_node(self):
        return self.node

    def get(self):
        A = torch.FloatTensor(self.A.to_numpy())
        A = [
            calculate_random_walk_matrix(A, self.args),
        ]
        return A

    def get_B(self):
        B = torch.FloatTensor(self.B.to_numpy())
        B = [
            calculate_random_walk_matrix(B, self.args),
        ]
        return self.B
