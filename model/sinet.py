
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from model.template import Proto

from logging import getLogger
# sample01で宣言したloggerの子loggerオブジェクトの宣言
logger = getLogger("Pytorch").getChild("model")


class MLP(nn.Module):
    """
    Multi Layer Perceptron with three layers
    """

    def __init__(self, in_features=25, out_features=2, hidden_features=[20, 20], dp=0.0, act='relu'):
        """
        Parameter:
        -----------
        in_features: int
            the number of input features

        out_features: int
            the number of output features

        hidden_features: [int, int]
            the number of hidden features
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features[0])
        self.fc2 = nn.Linear(hidden_features[0], hidden_features[1])
        self.fc3 = nn.Linear(hidden_features[1], out_features)

        self.bn1 = nn.BatchNorm1d(hidden_features[0])
        self.bn2 = nn.BatchNorm1d(hidden_features[1])

        self.dp = nn.Dropout(dp)
        if act == 'selu':
            self.act = F.selu
        else:
            self.act = F.relu

    def forward(self, x):
        y = []

        x = self.act(self.dp(self.bn1(self.fc1(x))))
        x = self.act(self.dp(self.bn2(self.fc2(x))))

        x = self.dp(x)
        x = self.fc3(x)
        y.append(x)
        return x, y


class GraphWavenetConvolution(nn.Module):
    """
    Graph Wavenet Convolution Layer 

    Attribute:
    -----
    W: Weight matrix
    """

    def __init__(self, in_features, out_features, dp):
        """
        Parameter:
        -----------
        in_features: int
            the number of input features

        out_features: int
            the number of output features
        """
        super(GraphWavenetConvolution, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1.0 / np.sqrt(out_features)
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, A, X, Z):
        """
        Inputs:
        -----------
        A: torch.FloatTensor
            Normalized adjacency matrix with (n, n) shape

        X: torch.FloatTensor
            Node Feature with (n, d) shape

        Z: torch.FloatTensor
            Node Embedding with (n, d') shape
        """
        if len(X.shape) == 3:
            batch = X.shape[0]
            d = X.shape[1]
            n = X.shape[2]
            X = X.reshape([-1, n])
        else:
            batch = 0

        X0 = X
        X = X0.unsqueeze(2)

        for _A in A:
            X1 = torch.sparse.mm(_A, X0.transpose(0, 1)).transpose(0, 1)
            X = torch.cat([X, X1.unsqueeze(2)], 2)

            X2 = 2*torch.sparse.mm(_A, X1.transpose(0, 1)).transpose(0, 1) - X0
            X = torch.cat([X, X2.unsqueeze(2)], 2)

        Az = F.softmax(F.relu(torch.mm(Z, Z.transpose(0, 1))), 0)
        Xz = torch.mm(Az, X0.transpose(0, 1)).transpose(0, 1)
        X = torch.cat([X, Xz.unsqueeze(2)], 2)

        if batch != 0:
            X = X.sum(2, keepdim=True).reshape([batch, d, n])
            # X = X.reshape([batch, d, n, 4])
            X = X.transpose(1, 2)
            X = X.reshape([batch, n, -1])

        X = torch.matmul(X, self.W)

        return X


class GWNet(nn.Module):
    """
    Graph WaveNet Network
    """

    def __init__(self, in_features, hidden_features, args, aggregate=True):
        """
        Parameter:
        -----------
        in_features: int
            the number of input features

        hidden_features: int
            the number of hidden features 

        out_features: int
            the numer of output features
        """
        super(GWNet, self).__init__()
        self.aggregate = aggregate
        self.gw1 = GraphWavenetConvolution(
            in_features, hidden_features[0], dp=args.dp)
        self.gw2 = GraphWavenetConvolution(
            hidden_features[0], hidden_features[1], dp=args.dp)
        if args.act == 'selu':
            self.act = F.selu
        else:
            self.act = F.relu

        self.dp = nn.Dropout(args.dp)

    def forward(self, A, X, Z):
        """
        Inputs:
        -----------
        A: torch.FloatTensor
            Normalized adjacency matrix with (n, n) shape

        X: torch.FloatTensor
            Node Feature with (n, d) shape

        Z: torch.FloatTensor
            Node Embedding with (n, d') shape
        """
        X = self.act(
            self.gw1(A, X, Z).transpose(1, 2)
        )
        X = self.act(
            self.gw2(A, X, Z).transpose(1, 2)
        )

        if self.aggregate:
            X = torch.mean(X, 2)
        return X


class SINet(Proto):
    """
    Spatial Intervention Network
    """

    def __init__(self, in_features, treat_features, out_features, A, W, y_scaler, args):
        """
        Parameter:
        -----------
        in_features: int
            the number of input features

        treat_features: int
            the number of treatment features 

        out_features: int
            the numer of output features

        A: torch.FloatTensor
            Normalized adjacency matrix with (n_theater, n_theater) shape

        W: list of torch.FloatTensor
            Normalized adjacency matrix with (n_seat, n_seat) shape
        """
        super().__init__(in_features, treat_features, out_features, A, y_scaler, args)
        self.A = A
        self.W = W

        self.repnet = GWNet(in_features, args.rep_hidden, args)
        self.repnet_xz = GWNet(
            args.rep_hidden[0], args.rep_hidden, args, True)

        self.outnet = MLP(
            args.rep_hidden[-1] + 1, out_features, args.out_hidden, args.dp, args.act)

        self.node_rep = nn.ParameterDict()
        for key in W.keys():
            n = W[key][0].size()[0]
            Z = torch.nn.parameter.Parameter(
                torch.torch.rand([n, args.node_hidden]))
            self.node_rep[key] = Z

        self.embeddings = nn.Embedding(18, args.rep_hidden[1])
        self.theater_rep = nn.ParameterDict()
        n = A[0].size()[0]
        Z = torch.nn.parameter.Parameter(
            torch.torch.rand([n, args.node_hidden]))
        self.theater_rep['theater'] = Z

        self.params =\
            list(self.repnet.parameters()) + \
            list(self.outnet.parameters()) +\
            list(self.theater_rep.parameters()) +\
            list(self.node_rep.parameters()) +\
            list(self.embeddings.parameters())
        self.optimizer = optim.Adam(
            params=self.params, lr=args.lr, weight_decay=args.wd)
        self.scheduler = StepLR(
            self.optimizer, step_size=args.step, gamma=args.steprate)

    def get_mmd(self, x_rep, z):
        znp = z.cpu().detach().numpy()
        id = np.zeros(znp.shape[0])

        # get the number of treatments
        values, counts = np.unique(znp, axis=0, return_counts=True)
        # set most as control
        _id = np.zeros(values.shape[0])
        _id[counts.argmax()] = 1

        # set the elements of the selected treatment's is to 1
        for i in range(znp.shape[0]):
            value_id = np.where((znp[i] == values).all(axis=1))[0]
            id[i] = _id[value_id]

        if len(values) == 1:
            return x_rep.sum()*0

        a0 = x_rep[id == 0, :].contiguous()
        a1 = x_rep[id == 1, :].contiguous()
        mmd = self.mmd_rbf(a0, a1, self.sigma)
        return mmd

    def data2xrep(self, data):
        oh1f = data['oh1f'].to(device=self.args.device).unsqueeze(1)
        oh2f = data['oh2f'].to(device=self.args.device).unsqueeze(1)
        oh3f = data['oh3f'].to(device=self.args.device).unsqueeze(1)
        oh4f = data['oh4f'].to(device=self.args.device).unsqueeze(1)
        ph = data['ph'].to(device=self.args.device).unsqueeze(1)
        tf = data['tf'].to(device=self.args.device).unsqueeze(1)

        oh1f_rep = self.repnet.forward(self.W['oh1f'],
                                       oh1f,
                                       self.node_rep['oh1f'])
        oh2f_rep = self.repnet.forward(self.W['oh2f'],
                                       oh2f,
                                       self.node_rep['oh2f'])
        oh3f_rep = self.repnet.forward(self.W['oh3f'],
                                       oh3f,
                                       self.node_rep['oh3f'])
        oh4f_rep = self.repnet.forward(self.W['oh4f'],
                                       oh4f,
                                       self.node_rep['oh4f'])
        ph_rep = self.repnet.forward(self.W['ph'],
                                     ph,
                                     self.node_rep['ph'])
        tf_rep = self.repnet.forward(self.W['tf'],
                                     tf,
                                     self.node_rep['tf'])

        oh1f_rep = oh1f_rep.reshape([len(oh1f_rep), -1])
        oh2f_rep = oh2f_rep.reshape([len(oh2f_rep), -1])
        oh3f_rep = oh3f_rep.reshape([len(oh3f_rep), -1])
        oh4f_rep = oh4f_rep.reshape([len(oh4f_rep), -1])
        ph_rep = ph_rep.reshape([len(ph_rep), -1])
        tf_rep = tf_rep.reshape([len(tf_rep), -1])

        x_rep = torch.cat(
            [tf_rep.unsqueeze(1),
             ph_rep.unsqueeze(1),
             oh4f_rep.unsqueeze(1),
             oh3f_rep.unsqueeze(1),
             oh2f_rep.unsqueeze(1),
             oh1f_rep.unsqueeze(1)
             ], axis=1)

        return x_rep

    def forward(self, data, data_cs):
        _z = data['treatment']
        z = torch.IntTensor(np.arange(9))*2
        z = (z+_z.int()).to(device=self.args.device)
        y = data['outcome'].to(device=self.args.device)
        m = data['mean'].to(device=self.args.device)
        if len(z.shape) == 3:
            z = z.squeeze(0)
            y = y.squeeze(0)
            m = m.squeeze(0)

        # get representation
        x_rep = self.data2xrep(data)
        z_rep = self.embeddings(z)
        if len(x_rep) != len(z_rep):
            x_rep = torch.tile(x_rep, [len(z_rep), 1, 1])
        xz_rep = torch.cat([x_rep, z_rep], axis=1).transpose(1, 2)

        xz_rep = self.repnet_xz.forward(self.A,
                                        xz_rep,
                                        self.theater_rep['theater'])
        xz_rep = xz_rep.reshape([len(xz_rep), -1])
        X = torch.tile(xz_rep.unsqueeze(1), [1, y.shape[1], 1])
        t = torch.tensor(
            np.arange(y.shape[1])/y.shape[1], requires_grad=True).to(torch.float).to(device=self.args.device)
        t = torch.tile(t.unsqueeze(0), [y.shape[0], 1]).unsqueeze(2)
        X = torch.cat([X, t], axis=2)
        X = X.reshape([-1, X.shape[2]])

        # get output
        y_hat, _ = self.outnet(X)
        y_hat = y_hat.reshape(y.shape)
        for i in range(y.shape[0]):
            torch.clamp_(y_hat[i], 0, y[i, -1])

        # get HSIC regularizer
        if self.training:
            hsic = self.HSIC(x_rep, z_rep, self.sigma)
            mmd = self.get_mmd(x_rep, z)
        else:
            hsic = 0.0
            mmd = 0.0

        # get Monotonic Regularizer
        if self.training:
            if not y_hat.grad_fn == None:
                grad_input = torch.autograd.grad(
                    y_hat.sum(), t, create_graph=True, allow_unused=True)[0]
                grad_input_neg = -grad_input
                # grad_input_neg += .2
                grad_input_neg += .1
                grad_input_neg[grad_input_neg < 0.] = 0.
                reg_loss = (grad_input_neg**2).mean()
        else:
            reg_loss = 0.0

        return y, y_hat, hsic, mmd, m, x_rep, z, reg_loss
