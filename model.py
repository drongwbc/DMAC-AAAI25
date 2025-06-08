import torch
import utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


class TestModel(nn.Module):
    def __init__(self, num_features, hidden_size, emblem_size, views, n_clusters):
        super(TestModel, self).__init__()
        # Encoder
        self.encoder_l1 = []
        self.encoder_l2 = []

        for i in range(0, views):
            self.encoder_l1.append(
                nn.Linear(num_features[i], hidden_size, bias=True))
            self.encoder_l2.append(
                nn.Linear(hidden_size, emblem_size, bias=True))

        self.encoder_l1 = nn.ModuleList(self.encoder_l1)
        self.encoder_l2 = nn.ModuleList(self.encoder_l2)

        # Decoder
        self.decoder_l1 = []
        self.decoder_l2 = []

        for i in range(0, views):
            self.decoder_l1.append(
                nn.Linear(emblem_size, hidden_size, bias=True))
            self.decoder_l2.append(
                nn.Linear(hidden_size, num_features[i], bias=True))

        self.decoder_l1 = nn.ModuleList(self.decoder_l1)
        self.decoder_l2 = nn.ModuleList(self.decoder_l2)

        # Noise Generator
        self.mu = nn.Linear(emblem_size, emblem_size, bias=True)
        self.zita = nn.Linear(emblem_size, emblem_size, bias=True)
        self.prelu_weight = nn.Parameter(torch.Tensor(emblem_size).fill_(0.25))

        # Anchor Graph Convolution
        self.agcn_l1 = []
        self.agcn_l2 = []

        for i in range(0, views):
            self.agcn_l1.append(
                nn.Linear(emblem_size, int(emblem_size / 4), bias=False))
            self.agcn_l2.append(
                nn.Linear(int(emblem_size / 4), n_clusters, bias=False))

        self.agcn_l1 = nn.ModuleList(self.agcn_l1)
        self.agcn_l2 = nn.ModuleList(self.agcn_l2)

    def forward(self, X, views, anchor_num, neighbor, preTrain=False):
        # Encoder
        Z = []
        for i in range(0, views):
            tmp_z = F.leaky_relu(self.encoder_l1[i](X[i]))
            tmp_z = F.leaky_relu(self.encoder_l2[i](tmp_z))
            tmp_z = F.normalize(tmp_z, p=2, dim=1)
            Z.append(tmp_z)

        # Fusion
        fuse_Z = sum(Z) / views
        fuse_Z = F.normalize(fuse_Z, p=2, dim=1)

        if preTrain:
            rec_X = []
            for i in range(0, views):
                tmp_X = F.leaky_relu(self.decoder_l1[i](fuse_Z))
                tmp_X = F.leaky_relu(self.decoder_l2[i](tmp_X))
                tmp_X = F.normalize(tmp_X, p=2, dim=1)
                rec_X.append(tmp_X)
            return rec_X


        anchor = KMeans(n_clusters=anchor_num, n_init=1,
                        init='k-means++').fit(fuse_Z.detach().cpu().numpy()).cluster_centers_
        anchor = torch.tensor(anchor, requires_grad=True).to(fuse_Z.device)

        mvn = torch.distributions.MultivariateNormal(
            torch.zeros(anchor.size(1)), torch.eye(anchor.size(1)))
        noise = mvn.sample((anchor.size(0),)).to(anchor.device)
        mu = F.prelu(self.mu(anchor), self.prelu_weight)
        zita = F.relu(self.zita(anchor))
        anchor = anchor + (mu + torch.mul(zita, noise))
        anchor = F.normalize(anchor, p=2, dim=1)

        # anchor_graph_learning
        A = []
        for i in range(0, views):
            tmp_A = utils.construct_anchor_graph(Z[i], anchor, neighbor)
            A.append(tmp_A)

        # anchor graph convolution
        Q = []
        for i in range(0, views):
            tmp_Q = F.leaky_relu(self.agcn_l1[i](self.anchor_convolution(A[i], anchor)))
            tmp_Q = F.softmax(self.agcn_l2[i](self.anchor_convolution(A[i], tmp_Q)))
            Q.append(tmp_Q)

        return Z, anchor, fuse_Z, A, Q
    
    def anchor_convolution(self, A, F):
        p_D = torch.pinverse(torch.diag(torch.sum(A, dim=0)))
        t1 = torch.mm(A, F)
        t2 = torch.mm(A.T, t1)
        t3 = torch.mm(p_D, t2)
        return t3
