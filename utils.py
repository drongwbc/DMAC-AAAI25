import scipy.io as scio
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans


def load_data(name, views):
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0], ))

    X = []
    for i in range(0, views):
        tmp = data['X' + str(i+1)]
        tmp = tmp.astype(np.float32)
        X.append(torch.from_numpy(tmp).to(dtype=torch.float))

    return X, labels


def anchor_selection_loss(Z, anchor, v=1):
    device = Z.device
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(Z.unsqueeze(1).to(device) - anchor, 2), 2) / v)
    q = q.pow((v + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1).to(device)).t()
    q = torch.clamp(q, min=1e-12)
    loss = torch.mean(-torch.sum(q * torch.log(q), dim=1))
    return loss


def mi_loss(Q1, Q2):
    n = Q1.size(0)
    mi = torch.zeros(n)

    joint_prob = Q1.unsqueeze(2) * Q2.unsqueeze(1)
    p_marginal = joint_prob.sum(dim=2, keepdim=True)
    q_marginal = joint_prob.sum(dim=1, keepdim=True)

    joint_prob = torch.clamp(joint_prob, min=1e-10)
    p_marginal = torch.clamp(p_marginal, min=1e-10)
    q_marginal = torch.clamp(q_marginal, min=1e-10)

    mi = (joint_prob * torch.log(joint_prob /
          (p_marginal * q_marginal))).sum(dim=(1, 2))

    return -mi.mean()


def structure_loss(Z, A):
    p_D = torch.pinverse(torch.diag(torch.sum(A, dim=0)))
    t = torch.mm(Z.T, Z) - \
        torch.mm(torch.mm(torch.mm(torch.mm(Z.T, A), p_D), A.T), Z)
    loss = torch.trace(t)
    return loss.mean()


def distance(X, Y, square=True):
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y
    y = y.repeat(n, 1)
    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def construct_anchor_graph(Z, anchor, k):
    distances = distance(Z.t(), anchor.t(), square=True)
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, k]
    top_k = torch.t(top_k.repeat(distances.shape[1], 1)) + 10 ** -10
    sum_top_k = torch.sum(sorted_distances[:, 0:k], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(distances.shape[1], 1))
    weights = torch.div(top_k - distances, k * top_k - sum_top_k + 1e-7)
    weights = weights.relu()
    return weights
