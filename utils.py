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

def random_anchor(Z, anchor_num):
    random_indices = torch.randperm(Z.size(0))
    anchor = Z[random_indices[:anchor_num]]
    return anchor

def tmp_loss(q):
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


def anchor_graph(Z, A, anchor):
    mse = nn.MSELoss()
    return mse(Z, torch.mm(A, anchor))


def fusion_loss(Z, A):
    p_D = torch.pinverse(torch.diag(torch.sum(A, dim=0)))
    t = torch.mm(Z.T, Z) - torch.mm(torch.mm(torch.mm(torch.mm(Z.T, A), p_D), A.T), Z)
    loss = torch.trace(t)
    return loss.mean()


def anchor_graph_normalize(A):
    D = torch.diag(torch.sum(A, dim=0))
    p_D = torch.pinverse(D)
    return torch.mm(torch.mm(A, p_D), A.T)


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
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


def compute_anchor_weights(X, anchors, k=5):
    """
    计算每个样本相对于锚点的重构权重。

    参数：
    X -- 样本数据，形状为 (n_samples, n_features)
    anchors -- 锚点数据，形状为 (m_anchors, n_features)
    k -- 使用的最近邻锚点数量

    返回：
    weights -- 重构权重，形状为 (n_samples, m_anchors)
    """
    n_samples, n_features = X.shape
    m_anchors = anchors.shape[0]

    # 计算样本和锚点之间的欧氏距离
    dist = torch.cdist(torch.tensor(X), torch.tensor(anchors))

    # 找到每个样本的最近邻锚点索引
    knn_indices = dist.topk(k, largest=False).indices

    # 初始化重构权重矩阵
    weights = torch.zeros((n_samples, m_anchors)).to(X.device)

    # 计算重构权重
    for i in range(n_samples):
        knn_dists = dist[i, knn_indices[i]]
        sum_knn_dists = knn_dists.sum()
        weights[i, knn_indices[i]] = (
            sum_knn_dists - knn_dists) / ((k - 1) * sum_knn_dists)

    return weights


def spectral_clustering(A, k, NCut=False, KMrep=1):
    D = np.diag(A.sum(0))
    lap = D - A
    if NCut:
        d_inv = np.linalg.pinv(D)
        sqrt_d_inv = np.sqrt(d_inv)
        lap = np.matmul(np.matmul(sqrt_d_inv, lap), sqrt_d_inv)
    x, V = np.linalg.eigh(lap)
    V = np.real(V)
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x: x[0])
    H = np.vstack([V[:, i] for (v, i) in x[:k]]).T
    label = KMeans(n_clusters=k, n_init=KMrep, init='k-means++').fit(H).labels_
    return label


def build_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return: Graph 
    """
    size = X.shape[1]
    num_neighbors = min(num_neighbors, size - 1)
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).to(X.device)
        weights += torch.eye(size).to(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(X.device)
    weights = weights.to(X.device)
    return weights, raw_weights


def graph_normalize(A):
    """
        Normalize the adj matrix
    """
    degree = torch.sum(A, dim=1).pow(-0.5)
    return (A * degree).t() * degree
