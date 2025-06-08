import torch
import utils
import torch.nn.functional as F
import torch.nn as nn
import warnings
import math
from model import TestModel
from torch.optim import RMSprop
from evaluation import eva
import random
import numpy as np
from sklearn.cluster import KMeans


def setup_seed(seed):
    if seed == 0:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    setup_seed(args.seed)

    X, Y = utils.load_data(args.name, args.views)

    args.samples = X[0].shape[0]
    args.n_clusters = len(np.unique(Y))
    args.num_features = []
    anchor_num = math.floor(math.sqrt(args.samples * args.n_clusters))
    neighbor = 5
    init_nei = neighbor
    upper = math.floor(anchor_num / args.n_clusters)

    for i in range(0, args.views):
        X[i] = F.normalize(X[i], p=2, dim=1).to(device)

    print('---------Dataset Details---------')
    print('name: ' + args.name)
    print(f'samples: {args.samples:.0f}')
    print(f'cluster: {args.n_clusters:.0f}')
    print(f'views: {args.views:.0f}')

    print(f'dimension: {X[0].shape[1]:.0f} ', end='')
    args.num_features.append(X[0].shape[1])
    for i in range(1, args.views):
        print(X[i].shape[1], end=' ')
        args.num_features.append(X[i].shape[1])
    print()
    print(f'anchor: {anchor_num:.0f} ', end='')
    print('\n---------------------------------')

    curModel = TestModel(num_features=args.num_features,
                         hidden_size=args.hidden_size,
                         emblem_size=args.emblem_size,
                         views=args.views,
                         n_clusters=args.n_clusters).to(device)

    optimizer = RMSprop(curModel.parameters(),
                        lr=args.lr1)

    if args.max_preEpoch > 0:
        print('---------Start PreTrain---------')
        mse = nn.MSELoss()
        for epoch in range(0, args.max_preEpoch):
            curModel.train()
            rec_X = curModel(
                X, args.views, anchor_num, neighbor, preTrain=True)

            loss = 0
            for i in range(0, args.views):
                loss += mse(X[i], rec_X[i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if epoch % 50 == 0:
                    print(f'epoch {epoch}: loss: {loss.item():.4f}')

    optimizer = RMSprop(curModel.parameters(),
                        lr=args.lr2)

    print('---------Start Train---------')
    mse = nn.MSELoss()
    retScore = [0, 0, 0, 0]
    for epoch in range(0, args.max_epoch):
        curModel.train()
        Z, anchor, fuse_Z, A, Q = curModel(
            X, args.views, anchor_num, neighbor)

        l1 = 0
        l2 = 0
        l3 = 0
        for i in range(0, args.views):
            l1 += utils.anchor_selection_loss(Z[i], anchor)
            l2 += utils.structure_loss(fuse_Z, A[i])
            for j in range(i + 1, args.views):
                l3 += utils.mi_loss(Q[i], Q[j])

        loss = l1 + args.alpha*l2 + args.beta*l3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            curModel.eval()
            Z, anchor, fuse_Z, A, Q = curModel(
                X, args.views, anchor_num, neighbor)

            label = KMeans(n_clusters=args.n_clusters, n_init=10,
                           init='k-means++').fit(fuse_Z.cpu().numpy()).labels_

            acc, nmi, ari, f1 = eva(Y, label)
            retScore[0] = max(retScore[0], acc)
            retScore[1] = max(retScore[1], nmi)
            retScore[2] = max(retScore[2], ari)
            retScore[3] = max(retScore[3], f1)

            if epoch > 0 and epoch % 20 == 0:
                neighbor = min(neighbor + init_nei, upper)

            print(
                f'epoch {epoch}: acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1: {f1:.4f}, loss: {loss.item():.4f}')

    print("---------------------")
    print(
        f'best  acc: {retScore[0]:.4f}, nmi: {retScore[1]:.4f}, ari: {retScore[2]:.4f}, f1: {retScore[3]:.4f}'
    )

    return retScore[0], retScore[1], retScore[2], retScore[3]
