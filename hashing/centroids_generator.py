import logging
import torch
from scipy.linalg import hadamard
import random
import numpy as np

def generate_centroids(nclass: int, nbit: int, init_method: str, device=torch.device('cuda:0')) -> torch.Tensor:
    if init_method == 'N':  # normal distribution
        centroids = torch.randn(nclass, nbit, device=device)
    elif init_method == 'U':  # uniform distribution
        centroids = torch.rand(nclass, nbit, device=device) - 0.5
    elif init_method == 'B':  # bernoulli distribution
        prob = torch.ones(nclass, nbit, device=device) * 0.5
        centroids = torch.bernoulli(prob) * 2. - 1.
    elif init_method == 'M':  # Maximum Hamming Distance
        centroids = get_maxhd(nclass, nbit)
    elif init_method == 'H':  # hadamard matrix
        centroids = get_hadamard(nclass, nbit)
    else:
        raise NotImplementedError(f'Centroid initialization method {init_method} is not implemented')
    return centroids.to(device).sign() 

def get_hd(a, b):
    return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)
    
def get_maxhd(nclass, nbit, maxtries=10000, initdist=0.61, mindist=0.2, reducedist=0.01):
    centroid = torch.zeros(nclass, nbit)
    i = 0
    count = 0
    currdist = initdist
    curri = -1
    while i < nclass:
        if curri != i:
            curri = i
            logging.info(f'Doing for class {i}')

        c = torch.randn(nbit).sign().float()
        nobreak = True

        # to compare distance with previous classes
        for j in range(i):
            if get_hd(c, centroid[j]) < currdist:
                i -= 1
                nobreak = False
                break

        if nobreak:
            centroid[i] = c
        else:
            count += 1

        if count >= maxtries:
            count = 0
            currdist -= reducedist
            logging.info(f'Max tried for {i}, reducing distance constraint {currdist}')
            if currdist < mindist:
                raise ValueError('cannot find')

        i += 1

    # shuffle the centroid
    centroid = centroid[torch.randperm(nclass)]
    return centroid


def get_hadamard(nclass, nbit, fast=True):
    H_K = hadamard(nbit)
    H_2K = np.concatenate((H_K, -H_K), 0)
    hash_targets = torch.from_numpy(H_2K[:nclass]).float()

    if H_2K.shape[0] < nclass:
        hash_targets.resize_(nclass, nbit)
        for k in range(20):
            for index in range(H_2K.shape[0], nclass):
                ones = torch.ones(nbit)
                # Bernouli distribution
                sa = random.sample(list(range(nbit)), nbit // 2)
                ones[sa] = -1
                hash_targets[index] = ones

            if fast:
                return hash_targets

            # to find average/min  pairwise distance
            c = []
            # print()
            # print(n_class)
            TF = (hash_targets.view(1, -1, nbit) != hash_targets.view(-1, 1, nbit)).sum(dim=2).float()
            TF_mask = torch.triu(torch.ones_like(TF), 1).bool()
            c = TF[TF_mask]

            # choose min(c) in the range of K/4 to K/3
            # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
            # but it is hard when bit is  small
            if c.min() > nbit / 4 and c.mean() >= nbit / 2:
                print(c.min(), c.mean())
                break

    return hash_targets
