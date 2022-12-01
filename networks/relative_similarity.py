import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_norm

from hashing.centroids_generator import generate_centroids

class RelativeSimilarity(nn.Module):
    def __init__(self, nbit, nclass, batchsize, init_method='M', device='cuda'):
        super(RelativeSimilarity, self).__init__()
        self.nbit = nbit
        self.nclass = nclass

        self.centroids = nn.Parameter(generate_centroids(nclass, nbit, init_method='N', device=device))
        self.centroids.requires_grad_(False)

        self.relative_pos = RelativePosition(nbit, batchsize, device=device)

        self.update_centroids(generate_centroids(nclass, nbit, init_method=init_method, device=device))
        
    def forward(self, z):
        z_star = self.relative_pos(z)
        return z_star @ self.c_star.T

    def update_centroids(self, centroids):
        self.centroids.data.copy_(centroids)
        self.c_star = self.centroids / torch.linalg.norm(self.centroids, dim=-1, keepdim=True) # normalize and fixed centroids
        self.c_star.requires_grad_(False)

    def extra_repr(self) -> str:
        return 'nbit={}, n_class={}'.format(self.nbit, self.nclass)


class RelativePosition(nn.Module):
    """
    Relative Position with numerical stability and optimized performance 
    """
    def __init__(self, k, b, ignore_constant=True, device='cuda'):
        """
        :param k: number of features, or nbit
        :param b: batchsize
        """
        super(RelativePosition, self).__init__()
        self.k = k
        self.n = b
        self.gamma = 1 if ignore_constant else torch.tensor(k * b).to(device).float().sqrt()
    
    def forward(self, z):
        a = z - z.mean()
        return self.gamma * a / matrix_norm(a)

        # mu = z.mean()
        # a = z - mu
        # zeta = matrix_norm(a)
        # return self.gamma * a / zeta
