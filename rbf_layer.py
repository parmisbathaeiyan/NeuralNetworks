"""
In this file RBF layer is made to be then used in RBF network.
This code is downloaded from github repository below:
https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
"""

import torch
import torch.nn as nn
import numpy as np


class RBF(nn.Module):
    def __init__(self, in_features, out_features, basis_func, c=None, std=None):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features)) #log(sigma) = 0 --> sigma = exp(0) = 1
        self.basis_func = basis_func
        self.reset_parameters(c, std)

    def reset_parameters(self, c, std):
        if c is None: nn.init.normal_(self.centres, 0, 1)
        else: self.centres = nn.Parameter(torch.flatten(torch.tensor(c).float(), 0, 1))
        if std is None: nn.init.constant_(self.log_sigmas, 0)
        else: self.log_sigmas = nn.Parameter(torch.tensor(np.log(std)).float())

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)  # repeat input data output_num times
        c = self.centres.unsqueeze(0).expand(size)  # repeat centers data_num times
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)  # sum(-1) is sum across the last array-dimension
        return self.basis_func(distances)


def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi
