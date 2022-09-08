"""
RBF is a neural network with only one layer. It's activation
function is gaussian and therefore, center and std of neurons
are also network parameters that need to be trained.
In this file RBF is implemented with help of pytorch.
"""

import rbf_layer as rbf
import torch
import torch.nn as nn


class RBFNet(nn.Module):
    def __init__(self, layer_widths, layer_centres, basis_func, c=None):
        super(RBFNet, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.rbf_layers = rbf.RBF(layer_widths[0], layer_centres, basis_func, c)
        self.linear_layers = nn.Linear(layer_centres, layer_widths[1])

    def forward(self, x):
        x = self.rbf_layers(x)
        x = self.linear_layers(x)
        return x

    def fit(self, x, y, epochs, lr, loss_func):
        self.train()
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(epochs):
            optimiser.zero_grad()
            y_hat = self.forward(x)
            loss = loss_func(y_hat.squeeze(1), y)
            loss.backward()
            optimiser.step()

