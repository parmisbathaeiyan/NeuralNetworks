"""
This program uses RBF network for 2D regression.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import rbf_net
import rbf_layer as rbf
import torch.nn as nn


def generate_data(num):
    X = torch.linspace(-4, 4, num).data.numpy()
    Y = func(X) + 0.2 * np.random.normal(0, 0.2, len(X))

    X_real = torch.linspace(-4, 4, 1000).data.numpy()
    Y_real = func(X_real)
    return X, Y, X_real, Y_real


def func(x):
    y = (1 - 4 * x - (x ** 3) / 17) * np.sin(x ** 2)
    return y


def present_initial(X, Y, X_real, Y_real):
    plt.cla()
    plt.grid(True)
    plt.plot(X_real, Y_real, label='Actual function', linewidth=2, c='darkslateblue', alpha=0.8)
    plt.scatter(X, Y, label='Data', s=15, c='darkslateblue')


def represent(X, predicted, line, text, epoch):
    if line: line.pop(0).remove()
    if text: text.remove()
    line = plt.plot(X, predicted, label='Predicted',  linewidth=2, c='crimson', alpha=0.8, marker='.', markersize=9)
    text = plt.text(-0.75, -12, 'Epoch: %d' % epoch, size=15, color='black', alpha=0.8,  name='helvetica')
    plt.legend(loc='lower center')
    plt.pause(0.01)
    return line, text


def main():
    data_num = 150
    max_epoch = 1000
    learning_rate = 0.2
    neuron_hidden_num = 25

    X, Y, X_real, Y_real = generate_data(num=data_num)

    plt.figure(figsize=(8, 6))
    present_initial(X, Y, X_real, Y_real)
    X_expanded = np.expand_dims(X, axis=1)

    basis_func = rbf.gaussian
    net = rbf_net.RBFNet([1, 1], neuron_hidden_num, basis_func)

    each_turn = 10
    line = None
    text = None
    for i in range(max_epoch//each_turn):
        net.fit(torch.from_numpy(X_expanded).float(), torch.from_numpy(Y).float(), each_turn, learning_rate, nn.MSELoss())
        net.eval()
        with torch.no_grad():
            prediction = net(torch.from_numpy(X_expanded).float()).data.numpy()
        line, text = represent(X, prediction, line, text, ((i+1) * each_turn))
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
