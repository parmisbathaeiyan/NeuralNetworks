"""
This program uses RBF network for 3D regression.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import rbf_net
import rbf_layer as rbf
import torch.nn as nn


fig = None
ax1 = None
ax2 = None


def get_constants():
    max_epoch = 1500
    each_turn = 75  # every each_turn epochs result shown will be updated
    learning_rate = 0.05
    neuron_hidden_num = 40
    return max_epoch, each_turn, learning_rate, neuron_hidden_num


# either func1 or func2 can be chosen as the base data
def generate_data():
    X = Y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(X, Y)
    Z = func1(X, Y)
    return X, Y, Z


def func1(X, Y):
    Z = np.sin(X) * np.cos(Y)
    return Z


def func2(X, Y):
    Z = np.sin(X * Y)
    return Z


def present_initial(X, Y, Z):
    global fig
    global ax1
    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='plasma')


def present(X, Y, predicted):
    global fig
    global ax2
    if ax2: ax2.remove()
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, predicted, cmap='plasma')
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.pause(0.01)


# this function will make the two plots' view angle change in the same way
def on_move(event):
    if event.inaxes == ax1:
        ax2.view_init(elev=ax1.elev, azim=ax1.azim)
    elif event.inaxes == ax2:
        ax1.view_init(elev=ax2.elev, azim=ax2.azim)
    else:
        return
    fig.canvas.draw_idle()


def main():
    max_epoch, each_turn, learning_rate, neuron_hidden_num = get_constants()
    X, Y, Z = generate_data()
    XY = np.column_stack([X.flat, Y.flat])
    present_initial(X, Y, Z)

    basis_func = rbf.gaussian
    net = rbf_net.RBFNet([2, 1], neuron_hidden_num, basis_func)
    for i in range(max_epoch//each_turn):
        net.fit(torch.from_numpy(XY).float(), torch.from_numpy(Z.flatten()).float(), each_turn, learning_rate, nn.MSELoss())
        net.eval()
        with torch.no_grad():
            prediction = net(torch.from_numpy(XY).float()).data.numpy()
        present(X, Y, prediction.reshape(Z.shape))
        print(f'\rEpoch {(i+1) * each_turn}/{max_epoch}', end='')
    plt.show()


if __name__ == '__main__':
    main()
