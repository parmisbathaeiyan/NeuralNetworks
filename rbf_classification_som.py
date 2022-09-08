"""
This programs considers 4 3D gaussian distributions and creates train data
relative to the distributions. It uses SOM network to find centers of clusters
in train data. According to the centers found, RBF network will be trained.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import datasets
from sklearn_som.som import SOM
import rbf_net
import rbf_layer as rbf
import torch.nn as nn


fig = None
ax1 = None
ax2 = None
ax3 = None


def get_constants():
    max_epoch = 1500
    each_turn = 75
    max_loss = 0.02
    learning_rate = 0.05
    neuron_hidden_num = 12
    centers_cluster = [[-1.5, -2.5], [-2, 2.3], [2.2, -2.2], [2.8, 2.5]]
    return max_epoch, each_turn, max_loss, learning_rate, neuron_hidden_num, centers_cluster


def generate_data(centers):
    X = Y = np.linspace(-5, 5, 30, endpoint=True)
    X, Y = np.meshgrid(X, Y)
    XY = np.column_stack([X.flat, Y.flat])
    XY = np.around(XY, 1)
    Z = func(XY, centers)
    Z = Z.reshape(X.shape)
    indexes, temp = datasets.make_blobs(n_samples=200, centers=centers, cluster_std=0.9, shuffle=False, random_state=10)
    return X, Y, XY, Z, indexes


def func(xy, centers):
    mu = np.array(centers[0])
    covariance = np.diag(np.array([1.1, 1.1]) ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

    mu = np.array(centers[1])
    covariance = np.diag(np.array([1.4, 1.4]) ** 2)
    z += multivariate_normal.pdf(xy, mean=mu, cov=covariance)

    mu = np.array(centers[2])
    covariance = np.diag(np.array([1.8, 1.8]) ** 2)
    z += multivariate_normal.pdf(xy, mean=mu, cov=covariance)

    mu = np.array(centers[3])
    covariance = np.diag(np.array([1.3, 1.3]) ** 2)
    z -= multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    return z


def present_som(indexes, centers):
    global fig
    global ax2
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    plt.text(1.5, -6, 'Click anywhere to continue.')
    ax2.scatter(indexes[:, 0], indexes[:, 1], linewidths=1, marker='.', color='lightcoral', label='x and y of train data')
    ax2.scatter(centers[:, :, 0], centers[:, :, 1], color='black', marker='*', label='centers SOM found')
    ax2.legend(loc='lower right')
    plt.waitforbuttonpress()
    plt.close(fig)


def present_initial(X, Y, Z):
    global fig
    global ax1
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='plasma')


def present(X, Y, predicted):
    global fig
    global ax3
    if ax3: ax3.remove()
    ax3 = fig.add_subplot(122, projection='3d')
    ax3.plot_surface(X, Y, predicted, cmap='plasma')
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.pause(0.01)


def on_move(event):
    if event.inaxes == ax1:
        ax3.view_init(elev=ax1.elev, azim=ax1.azim)
    elif event.inaxes == ax3:
        ax1.view_init(elev=ax3.elev, azim=ax3.azim)
    else:
        return
    fig.canvas.draw_idle()


def main():
    max_epoch, each_turn, max_loss, learning_rate, neuron_hidden_num, centers_cluster = get_constants()

    X, Y, XY, Z, indexes = generate_data(centers_cluster)

    # find centers with SOM network
    som = SOM(m=3, n=4, dim=2, random_state=20)
    som.fit(np.array(indexes))
    centers_som = som.cluster_centers_

    present_som(indexes, centers_som)
    present_initial(X, Y, Z)

    basis_func = rbf.gaussian
    net = rbf_net.RBFNet([2, 1], neuron_hidden_num, basis_func, c=centers_som)
    for i in range(max_epoch//each_turn):
        net.fit(torch.from_numpy(XY).float(), torch.from_numpy(Z.flatten()).float(), each_turn, learning_rate, nn.MSELoss())
        net.eval()
        with torch.no_grad():
            prediction = net(torch.from_numpy(XY).float()).data.numpy()
        present(X, Y, prediction.reshape(Z.shape))
        print(f'\rEpoch {(i+1) * each_turn}/{max_epoch}', end='')

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.show()


if __name__ == '__main__':
    main()
