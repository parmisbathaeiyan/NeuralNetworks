"""
This program uses LVQ neural network to make a number of prototypes for
each class of gaussian distributed data. Throughout training prototype find their place and according
to those, network will predict test dataset.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import lvq


def generate_data():
    indexes, labels = datasets.make_blobs(n_samples=200, centers=[(5, 5), (5, -5), (-5, -5), (-5, 5)],
                                          cluster_std=1.5, shuffle=True, random_state=21)
    labels = [(0, 1)[i % 2] for i in labels]
    return indexes, labels


def present_initial(indexes, labels,  initial_weights):
    plt.figure()
    plt.scatter(indexes[:, 0], indexes[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    proto_marks = plt.plot(initial_weights[:, 0], initial_weights[:, 1], alpha=0.6, marker='D', linestyle='None', color='black')
    plt.pause(1)
    return proto_marks


def present(prototypes, marks, done=False):
    marks.pop(0).remove()
    marks = plt.plot(prototypes[:, 0], prototypes[:, 1], alpha=0.6, marker='D', linestyle='None', color='black')
    if not done:
        plt.pause(0.2)
        return marks
    plt.text(-3, 0, 'Click anywhere to continue.')
    plt.pause(0.2)
    plt.waitforbuttonpress()
    plt.close()


def main():
    proto_num = 10  # prototype num is for each cluster
    max_epoch = 25
    learning_rate = 0.1

    indexes, labels = generate_data()
    labels = np.array(labels)

    net = lvq.LVQ(learning_rate, proto_num)
    prototypes, proto_labels = net.initiate_weights(indexes, labels)

    marks = present_initial(indexes, labels, prototypes)

    for epoch in range(max_epoch):
        prototypes, proto_labels = net.train(indexes, labels)
        marks = present(prototypes, marks)
        print('\rEpoch %d/%d' % (epoch+1, max_epoch), end='')
    present(prototypes, marks, done=True)

    xx, yy = np.meshgrid(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1))
    points = np.c_[xx.ravel(), yy.ravel()]
    labels_pred = net.predict(points)
    plt.figure()
    plt.contourf(xx, yy, np.reshape(labels_pred, xx.shape), cmap='seismic', alpha=0.8, levels=101)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()

