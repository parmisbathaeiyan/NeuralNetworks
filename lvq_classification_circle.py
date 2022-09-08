"""
This program creates a dataset with two classes and with LVQ network
finds a number of prototypes for each class.
"""

import math
import numpy as np
from matplotlib import pyplot as plt
import lvq


def generate_data():
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 40))
    indexes = np.c_[xx.ravel(), yy.ravel()]

    labels = []
    for index in indexes:
        x = index[0]
        y = index[1]
        if ((x**2) <= 16) and (y > -math.sqrt(16 - (x ** 2))) and (y < math.sqrt(16 - (x ** 2))):
            labels.append(1)
        else:
            labels.append(0)

    np.random.seed(10)
    np.random.shuffle(indexes)
    np.random.seed(10)
    np.random.shuffle(labels)
    return indexes, labels


def present_initial(indexes, labels, initial_weights):
    plt.figure()
    plt.scatter(indexes[:, 0], indexes[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    proto_marks = plt.plot(initial_weights[:, 0], initial_weights[:, 1], alpha=0.6, marker='D', linestyle='None', color='black')
    plt.pause(1)
    return proto_marks


def present(prototypes, marks):
    marks.pop(0).remove()
    marks = plt.plot(prototypes[:, 0], prototypes[:, 1], alpha=0.6, marker='D', linestyle='None', color='black')
    plt.pause(0.1)
    return marks


def main():
    proto_num = 10
    learning_rate = 0.1
    max_epoch = 40

    indexes, labels = generate_data()
    labels = np.array(labels)

    net = lvq.LVQ(learning_rate, proto_num)
    prototypes, proto_labels = net.initiate_weights(indexes, labels)

    marks = present_initial(indexes, labels, prototypes)

    for epoch in range(max_epoch):
        prototypes, proto_labels = net.train(indexes, labels)
        marks = present(prototypes, marks)
        print('\rEpoch %d/%d' % (epoch+1, max_epoch), end='')
    plt.waitforbuttonpress()
    plt.close()


if __name__ == '__main__':
    main()

