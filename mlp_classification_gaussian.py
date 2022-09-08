"""
This program classifies 5 gaussian clusters using mlp_manual module.
--> implemented manually with one hidden layer.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import mlp_manual


def generate_data():
    indexes, labels = datasets.make_blobs(n_samples=250, centers=[(-0.8, -1), (-1.5, 0.25), (0, 1), (1.5, 0.25), (0.8, -1)],
                                          cluster_std=0.25, random_state=20)
    labels_onehot = [[0, 0, 0, 0, 0] for _ in range(250)]
    for i in range(250):
        label = labels[i]
        labels_onehot[i][label] = 1
    return indexes, labels, labels_onehot


def present_initial(indexes, labels):
    plt.figure()
    plt.title("Decision Boundary")
    plt.scatter(indexes[:, 0], indexes[:, 1], c=labels)
    plt.waitforbuttonpress()


# because the output of the network isn't 0 & 1, we choose the highest value as chosen class
def classify(arr):
    predicted = []
    for element in arr:
        chosen = np.argmax(element)
        predicted.append(chosen)
    return predicted


def present(indexes, predicted, epoch, loss):
    plt.cla()
    plt.scatter(indexes[:, 0], indexes[:, 1], c=predicted[:], alpha=0.6)
    plt.text(-2, 1.2, 'Epoch = %d\nLoss = %.4f' % (epoch, loss), fontdict={'size': 12, 'color': 'dimgray'})
    plt.pause(0.5)


def main():
    neuron_num = 10  # num of neurons in hidden layer
    output_num = 5  # num of classes
    input_num = 2
    max_epochs = 100
    max_loss = 0.6
    learning_rate = 0.01

    indexes, labels, labels_onehot = generate_data()
    present_initial(indexes, labels)

    net = mlp_manual.MLP(input_num, output_num, neuron_num, learning_rate, mlp_manual.sigmoid)
    for epoch in range(max_epochs):
        net.train(indexes, labels_onehot)
        loss, predicted = net.predict(indexes, labels_onehot)
        pred_int = classify(predicted)
        present(indexes, pred_int, epoch, loss)

        if loss <= max_loss:
            break
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
