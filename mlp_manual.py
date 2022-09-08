"""
In this module MLP network with 1 hidden layer is implemented manually.
"""

import numpy as np


class MLP:
    def __init__(self, input_num, output_num, neuron_num, learning_rate, activation):
        self.learning_rate = learning_rate
        self.activation = activation
        self.w1 = 2 * np.random.rand(input_num, neuron_num) - 1
        self.bw1 = 2 * np.random.rand(neuron_num) - 1
        self.w2 = 2 * np.random.rand(neuron_num, output_num) - 1
        self.bw2 = 2 * np.random.rand(output_num) - 1

    def train(self, indexes, labels_onehot):
        loss = 0
        predicted = []
        for index, label in zip(indexes, labels_onehot):
            o1 = self.activation(np.dot(index, self.w1) + self.bw1)
            o2 = self.activation(np.dot(o1, self.w2) + self.bw2)
            predicted.append(o2)

            delta2 = 2 * (np.subtract(o2, label)) * self.activation(o2, der=True)
            delta1 = [(2 * a * b) for a, b in zip(np.dot(self.w2, delta2), self.activation(o1, der=True))]

            Delta = self.learning_rate * np.kron(o1, delta2).reshape(self.w2.shape)
            self.w2 = np.subtract(self.w2, Delta)
            Delta = self.learning_rate * delta2
            self.bw2 = np.subtract(self.bw2, Delta)

            Delta = self.learning_rate * np.kron(index, delta1).reshape(self.w1.shape)
            self.w1 = np.subtract(self.w1, Delta)
            Delta = self.learning_rate * np.array(delta1)
            self.bw1 = np.subtract(self.bw1, Delta)

            loss += np.sum(abs(np.subtract(o2, label))) / 2
        loss /= indexes.shape[0]
        return loss

    def predict(self, indexes, labels):
        prediction = []
        loss = 0
        for index, label in zip(indexes, labels):
            o1 = self.activation(np.dot(index, self.w1) + self.bw1)
            o2 = self.activation(np.dot(o1, self.w2) + self.bw2)
            prediction.append(o2)
            loss += np.sum(abs(np.subtract(o2, label))) / 2
        loss /= indexes.shape[0]
        return loss, prediction


def sigmoid(val, der=False):
    f = 1 / (1+np.exp(-val))
    if der:
        f = f * (1-f)
    return f


def tanh(x, der=False):
    f = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    if der:
        f = 1 - f**2
    return f
