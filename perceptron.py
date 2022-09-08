"""
Perceptron is neural network only consisting of an input and an output layer. It's activation function is normally
threshold so outputs are either 0 or 1.
Incremental wight correction is implemented meaning for each data weight is updated.
This module is used in perceptron_classification python program.
"""

import numpy as np


class Net:
    def __init__(self, input_num, output_num, learning_rate):
        self.learning_rate = learning_rate
        self.w = np.random.rand(input_num, output_num) - 0.5

    def train(self, indexes, labels):
        misclassified = 0
        labels_predicted = self.predict(indexes)
        for index, label, label_pred in zip(indexes, labels, labels_predicted):
            if np.sum(np.abs(label - label_pred)) != 0:
                self.w[1:] += self.learning_rate * index.reshape(index.shape[0], 1) * (label - label_pred)
                self.w[0] += self.learning_rate * (label - label_pred)
                misclassified += 1
        return misclassified

    def predict(self, indexes):
        prediction = []
        for index in indexes:
            summation = np.dot(index, self.w[1:]) + self.w[0]
            prediction.append((summation > 0) * 1.0)
        return prediction

