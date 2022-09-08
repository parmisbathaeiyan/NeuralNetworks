"""
LVQ is a neural network which works by making a number of prototypes for
each class and then, predicting data according to the prototypes found.
Label of whichever prototype is closer to a given data, will be chosen.
Prototype vectors are found according to train data. By visiting train data,
if a data has the same label as the prototype, the prototype will get closer
to it and if not, will get farther from it.
Here this network is implemented manually.
"""

import math
import numpy as np


class LVQ:
    def __init__(self, learning_rate, proto_num):
        self.learning_rate = learning_rate
        self.proto_num = proto_num

    def initiate_weights(self, indexes, labels):
        unique_labels = list(set(labels))
        self.proto_labels = np.array(
            [[unique_labels[j] for _ in range(self.proto_num)] for j in range(len(unique_labels))]
            ).flatten()
        initial_weights = []
        for i in unique_labels:
            initial_weights.append(indexes[labels == i][0:self.proto_num])
        self.prototypes = np.array(initial_weights).flatten().reshape(((len(unique_labels)*self.proto_num), 2))
        return self.prototypes, self.proto_labels

    def distance(self, vect1, vect2):
        sum = 0
        for (i1, i2) in zip(vect1, vect2):
            sum += (i1 - i2) ** 2
        dist = math.sqrt(sum)
        return dist

    def train(self, indexes, labels):
        won = 0
        for (index, label) in zip(indexes, labels):
            dist = float('inf')
            for i, prototype in enumerate(self.prototypes):
                current = self.distance(prototype, index)
                if current < dist:
                    dist = current
                    won = i

            change = self.learning_rate * np.subtract(index, self.prototypes[won])
            if self.proto_labels[won] == label:
                self.prototypes[won] += change
            else:
                self.prototypes[won] -= change
        return self.prototypes, self.proto_labels

    def predict(self, indexes):
        labels = []
        for index in indexes:
            dist = float('inf')
            for i, prototype in enumerate(self.prototypes):
                current = self.distance(prototype, index)
                if current < dist:
                    dist = current
                    won = i
            if self.proto_labels[won] == 1:
                labels.append(1)
            else:
                labels.append(0)
        return labels

