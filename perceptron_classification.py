"""
This program classifies data with perceptron neural network which is implemented in perceptron module.
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import choose_data
import perceptron


# gets data provision technique from user. data can be 2-class prompt, 4-class prompt, and entered by user
def get_technique():
    print('Choose data provision technique:')
    technique = int(input('1.use prompt 2-class gaussian data \n2.use prompt 4-class gaussian data \n3.select data points \n'))
    while technique not in (1, 2, 3):
        technique = int(input('enter 1, 2 or 3: '))
    return technique


# indexes array is an array of (x, y) data points
def generate_data(data_num, class_num):
    if class_num == 2: centers = [(-1, -1), (1, 1)]
    elif class_num == 4: centers = [(-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5), (1.5, -1.5)]
    indexes, labels = datasets.make_blobs(n_samples=data_num, centers=centers, cluster_std=0.4, random_state=20)
    return indexes, labels


# returns an array with data_num rows and output_num columns
# 2 --> [0, 0, 1, 0]
def one_hot(labels):
    labels_onehot = []
    for label in labels:
        row = [0 for _ in range(max(labels) + 1)]
        row[label] = 1
        labels_onehot.append(row)
    return labels_onehot


# in this function predicted one-hot labels are turend to class numbers and returned.
# the reason I hard coded this part so for each label a particular number is returned is that I wanted colors of
# train and test dots that are in the same class be the same.
# The whole if and elif block could be replaced by the commented for loop but the colors will not match.
def onehot_reverse(labels_pred_onehot, output_num):
    labels_pred = []
    if output_num == 2:
        for pred in labels_pred_onehot:
            if np.sum(pred) == 1:
                if pred[0]: labels_pred.append(0)
                elif pred[1]: labels_pred.append(3)
            else:
                labels_pred.append(2)
    elif output_num == 4:
        for pred in labels_pred_onehot:
            if np.sum(pred) == 1:
                if pred[0]: labels_pred.append(0)
                elif pred[1]: labels_pred.append(2)
                elif pred[2]: labels_pred.append(4)
                elif pred[3]: labels_pred.append(5)
            else:
                labels_pred.append(3)

    # for pred in labels_pred_onehot:
    #     if np.sum(pred) == 1:
    #         labels_pred.append(pred.tolist().index(1))
    #     else:
    #         labels_pred.append(len(pred))

    return labels_pred


def represent(indexes, labels,  xx, yy, labels_pred):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3)
    plt.sca(ax)
    plt.title("Decision Boundary")
    plt.contourf(xx, yy, np.reshape(labels_pred, xx.shape), cmap='seismic', alpha=0.8)
    plt.scatter(indexes[:, 0], indexes[:, 1], c=labels, edgecolors='gray', cmap='seismic')
    plt.show()


def main():
    data_num = 100  # data for each cluster
    input_num = 2 + 1  # bias, x-index, y-index
    epoch_max = 50
    learning_rate = 0.01

    technique = get_technique()
    if technique == 1:
        output_num = 2  # num of different
        indexes, labels = generate_data(data_num, output_num)
    elif technique == 2:
        output_num = 4
        indexes, labels = generate_data(data_num, output_num)
    else:
        output_num = 2
        get_data = choose_data.GetData()
        get_data.start()
        indexes, labels = get_data.result()
    labels_onehot = one_hot(labels)

    net = perceptron.Net(input_num, output_num, learning_rate)
    for i in range(epoch_max):
        misclassified = net.train(indexes, labels_onehot)
        if misclassified == 0:
            break

    step = 0.1
    xx, yy = np.meshgrid(np.arange(-3, 3, step), np.arange(-3, 3, step))
    points = np.c_[xx.ravel(), yy.ravel()]

    labels_pred_onehot = net.predict(points)
    labels_pred = onehot_reverse(labels_pred_onehot, output_num)

    represent(indexes, labels,  xx, yy, labels_pred)


if __name__ == '__main__':
    main()

