"""
This program compares different methods of classification. More precisely, this program gets image data
of numbers, then trains four networks. One uses block average features of this dataset, one uses hog features of it,
one get both block-average and hog features as input, and eventually, we use ensemble method to find average of
the first two networks and then classify and calculate loss.
At the end of each set of epochs, loss for all four techniques is displayed.
"""

from cv2 import imread, IMREAD_GRAYSCALE
import glob
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
import mlp
import feature_extraction as fe
import numpy as np


def one_hot(arr):
    width = max(arr)
    arr_expanded = []
    for element in arr:
        element_expanded = [0 for i in range(width+1)]
        element_expanded[element] = 1
        arr_expanded.append(element_expanded)
    return arr_expanded


def present(each_round, epoch, loss_block, loss_hog, loss_ave, loss_total):
    x = range(each_round, epoch+2, each_round)
    plt.cla()
    plt.title("Loss for Test Dataset")
    plt.plot(x, loss_block, color='crimson', label='MLP with Block Average', alpha=0.4, marker='o')
    plt.plot(x, loss_hog, color='green', label='MLP with HOG', alpha=0.4, marker='o')
    plt.plot(x, loss_ave, color='blue', label='Average Ensemble', alpha=0.4, marker='o')
    plt.plot(x, loss_total, color='orange', label='All Features', alpha=0.4, marker='o')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.pause(0.01)


def main():
    MAX_EPOCH = 2000
    LEARNING_RATE = 0.1
    PATH = './bmp/*.bmp'
    each_round = 20
    BATCH_SIZE = 20

    files = []
    files.extend(glob.glob(PATH))
    labels = [int(file.split('_')[0][-1]) for file in files]
    paths_train, paths_test, labels_train, labels_test = train_test_split(
        files, labels, test_size=0.4)

    # reading images
    images_train = [imread(file, IMREAD_GRAYSCALE) for file in paths_train]
    images_test = [imread(file, IMREAD_GRAYSCALE) for file in paths_test]

    # expanding labels
    labels_train_expanded = one_hot(labels_train)
    labels_test_expanded = one_hot(labels_test)

    # preparing block-average features
    train_block = fe.block_ave(images_train)
    test_block = fe.block_ave(images_test)
    net_block = mlp.MLP(LEARNING_RATE, BATCH_SIZE, [64, 20, 10])

    # preparing hog features
    train_hog = fe.hog_extract(images_train)
    test_hog = fe.hog_extract(images_test)
    net_hog = mlp.MLP(LEARNING_RATE, BATCH_SIZE, [144, 20, 10])

    # preparing a set containing both prepared features
    train_total = np.concatenate((train_block, train_hog), axis=1)
    test_total = np.concatenate((test_block, test_hog), axis=1)
    net_total = mlp.MLP(LEARNING_RATE, BATCH_SIZE, [208, 20, 10])

    loss_block = []
    loss_hog = []
    loss_ave = []
    loss_total = []
    for epoch in range(MAX_EPOCH):
        net_block.train(train_block, labels_train_expanded)
        net_hog.train(train_hog, labels_train_expanded)
        net_total.train(train_total, labels_train_expanded)
        print('\r Epoch: %d' % (epoch+1), end='')

        if epoch % each_round == each_round - 1:
            prediction_block, loss = net_block.predict(test_block, labels_test_expanded)
            loss_block.append(loss)
            prediction_hog, loss = net_hog.predict(test_hog, labels_test_expanded)
            loss_hog.append(loss)
            prediction_ave = (prediction_block.data.numpy() + prediction_hog.data.numpy()) / 2
            loss_ave.append(metrics.mean_squared_error(prediction_ave, labels_test_expanded))
            prediction_total, loss = net_total.predict(test_total, labels_test_expanded)
            loss_total.append(loss)

            present(each_round, epoch, loss_block, loss_hog, loss_ave, loss_total)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
