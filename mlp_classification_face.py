"""
This program uses MLP net from mlp module to train ORL face dataset. This dataset is first
divided into test and train. After training, the result of prediction of network on test data
along with total loss is displayed.
"""

import glob
import os
import random
import cv2
import numpy as np
import torch
from torch.nn.functional import one_hot
from skimage.io import imread
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import mlp


# each folder contains one class' data. each folder's image
# paths are found and are split into train and test dataset
def read_files(root_dir):
    labels = []
    folders = []
    for it in os.scandir(root_dir):
        if it.is_dir():
            path = it.path
            folders.append(path)
            labels.append(int(path.split('s')[2]))

    files_train = []
    files_test = []
    for folder in folders:
        files = []
        files.extend(glob.glob(folder+'/*.pgm'))
        paths_train, paths_test, temp, temp = train_test_split(
            files, np.zeros(np.array(files).shape), test_size=0.4)
        files_train.append(paths_train)
        files_test.append(paths_test)

    labels_train = []
    labels_test = []
    for label in labels:
        labels_train.append([label-1] * len(files_train[0]))
        labels_test.append([label-1] * len(files_test[0]))

    labels_train = np.array(labels_train).flatten()
    labels_test = np.array(labels_test).flatten()
    files_train = np.array(files_train).flatten()
    files_test = np.array(files_test).flatten()

    return files_train, files_test, labels_train, labels_test


# gets one label's images and returns a 2d array rows being each image, columns being features
def extract(files):
    features_all = []
    for file in files:
        img = imread(file, as_gray=True)
        img = cv2.resize(img, (64, 64))
        features = hog(img, pixels_per_cell=(13, 13), cells_per_block=(1, 1))
        features_all.append(features)
    return features_all


def represent(labels, prediction, loss):
    print('\nLoss: ', loss)
    for (label, pred) in zip(labels, prediction):
        chosen = torch.argmax(pred)
        print(f'{label+1} was predicted {chosen+1}')


def main():
    learning_rate = 0.1
    batch_size = 5
    epoch_num = 500
    files_train, files_test, labels_train, labels_test = read_files('./orl_faces')

    # shuffling train data so net won't get fit to the last input classes
    random.seed(0)
    random.shuffle(files_train)
    random.seed(0)
    random.shuffle(labels_train)

    labels_train_onehot = one_hot(torch.tensor(labels_train))
    labels_test_onehot = one_hot(torch.tensor(labels_test))

    train_whole = np.array(extract(files_train))
    test_whole = np.array(extract(files_test))

    net = mlp.MLP(learning_rate, batch_size, [144, 30, 40])

    for epoch in range(epoch_num):
        print('\rEpoch Number: %d/%d' % (epoch+1, epoch_num), end='')
        net.train(train_whole, labels_train_onehot)

    prediction, loss = net.predict(test_whole, labels_test_onehot)
    represent(labels_test, prediction, loss)


if __name__ == '__main__':
    main()
