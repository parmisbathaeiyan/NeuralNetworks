"""
This program uses SVM neural network to train an image dataset of numbers.
"""

from cv2 import imread, IMREAD_GRAYSCALE
import glob
from sklearn.model_selection import train_test_split
import feature_extraction as fe
import numpy as np
from sklearn import svm


def present_result(result):
    print('\nTable below represents number of data prediction in each cell.'
          '\nRows are actual values and columns are predicted values.\n')
    txt = ''
    for i in range(10):
        txt += "{:^6}".format(i)
    print(' ' * 8 + txt)
    print(' ' * 7 + '-' * 60)
    for i in range(10):
        print("{:^6}| ".format(i), end='')
        for j in range(10):
            print("{:^6}".format(int(result[i][j])), end='')
        print()


def main():
    # dividing paths of images into train and test
    PATH = './bmp/*.bmp'
    files = []
    files.extend(glob.glob(PATH))
    labels = [int(file.split('_')[0][-1]) for file in files]
    paths_train, paths_test, labels_train, labels_test = train_test_split(
        files, labels, test_size=0.4)

    # reading images
    images_train = [imread(file, IMREAD_GRAYSCALE) for file in paths_train]
    images_test = [imread(file, IMREAD_GRAYSCALE) for file in paths_test]

    # calculating block mean of images as features
    data_train = fe.block_ave(images_train)
    data_test = fe.block_ave(images_test)

    # making SVM network and training and predicting
    model = svm.LinearSVC(C=10, max_iter=50000)
    model.fit(data_train, labels_train)
    prediction = model.predict(data_test)

    # creating a table representing predictions
    result = np.zeros((10, 10))
    for real, pred in zip(labels_test, prediction):
        result[real, pred] += 1

    present_result(result)
    print('\nAccuracy: %.3f' % model.score(data_test, labels_test))


if __name__ == '__main__':
    main()

