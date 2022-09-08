import numpy as np
import torch
import cv2
from skimage.feature import hog


def block_ave(data):
    def resize(images):
        resized = []
        for image in images:
            row_num = image.shape[0]
            col_num = image.shape[1]

            length = torch.linspace(0, row_num, 9).data.numpy()
            width = torch.linspace(0, col_num, 9).data.numpy()

            num = [0 for j in range(8)]
            rows8 = [[0 for i in range(col_num)] for j in range(8)]
            for i in range(row_num):
                for l in range(1, 9):
                    if i <= length[l]:
                        temp = np.array(rows8[l - 1]) * num[l - 1]
                        rows8[l - 1] = np.add(temp, image[i]) / (num[l - 1] + 1)
                        num[l - 1] += 1
                        break

            num = [0 for j in range(8)]
            data8 = [[0 for i in range(8)] for j in range(8)]
            for i in range(col_num):
                for w in range(1, 9):
                    if i <= width[w]:
                        temp = np.array(data8[w - 1]) * num[w - 1]
                        data8[w - 1] = np.add(temp, np.array(rows8).transpose()[i]) / (num[w - 1] + 1)
                        num[w - 1] += 1
                        break

            data8 = np.array(data8).transpose()
            resized.append(np.array(data8))
        return resized

    def flatten(arr3d):
        flat = []
        for arr2d in arr3d:
            flat.append(np.matrix.flatten(np.array(arr2d)))
        return flat

    resized = resize(data)
    flattened = flatten(resized)
    # normalized = preprocessing.normalize(flattened)
    normalized = np.array(flattened) / 255
    return normalized


def hog_extract(data):
    features_all = []
    for img in data:
        img = cv2.resize(img, (64, 64))
        features = hog(img, pixels_per_cell=(13, 13), cells_per_block=(1, 1))
        features_all.append(features)
    return np.array(features_all)
