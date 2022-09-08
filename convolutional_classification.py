"""
Images of numbers are fed to a Convolutional network and after training,
accuracy af the network on test data is printed.
"""

import cv2
import numpy as np
from cv2 import imread, IMREAD_GRAYSCALE
import glob
from sklearn.model_selection import train_test_split
# in case an error is shown on the next line, it won't have any effect when runtime
from tensorflow.keras.utils import to_categorical
from keras import layers, models


print('Preparing Data...')
# finding paths of images and deviding them into train and test section
path = './bmp/*.bmp'
files = []
files.extend(glob.glob(path))
labels = [int(file.split('_')[0][-1]) for file in files]
paths_train, paths_test, labels_train, labels_test = train_test_split(
    files, labels, test_size=0.4)

# making labels one-hot
labels_train = to_categorical(labels_train)
labels_test = to_categorical(labels_test)

# reading and resizing images
size = 28
images_train = [cv2.resize(imread(file, IMREAD_GRAYSCALE), (size, size)) for file in paths_train]
images_test = [cv2.resize(imread(file, IMREAD_GRAYSCALE), (size, size)) for file in paths_test]

# normalizing images
images_train = np.array(images_train)/255
images_test = np.array(images_test)/255

# making image data 3D
images_train = np.expand_dims(images_train, -1)
images_test = np.expand_dims(images_test, -1)

print('Making the Model...')
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'CategoricalCrossentropy'])

print('Training the Model...')
model.fit(images_train, labels_train, epochs=10, batch_size=50)

print('Testing the Model...')
test_loss, test_acc, temp = model.evaluate(images_test, labels_test)

print('Accuracy: ', test_acc)

