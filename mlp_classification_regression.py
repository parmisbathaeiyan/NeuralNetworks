"""
This program tries to do regression on a curve (func1 or func2) using
mlp_manual module and represents the estimated curve throughout training.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
import mlp_manual


# this generates data according to one function. func1 or func2 can be used.
def generate_data(num):
    X = torch.linspace(-1, 1, num).data.numpy()
    Y = func1(X) + 0.2 * np.random.normal(0, 0.2, len(X))

    X_real = torch.linspace(-1, 1, 1000).data.numpy()
    Y_real = func1(X_real)
    return X, Y, X_real, Y_real


def func1(x):
    y = 0.5 * (np.sin(2 * np.pi * (x**2)) + 1)
    return y


def func2(x):
    y = 0.75 * (x * np.sin(2 * np.pi * (x**2)) - 1)
    return y


def present_initial(X, Y, X_real, Y_real):
    plt.cla()
    plt.scatter(X, Y, label='Data', s=10, c='green')
    plt.plot(X_real, Y_real, label='Actual function', linewidth=1.5, c='black', alpha=0.8)


def represent(X, predicted, loss, epoch, line, text):
    if line: line.pop(0).remove()
    if text: text.remove()
    line = plt.plot(X, np.array(predicted).flatten(), label='Output',  linewidth=1.5, c='red', alpha=0.8)
    text = plt.text(-0.2, 1.13, 'Loss=%.4f\nEpoch=%d' % (loss, epoch+1), fontdict={'size': 11, 'color': 'black'})
    plt.legend()
    plt.pause(0.1)
    return line, text


def main():
    data_num = 100
    max_epoch = 2000
    max_loss = 0.01
    learning_rate = 0.02
    neuron_hidden_num = 40
    input_num = 1
    output_num = 1

    X, Y, X_real, Y_real = generate_data(num=data_num)

    plt.figure()
    present_initial(X, Y, X_real, Y_real)

    net = mlp_manual.MLP(input_num, output_num, neuron_hidden_num, learning_rate, mlp_manual.tanh)
    line = None
    text = None
    for epoch in range(max_epoch):
        loss = net.train(X, Y)
        if epoch % 15 == 4:
            loss, predicted = net.predict(X, Y)
            line, text = represent(X, predicted, loss, epoch, line, text)
        if loss < max_loss:
            break
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()

