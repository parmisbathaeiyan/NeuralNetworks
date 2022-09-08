"""
This module shows a -3 to 3 2D visual coordinate plane so users can choose desired indexes in two classes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button


class GetData:
    def start(self):
        self.current_color = 'red'
        self.data_red = []
        self.data_blue = []

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.subplots_adjust(left=0.3)
        plt.sca(self.ax)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)

        rax = plt.axes([0.05, 0.4, 0.15, 0.15], facecolor='azure')
        radio2 = RadioButtons(rax, ('red', 'blue'))
        radio2.on_clicked(self.color_func)

        axes = plt.axes([0.05, 0.3, 0.15, 0.075])
        bax = Button(axes, 'Train', color='paleturquoise')
        bax.on_clicked(self.done)

        self.represent()

    def onclick(self, event):
        if event.inaxes == self.ax:
            x = event.xdata
            y = event.ydata
            self.data_red.append([x, y]) if self.current_color == 'red' else self.data_blue.append([x, y])
            self.represent()

    def represent(self):
        if self.data_red:
            self.ax.plot(np.array(self.data_red)[:, 0], np.array(self.data_red)[:, 1], '.', color='red')
        if self.data_blue:
            self.ax.plot(np.array(self.data_blue)[:, 0], np.array(self.data_blue)[:, 1], '.', color='blue')
        plt.show()

    def color_func(self, label):
        self.current_color = label
        plt.draw()

    def done(self, event):
        plt.close()

    def result(self):
        indexes = self.data_red + self.data_blue
        labels = [1] * len(self.data_red) + [0] * len(self.data_blue)
        return np.array(indexes), labels


