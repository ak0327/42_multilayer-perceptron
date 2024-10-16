import os
import sys
sys.path.append(os.pardir)


import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation


class RealtimePlot:
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.iterations = []
        self.train_losses = []
        self.train_accs = []
        self.valid_losses = []
        self.valid_accs = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.ion()  # インタラクティブモード

        self.dt = 0.00001

    def update(self, i, train_loss, train_acc, valid_loss, valid_acc):
        self.iterations.append(i)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.valid_losses.append(valid_loss)
        self.valid_accs.append(valid_acc)

        self._plot()
        plt.pause(self.dt)

    def _plot(self):
        self.ax1.clear()
        self.ax1.plot(self.iterations, self.train_losses, color='C0', linestyle='-', label='Train Loss')
        self.ax1.plot(self.iterations, self.valid_losses, color='C1', linestyle='--', label='Valid Loss')
        self.ax1.set_xlabel('Iterations')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.set_xlim(0, self.max_iter)

        self.ax2.clear()
        self.ax2.plot(self.iterations, self.train_accs, color='C0', linestyle='-', label='Train Acc')
        self.ax2.plot(self.iterations, self.valid_accs, color='C1', linestyle='--', label='Valid Acc')
        self.ax2.set_xlabel('Iterations')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        self.ax2.set_xlim(0, self.max_iter)
        self.ax2.set_ylim(0, 1)

    def plot(self):
        plt.ioff()
        plt.show()
