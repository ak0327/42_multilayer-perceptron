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


def plot_multiple_models(max_itr, models_results, figsize=(10, 10)):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models_results)))

    for i, result in enumerate(models_results):
        color = colors[i]
        name = result['name']

        # Loss plot
        ax1.plot(result['iterations'], result['train_losses'], color=color, linestyle='-', label=f'{name} Train Loss')
        ax1.plot(result['iterations'], result['valid_losses'], color=color, linestyle='--', label=f'{name} Valid Loss')

        # Accuracy plot
        ax2.plot(result['iterations'], result['train_accs'], color=color, linestyle='-', label=f'{name} Train Acc')
        ax2.plot(result['iterations'], result['valid_accs'], color=color, linestyle='--', label=f'{name} Valid Acc')

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')
    ax1.set_xlim(0, max_itr)

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curves')
    ax2.set_xlim(0, max_itr)

    plt.tight_layout()
    plt.show()
