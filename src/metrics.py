import math
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import Tensor


class Metrics():
    def __init__(self, debug=False):
        self.plots = []
        self.debug = debug

    def show_confusion_matrix(self, target, predicted, xlabel = None, ylabel = None):
        if self.debug is False:
            return
        cm = confusion_matrix(target, predicted)
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot()
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.show()

    def add_heatmap(self, data: Tensor, xlabel = None, ylabel = None):
        if self.debug is False:
            return
        self.plots.append({
            'data': data.clone().detach().cpu(),
            'xlabel': xlabel,
            'ylabel': ylabel
        })

    def show_heatmaps(self):
        if self.debug is False:
            return
        num_cols = math.ceil(math.sqrt(len(self.plots)))
        num_rows = math.ceil(len(self.plots) / num_cols)
        # At least 2x2 grid
        if num_rows < 2:
            num_rows = 2
        if num_cols < 2:
            num_cols = 2
        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)

        row = 0
        col = 0
        for plot in self.plots:
            data_clone = plot['data'].clone().detach().requires_grad_(False)
            seaborn.heatmap(ax=ax[row, col], data=data_clone)
            if plot['xlabel'] is not None:
                ax[row, col].set_xlabel(plot['xlabel'])
            if plot['ylabel'] is not None:
                ax[row, col].set_ylabel(plot['ylabel'])
            col += 1
            if (col >= num_cols):
                col = 0
                row += 1

        plt.show()

    def show_heatmap(self, data, xlabel = None, ylabel = None):
        if self.debug is False:
            return
        data_clone = data.clone().detach().requires_grad_(False)
        seaborn.heatmap(data=data_clone)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.show()