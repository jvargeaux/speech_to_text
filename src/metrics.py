from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn
from matplotlib import figure
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

if TYPE_CHECKING:
    from torch import Tensor


class Metrics:
    def __init__(self, debug: bool = False) -> None:
        self.plots = []
        self.debug = debug

    @staticmethod
    def draw_confusion_matrix(target: Tensor, predicted: Tensor, xlabel: str | None = None, ylabel: str | None = None) -> None:
        cm = confusion_matrix(target, predicted)
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot()
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

    def add_heatmap(self, data: Tensor, xlabel: str | None = None, ylabel: str | None = None) -> None:
        self.plots.append({
            'data': data.clone().detach().cpu(),
            'xlabel': xlabel,
            'ylabel': ylabel,
        })

    def draw_heatmaps(self) -> figure:
        num_cols = math.ceil(math.sqrt(len(self.plots)))
        num_rows = math.ceil(len(self.plots) / num_cols)
        # At least 2x2 grid
        MIN_DIM = 2
        num_rows = max(num_rows, MIN_DIM)
        num_cols = max(num_cols, MIN_DIM)
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
        return fig

    @staticmethod
    def show_heatmap(data: Tensor, xlabel: str | None = None, ylabel: str | None = None) -> None:
        data_clone = data.clone().detach().requires_grad_(False)
        seaborn.heatmap(data=data_clone)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.show()
