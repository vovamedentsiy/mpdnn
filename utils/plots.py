import matplotlib
matplotlib.use('PS')
from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np


import random


def autolabel(rects, ax, coeff = 1):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        #ax.annotate('{:.3f}'.format(height, prec=3),
        ax.annotate('{:.1f}%'.format(height * 100, prec=1),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=1.0)


def plot_weight_histograms(frequencies, names, name_to_save):

    fig, ax = plt.subplots(5, 4, figsize=(15, 10))
    fig.tight_layout()
    i = 0

    for row in ax:
        for ax_ in row:
            plt.sca(ax_)
            freqs = list(frequencies[i].values())
            grid = np.array(list(frequencies[i].keys()))
            bits_info = str(int(np.ceil(np.log2(len(grid)))))
            if len(grid) > 32:
                # not to overload plots
                freqs = freqs[:16]
                grid = grid[:16]
                bits_info += ' No plot'

            rects1 = plt.bar(grid , width=0.75, height=freqs)
            autolabel(rects1, ax_)
            plt.title(names[i] + ' B'+bits_info, size=10)
            plt.xticks([], [])
            plt.yticks([], [])
            i+=1

    filename = name_to_save + '.eps'
    plt.savefig(filename, format='eps')
    plt.close()