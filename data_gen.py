from causalml.dataset import *
import matplotlib.pyplot as plt

def treated_vs_untreated_plot(y, w, setup, subplot):
    plt.subplot(subplot)
    alpha = 0.2
    bins = 150
    plt.hist(y[w == 1], alpha=alpha, bins=bins, label='treated')
    plt.hist(y[w == 0], alpha=alpha, bins=bins, label='untreated')

    plt.axis([-4, 6, 0, 100])
    plt.title(setup, fontsize=15)

    plt.grid(True)
