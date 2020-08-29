from causalml.dataset import *
import matplotlib.pyplot as plt


def treated_vs_untreated_plot(y, w, setup, subplot):
    plt.subplot(subplot)
    alpha = 0.2
    bins = 150
    plt.hist(y[w == 1], alpha=alpha, bins=bins, label='treated')
    plt.hist(y[w == 0], alpha=alpha, bins=bins, label='untreated')

    plt.axis([-5, 5, 0, 20])
    plt.title(setup, fontsize=15)

    plt.grid(True)


y_easy, X_easy, w_easy, tau_easy, b_easy, e_easy = simulate_nuisance_and_easy_treatment(n=1000, p=5)
y_rand, X_rand, w_rand, tau_rand, b_rand, e_rand = simulate_randomized_trial(n=1000, p=5)
y_difficult, X_difficult, w_difficult, tau_difficult, b_difficult, e_difficult = simulate_easy_propensity_difficult_baseline(
    n=1000, p=5)