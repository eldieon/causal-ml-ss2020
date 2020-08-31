from causalml.dataset import *
from causalml.inference.meta import BaseRRegressor
from causalml.inference.meta import BaseTRegressor
from causalml.metrics.visualize import *
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def treated_vs_untreated_plot(y, w, setup, subplot):
    """

    :param y: vector of individual outcomes
    :param w: treatment dummy
    :param setup: name of data generating function
    :param subplot: param for matplotlib.plot
    :return: a histogram showing the distribution of outcomes over treated vs no treated observations
    """
    plt.subplot(subplot)
    alpha = 0.2
    bins = 150
    plt.hist(y[w == 1], alpha=alpha, bins=bins, label='treated')
    plt.hist(y[w == 0], alpha=alpha, bins=bins, label='untreated')

    plt.axis([-4, 6, 0, 100])
    plt.title(setup, fontsize=15)

    plt.grid(True)


def generate_predicitons_by_learner(estimators):
    """

    :param estimators: dict of estimators from causal ml
    :return: dict of synthetic predictions, for training and test, to use in an ensemble
    """
    predictions = {}
    predictions_easy_treatment = get_synthetic_preds(simulate_nuisance_and_easy_treatment,
                                                     n=1000,
                                                     estimators=estimators)
    predictions_easy_treatment_test = get_synthetic_preds(simulate_nuisance_and_easy_treatment,
                                                          n=1000,
                                                          estimators=estimators)

    predictions_randomized_trial = get_synthetic_preds(simulate_randomized_trial,
                                                       n=1000,
                                                       estimators=estimators)
    predictions_randomized_trial_test = get_synthetic_preds(simulate_randomized_trial,
                                                            n=1000,
                                                            estimators=estimators)

    predictions_easy_propensity = get_synthetic_preds(simulate_easy_propensity_difficult_baseline,
                                                      n=1000,
                                                      estimators=estimators)
    predictions_easy_propensity_test = get_synthetic_preds(simulate_easy_propensity_difficult_baseline,
                                                           n=1000,
                                                           estimators=estimators)

    predictions['predictions_easy_treatment'] = predictions_easy_treatment
    predictions['predictions_easy_treatment_test'] = predictions_easy_treatment_test
    predictions['predictions_randomized_trial'] = predictions_randomized_trial
    predictions['predictions_randomized_trial_test'] = predictions_randomized_trial_test
    predictions['predictions_easy_propensity'] = predictions_easy_propensity
    predictions['predictions_easy_propensity_test'] = predictions_easy_propensity_test

    return predictions

estimators_R = {#'learner_dtr': BaseRRegressor(learner=DecisionTreeRegressor()),
                'learner_xgb': BaseRRegressor(learner=XGBRegressor()),
                'learner_lr': BaseRRegressor(learner=LinearRegression())}


estimators_T = {'learner_xgb': BaseTRegressor(learner=XGBRegressor()),
                'learner_lr': BaseTRegressor(learner=LinearRegression())}
