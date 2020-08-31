import matplotlib.pyplot as plt
import pandas as pd
from causalml.dataset import *
from causalml.metrics import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


"""
HERE U EXPLAIN WHAT THIS SCRIPT IS FOR. 
"""

def create_simple_ml_model(X, y, w, tau, reg_):
    """
    helper function for make_simple_predictions_for_stacking(X_train, X_test, y_train, w_train, regressors)
    follows the methodology of the S-learner; fit a model to treatment and control groups together,
    and then estimate the treatment effect as the difference of each function.

    :param X: X variables (floats)
    :param y: endogenous variable (float)
    :param w: assignment to treatment (0 or 1)
    :param tau: true treatment effect
    :param reg_: a machine learning model instance
    :return: the estimated treatment effect for the testing set, as the difference between predicted outcome of treated and predicted outcome of untreated, corresponding true treatmenet effect for each observation
    """

    y_train, y_test, X_train, X_test, w_train, w_test, tau_train, tau_test = train_test_split(y, X, w, tau,
                                                                                              test_size=0.25,
                                                                                              random_state=42)
    #learn the treatment and control groups together
    df = pd.DataFrame(data=X_train)
    df['assignment'] = w_train

    y_df = pd.DataFrame(data=y_train)
    reg = reg_.fit(df, y_df)

    X_neg = pd.DataFrame(data=X_test)
    X_neg['assignment'] = 0

    X_pos = pd.DataFrame(data=X_test)
    X_pos['assignment'] = 1
    ret = reg.predict(X_pos) - reg.predict(X_neg)  ## this could just return predicitons for positive and negative...
    return ret, tau_test


def make_simple_predictions_for_stacking(X, y, w, tau, regressors):
    """
    :param X: X variables (floats)
    :param y: endogenous variable (float)
    :param w: assignment to treatment (0 or 1)
    :param regressors: dictionary of name, regressor
    :return: dict of regressors (name as string), predictions
    """
    predictions_dict = {}

    for name in regressors.keys():
        preds, tau_test = create_simple_ml_model(X, y, w, tau, regressors[name])
        predictions_dict[name] = preds

    return predictions_dict, tau_test


# TODO: add average line to plot, add true treatment effect to plot.
def multilayer_hist(dictionary, true_vals, subplot, xmin=-4, xmax=4, ymin=0, ymax=80):
    """
    :param dictionary: a dictionary of predictions, keys as name of model
    :param true_vals: true treatment effect or outcome prediction
    :param subplot: param for matplotlib.plot subplot
    :param xmin, xmax, ymin, ymax: param for matplotlib.plot subplot axis limits
    :return: a plot of predicitons from each model in the dict
    """
    plt.subplot(subplot)
    alpha = 0.2
    bins = 50
    for name, predictions in dictionary.items():
        plt.hist(predictions, alpha=alpha, bins=bins, label=name)

    plt.hist(true_vals, alpha=alpha, bins=bins, label='te')

    plt.axis([xmin, xmax, ymin, ymax])
    _ = plt.legend()


####it doesnt make sense to plot all predictors individually against the same true treatment effect when they can just as easily fit on the same plot.
def multilayer_hist_individual_regressors(dictionary, true_vals):
    """
    :param dictionary: a dictionary of predictions, keys as name of model
    :return: a plot of predicitons from each model in the dict
    """
    alpha = 0.2
    bins = 250

    for name, predictions in dictionary.items():
        plt.hist(predictions, alpha=alpha, bins=bins, label=name)
        plt.hist(true_vals, alpha=alpha, bins=bins, label='te')

        plt.axis([-5, 5, 0, 60])
        plt.title('simple predictions of individual treatment effect.')
        plt.xlabel('Individual Treatment Effect (ITE/CATE)')
        plt.ylabel('number of observations')
        _ = plt.legend()
        plt.show()
        print("r2 score is:" + str(r2_score(true_vals, predictions)))
        print("MSE is:" + str(mean_squared_error(true_vals, predictions)))


def get_treated_obs_for_evaulation(X, y, te, w):
    """
    generates a dataset for generating predictions as, extracts only the treated in order to create a dataset for fitting a stacking model
    :param X: X variables (floats)
    :param y: endogenous variable (float)
    :param te: treatment effect
    :param w: assignment to treatment (0 or 1)
    :return:
    """

    new_data = X[w == 1]
    new_outcome = y[w == 1]
    new_treatment_effect = te[w == 1]

    return new_data, new_outcome, new_treatment_effect


def compare_single_models(X, y, w, tau, regressors):
    """
    :param regressors: dict of regressors to make predcitons for treatment effect
    :param X: Dataset of independent variables
    :param y: outcome variable
    :param w: assignment variable
    :param tau: true treatment effect
    :return: predicitons dict, true treatmenet effect of the test set
    """

    predicitons, tau_test = make_simple_predictions_for_stacking(X, y, w, tau, regressors)

    return predicitons, tau_test
