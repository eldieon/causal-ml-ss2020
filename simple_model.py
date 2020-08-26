import matplotlib.pyplot as plt
import pandas as pd

"""
HERE U EXPLAIN WHAT THIS SCRIPT IS FOR. 
"""


def create_simple_ml_model(X, y, w, reg_):
    """
    helper function for make_simple_predictions_for_stacking(X, y, w, regressor, name, dict)

    :param X: X variables (floats)
    :param y: endogenous variable (float)
    :param w: assignment to treatment (0 or 1)
    :param regressor: a machine learning model
    :return: the estimated treatment effect, as the difference between predicted outcome of treated and predicted outcome of untreated.
    """

    df = pd.DataFrame(data=X)
    df['assignment'] = w

    y_df = pd.DataFrame(data=y)
    reg = reg_.fit(df, y_df)

    X_neg = pd.DataFrame(data=X)
    X_neg['assignment'] = 0

    X_pos = pd.DataFrame(data=X)
    X_pos['assignment'] = 1
    ret = reg.predict(X_pos) / reg.predict(X_neg)
    return ret


def make_simple_predictions_for_stacking(X, y, w, regressors):
    """
    :param X: X variables (floats)
    :param y: endogenous variable (float)
    :param w: assignment to treatment (0 or 1)
    :param regressors: dictionary of name, regressor
    :return: dict of regressors (name as string), predictions
    """
    predictions_dict = {}

    for name in regressors.keys():
        preds = create_simple_ml_model(X, y, w, regressors[name])
        predictions_dict[name] = preds

    return predictions_dict


# TODO: add average line to plot, add true treatment effect to plot.
def multilayer_hist(dict):
    """
    :param dict: a dictionary of predictions, keys as name of model
    :return: a plot of predicitons from each model in the dict
    """

    alpha = 0.2
    bins = 1000

    for name, predictions in dict.items():
        plt.hist(predictions, alpha=alpha, bins=bins, label=name)

    plt.axis([-30, 30, 0, 200])
    plt.title('simple predictions of individual treatment effect.')
    plt.xlabel('Individual Treatment Effect (ITE/CATE)')
    plt.ylabel('# of Samples')
    _ = plt.legend()
    #plt.show()


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

#TODO: generate a table for showing mse of each learner? then mse as in the group ? something like that.


#y_rand, X_rand, w_rand, tau_rand, b_rand, e_rand = simulate_randomized_trial()
#y_easy_p, X_easy_p, w_easy_p, tau_easy_p, b_easy_p, e_easy_p = simulate_easy_propensity_difficult_baseline()
#y_hidden, X_hidden, w_hidden, tau_hidden, b_hidden, e_hidden = simulate_hidden_confounder()

###########################################################################################
#experimenting
###########################################################################################
import numpy as np
from causalml.dataset import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import stacking_helpers


regressors = { 'regressor_nn': MLPRegressor(),
               'regressor_dt' : DecisionTreeRegressor(),
               'regressor_xgb' : XGBRegressor()}

np.random.seed(42) # does this seed affect the causalml generated stuff? i think no..
y_easy_t, X_easy_t, w_easy_t, tau_easy_t, b_easy_t, e_easy_t = simulate_nuisance_and_easy_treatment()
easy_t_preds = make_simple_predictions_for_stacking(X_easy_t, y_easy_t, w_easy_t, regressors) ## needs dict of regressors --

multilayer_hist(easy_t_preds)

thing = stacking_helpers.do_stacking(easy_t_preds, tau_easy_t)

stacking_helpers.plot_stacking_preds(thing, tau_easy_t)

mse_dict = stacking_helpers.evaluate_models_compare_to_stacking(easy_t_preds, tau_easy_t, thing.fittedvalues)