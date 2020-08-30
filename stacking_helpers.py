import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from mlens.ensemble import SuperLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
from causalml.dataset import *
from causalml.metrics import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
import pandas as pd

import copy

import simple_model


def do_stacking(predictions_dict, true_te):
    """
    :param predictions_dict: dict of predictions (genereated from simple_model.make_simple_predictions_for_stacking)
    :param true_te: the treatment effect provided with the causalml.datasets synthetic data generators

    :return: a dict of params.
    """
    pd = copy.deepcopy(predictions_dict)
    pd.pop('Actuals')
    pd.pop('generated_data')

    x_stacking = np.vstack((pd.values())).T

    model = sm.OLS(true_te, x_stacking)
    model2 = model.fit_regularized(alpha=0.0, L1_wt=1.0, start_params=None, profile_scale=False, refit=False)
    return model2

def do_stacking_simple_models(regressors, X, y, w, meta):
    """
    do stacking witht the mlens library.

    :param regressors: a dict of regressors to feed into the ensemble pipeline
    :param X: training dataset
    :param y: outcome varaible y
    :param w: assignment variable
    :param meta: regressor (found in regressors dict for ensemble)

    :return: CATE predictions from the ensemble estimator
    """

    ensemble = SuperLearner(scorer=mean_squared_error, random_state=42)
    ensemble.add([x for x in regressors.values()])
    ensemble.add_meta(regressors[meta])

    e_preds, tau_test = simple_model.create_simple_ml_model(X, y, w, ensemble)

    return e_preds

##need a way to evaluate stacking.

def plot_stacking_preds(model, true_te):
    """
    :param model:
    :param true_te:
    :return: plot a matplotlib.pyplot plot (histogram of predictions compared to the true treatment effect.)
    """
    alpha = 0.4
    bins = 20

    stacking_fitted_vals = model.fittedvalues
    weights = model.params

    plt.hist(stacking_fitted_vals, alpha=alpha, bins=bins, label='predicted values (stacking)')
    plt.hist(true_te, alpha=alpha, bins=bins, label='true treatment effect')

    plt.axis([-10, 10, 0, 200])
    plt.title('predictions made by tacking model, vs true treatment effect')
    plt.xlabel('Individual Treatment Effect (ITE/CATE)')
    plt.ylabel('# of Samples')
    _ = plt.legend()
    plt.show()

def evaluate_models_compare_to_stacking_mse(predictions_dict, true_treatment_effect, stacking_predictions):
    """
    :param predictions_dict:
    :param true_treatment_effect:
    :param stacking_predictions: an array of predicitons given by the stacking model
    :return: a table/array of evaluation metrics for each model
    """

    mse_dict = {}
    for key in predictions_dict.keys():
        mse_dict[key] = mean_squared_error(true_treatment_effect, predictions_dict[key])

    mse_dict['stacking'] = mean_squared_error(true_treatment_effect, stacking_predictions)
    return mse_dict

def evaluate_models_compare_to_stacking_r_square(predictions_dict, true_treatment_effect, stacking_predictions):
    """
    :param predictions_dict:
    :param true_treatment_effect:
    :param stacking_predictions: an array of predicitons given by the stacking model
    :return: a table/array of evaluation metrics for each model
    """

    r2_dict = {}
    for key in predictions_dict.keys():
        r2_dict[key] = r2_score(true_treatment_effect, predictions_dict[key])

    r2_dict['stacking'] = r2_score(true_treatment_effect, stacking_predictions)
    return r2_dict


def show_MSE_r_square(indv_predictions, tau_test, ensemble_predictions):
    """
    :param indv_predictions: dict of predicted treatmenet effects from individual, somple models
    :param tau_test: true treatment effect (corresponding to predictions of indv_predictions)
    :param ensemble_predictions: predicted treatmenet effect of the ensemble
    :return: visualize a table of mse and R squared.
    """

    mseDict = evaluate_models_compare_to_stacking_mse(indv_predictions, tau_test, ensemble_predictions)
    r2Dict = evaluate_models_compare_to_stacking_r_square(indv_predictions, tau_test, ensemble_predictions)

    mseDf = pd.DataFrame(mseDict, index=['MSE',])
    r2DF = pd.DataFrame(r2Dict, index=['R-Squared',])

    display(mseDf)
    display(r2DF)