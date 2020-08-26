import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def do_stacking(predictions_dict, true_te):
    """
    TODO: Unfortunately the stacking only works for predictions which return an ndarray: (n, ), which is not the case for some ML models provided by sklearn. above maybe refactor the predictions generating functions to always wrangle into ndarray(n, )

    :param predictions_dict: dict of predictions (genereated from simple_model.make_simple_predictions_for_stacking)
    :param true_te: the treatment effect provided with the causalml.datasets synthetic data generators
    :return: a dict of params.
    """
    x_stacking = np.vstack((predictions_dict.values())).T

    model = sm.OLS(true_te, x_stacking)
    model2 = model.fit_regularized(alpha=0.0, L1_wt=1.0, start_params=None, profile_scale=False, refit=False)
    return model2


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

def evaluate_models_compare_to_stacking(predictions_dict, true_treatment_effect, stacking_predictions):
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