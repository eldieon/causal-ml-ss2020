import numpy as np
import statsmodels.api as sm

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