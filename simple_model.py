import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
"""
HERE U EXPLAIN WHAT THIS SCRIPT IS FOR. 
"""


def create_simple_ml_model(X_train, X_test, y_train, w_train, reg_):
    """
    helper function for make_simple_predictions_for_stacking(X_train, X_test, y_train, w_train, regressors)

    :param X_train: X variables (floats) (training set)
    :param X_test: X variables (floats) (testing set)
    :param y_train: endogenous variable (float) (training set)
    :param w_train: assignment to treatment (0 or 1) (training set)
    :param reg_: a machine learning model instance
    :return: the estimated treatment effect for the testing set,
        as the difference between predicted outcome of treated and predicted outcome of untreated.
    """

    df = pd.DataFrame(data=X_train)
    df['assignment'] = w_train

    y_df = pd.DataFrame(data=y_train)
    reg = reg_.fit(df, y_df)

    X_neg = pd.DataFrame(data=X_test)
    X_neg['assignment'] = 0

    X_pos = pd.DataFrame(data=X_test)
    X_pos['assignment'] = 1
    ret = reg.predict(X_pos) - reg.predict(X_neg) ## this could just return predicitons for positive and negative...
    return ret


def make_simple_predictions_for_stacking(X_train, X_test, y_train, w_train, regressors):
    """
    :param X_train: X variables (floats) (training set)
    :param X_test: X variables (floats) (testing set)
    :param y_train: endogenous variable (float) (training set)
    :param w_train: assignment to treatment (0 or 1) (training set)
    :param regressors: dictionary of name, regressor
    :return: dict of regressors (name as string), predictions
    """
    predictions_dict = {}

    for name in regressors.keys():
        preds = create_simple_ml_model(X_train, X_test, y_train, w_train, regressors[name])
        predictions_dict[name] = preds

    return predictions_dict


# TODO: add average line to plot, add true treatment effect to plot.
def multilayer_hist(dictionary, true_vals):
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
        plt.ylabel('# of Samples')
        _ = plt.legend()
        plt.show()
        print("r2 score is:"+str(r2_score(true_vals, predictions)))
        print("MSE is:"+str(mean_squared_error(true_vals, predictions)))
    

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