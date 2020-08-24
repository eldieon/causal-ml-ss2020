import sklearn
from causalml.dataset import *
from causalml.inference.meta import BaseRRegressor
from causalml.inference.tree import CausalTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import *
import copy
import numpy as np


from datetime import datetime
datetime.utcnow()

learner_xgb = BaseRRegressor(learner=XGBRegressor())
learner_lr = BaseRRegressor(learner=LinearRegression())
learner_dtr = BaseRRegressor(learner=DecisionTreeRegressor())

###would be cool to find some other working learners, and to start messing with the params of each!!!!
learner_knr = BaseRRegressor(learner=KNeighborsRegressor())
learner_svr = BaseRRegressor(learner=SVR())
learner_ctr = BaseRRegressor(learner=CausalTreeRegressor())
learner_nnr = BaseRRegressor(learner=MLPRegressor()) ##Multi-layer Perceptron regressor
##


estimators = {'learner_xgb': BaseRRegressor(learner=XGBRegressor()),
              'learner_lr': BaseRRegressor(learner=LinearRegression()),
              ##'learner_dtr': BaseRRegressor(learner=DecisionTreeRegressor()),
              'learner_sgd': BaseRRegressor(learner=SGDRegressor())
}
##ElasticNet()

predictions = get_synthetic_preds(simulate_nuisance_and_easy_treatment,
                                               n=50000,
                                               estimators=estimators)

'''
            outcome_learner (optional): a model to estimate outcomes
            effect_learner (optional): a model to estimate treatment effects. It needs to take `sample_weight` as an
            input argument for `fit()`
'''

print('hi')

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
##sm.OLS(Y,X)

y_stacking = predictions['generated_data']['tau']
pred_copy = copy.deepcopy(predictions)
pred_copy.pop('Actuals')
pred_copy.pop('generated_data')

x_stacking = np.vstack((predictions['learner_xgb'], predictions['learner_lr'], predictions['learner_sgd'])).T

### for ridge regression you need all predictions as ... ?
model = sm.OLS(y_stacking, x_stacking)
model2 = model.fit_regularized(alpha=0.0, L1_wt=1.0, start_params=None, profile_scale=False, refit=False)
model2.params

