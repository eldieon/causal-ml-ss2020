from causalml.dataset import *
from causalml.inference.meta import BaseRRegressor
from causalml.inference.tree import CausalTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

learner_xgb = BaseRRegressor(learner=XGBRegressor())
learner_lr = BaseRRegressor(learner=LinearRegression())
learner_dtr = BaseRRegressor(learner=DecisionTreeRegressor())

###would be cool to find some other working learners, and to start messing with the params of each!!!!
learner_knr = BaseRRegressor(learner=KNeighborsRegressor())
learner_svr = BaseRRegressor(learner=SVR())
learner_ctr = BaseRRegressor(learner=CausalTreeRegressor())


estimators = {'learner_xgb': BaseRRegressor(learner=XGBRegressor()),
              'learner_lr': BaseRRegressor(learner=LinearRegression()),
              'learner_dtr': BaseRRegressor(learner=DecisionTreeRegressor())}


predictions = get_synthetic_preds(simulate_nuisance_and_easy_treatment,
                                               n=50000,
                                               estimators=estimators)



## predictions not great, mostly quite bad.
## need to feed in a propensity score model for this!!!
## maybe works better on different data sets.

### NEXT: HOW TO PLOT IT,
### HOW TO EVALUATE IT.

'''
            outcome_learner (optional): a model to estimate outcomes
            effect_learner (optional): a model to estimate treatment effects. It needs to take `sample_weight` as an
            input argument for `fit()`
'''