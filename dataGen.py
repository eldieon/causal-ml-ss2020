import numpy as np
import sklearn
import opossum
from sklearn import datasets

N = 20
k = 3
b = map(lambda x: 1.0/x, range(1, k+1))

sigma = sklearn.datasets.make_spd_matrix(k)
z_fix = np.random.multivariate_normal(np.zeros(k), sigma, size = N)

'''
to here: generated a dataset of covariates with a certian variance-covariance matrix. 
the configureable part of this dataset is the spd matrix;
maybe in other circumstances we would like for the x1 and x2 to have a relationship w/some variance..
how would we do that? or is this the best case scenario where all are totally independent?
'''

'''
now you want to generate a set of outcome varaibles associated with each observation
simulate these outcome variables by first deciding what kind of treatment effect is present => "theta"
can be linear, constant or ...something else with a high amount of confounding on the covariates.

in the R script - theta continuous and linear is sum of var 1 and 2 plus some random normal var. noise.
theta continuous and nonlinear is sine 

Nie and Wager: 
    a.) lots of confounding of the covariates, but clear treatment effect. (EQUIVALENT TO R SCRIPT : theta_s)
    b.) randomized trial ; generate data (skewed perhaps to some covariates?) and see how stacking works compared to single model.. (IN R SCRIPT : just the parameter random assignment.)
    c.) "easy propensity score and difficult baseline" (WHAT IS THE DATA GENERATION PROCESS HERE AS IN R SCRIPT?)
    d.) unrelated treatment and control arms - (ie no way to compare the treated and the untreated? HOW WOULD WE DO THIS EITHER?)
    
'''
###### EXPERIMENT WITH SAMPLE SIZE.
###### TRY DIFFERENT MODELS TO ESTIMATE PROPENSITY SCORE. AS WELL AS
###### how to use sample splitting efficiently, and what is the 3 part method of splitting samples.
###### BENCHMARK

## difficult nuisance components but a clear treatment effect:
## uses the "scaled Friedman function for the baseline main effect:

## b_star(X_i) = sin(pi * X_i1 * X_i2) + 2(X_i3 - 0.5)^2

b_star_A = np.sin(np.pi * z_fix[:,0] * z_fix[:,1]) + 2.0 * (z_fix[:,2] - 0.5)**2 ## justification for 0.5?

# number of observations N and number of covariates k
N = 10000
k = 50
# initilizing class
u = UserInterface(N, k, seed=None, categorical_covariates = None)
# assign treatment and generate treatment effect inside of class object
u.generate_treatment(random_assignment = True,
                     assignment_prob = 0.5,
                     constant_pos = True,
                     constant_neg = False,
                     heterogeneous_pos = False,
                     heterogeneous_neg = False,
                     no_treatment = False,
                     discrete_heterogeneous = False,
                     treatment_option_weights = None,
                     intensity = 5)
# generate output variable y and return all 4 variables
y, X, assignment, treatment = u.output_data(binary=False, x_y_relation = 'partial_nonlinear_simple')