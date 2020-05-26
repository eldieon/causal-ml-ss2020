import numpy as np
import sklearn.datasets

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
'''


