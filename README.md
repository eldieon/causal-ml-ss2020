This repo contains a jupyter notebook and the accompanying python helper functions for the module _Causal Machine Learning_ for HU Summer term 2020. 

We compare the effectiveness of various methods of stacking on different types of data comprised of exogenous variables, and a treatment effect.
This repository builds largely on the work done by the causalml team at uber : https://causalml.readthedocs.io


The helper functions are divided into three major parts:

- **data_gen.py** : includes functions for generating and visualising the synthetic data as provided by the causalml library for the study of treatment effects
- **simple_model.py** : functions used to fit a simple machine learning model from synthetic data, and to visualise predictions made by a  model or group of models.
- **stacking_helpers.py** : functions used for stacking predictions resulting from the simple_model module, as well as visualization, and evaluation functions

The notebook and helper functions are written with python3

NB: a strange, perhaps jupyter cache related bug of this notebook, is that one of the last plots generated in the notebook often appears first in the notebook when viewed in the browser. 
Re-running the cell will remove this plot. 

the github repository for this notebook and related scripts can be found here: https://github.com/eldieon/causal-ml-ss2020