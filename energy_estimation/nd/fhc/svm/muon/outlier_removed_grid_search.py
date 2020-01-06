#!/usr/bin/env python

from __future__ import print_function

from root_numpy import root2array
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
import numpy as np
import os

# prepare parameters to scan
C_ranges = [
  np.logspace(-3, 3, 7)
]
gamma_ranges = [
  np.array([0.01, 0.1, 1, 2, 4, 8])
]

par_group = 0
scaledown = 50
param_grid = dict(gamma=gamma_ranges[par_group], C=C_ranges[par_group])

# prepare grid
grid = GridSearchCV(SVR(), param_grid, cv=8, scoring='neg_mean_squared_error', n_jobs=-1, verbose=10)

# retrieve training data
X = joblib.load('outlier_removed_data/muon_trklen_active_step{}neighbor50.pkl'.format(scaledown))
y = joblib.load('outlier_removed_data/muon_truee_active_step{}neighbor50.pkl'.format(scaledown))

# rescale the regressors
scaler = preprocessing.StandardScaler().fit(X)

# fit the grid with data
grid.fit(scaler.transform(X), y)

# view the complete results (list of named tuples)
print(grid.grid_scores_)
os.system('mkdir -p performance_figures')
with open('performance_figures/outlier_removed_grid_search_mse_parameter_set_{}_step{}.txt'.format(par_group, scaledown), 'w') as output:
  for line in grid.grid_scores_:
    output.write(str(line)+'\n')
