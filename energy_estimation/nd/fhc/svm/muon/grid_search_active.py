#!/usr/bin/env python

# This script uses the grid search cross validation method to optimize
# C and gamma.

from __future__ import print_function

from root_numpy import root2array
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
import numpy as np
import os


# prepare parameters to scan
C_ranges = [
  np.logspace(-3, 3, 7),
  np.logspace(-4, -4, 1),
  np.logspace(-3, -3, 1),
  np.logspace(-3, -3, 1),
  np.linspace(1e-4, 1e-3, 4),
  np.logspace(-3, -3, 1),
  np.logspace(-4, -1, 4)
]
gamma_ranges = [
  np.logspace(-3, 3, 7),
  np.logspace(-4, 3, 16),
  np.logspace(-4, 3, 16),
  np.logspace(-4, 3, 15),
  np.logspace(-4, 0, 9),
  np.linspace(0, 0.5, 51),
  np.logspace(-4, -1, 4)
]

par_group = 6
param_grid = dict(gamma=gamma_ranges[par_group], C=C_ranges[par_group])

# prepare grid
grid = GridSearchCV(SVR(), param_grid, cv=8, scoring='neg_mean_squared_error', n_jobs=2, verbose=10)

# retrieve data
scaledown = 50
X = root2array('../grid_output_stride2_offset0.root', branches='recotrklenact',
               selection='mustopz<1275', step=scaledown).reshape(-1,1)
y = root2array('../grid_output_stride2_offset0.root', branches='trueemu',
               selection='mustopz<1275', step=scaledown)

# fit the grid with data
grid.fit(X, y)

# view the complete results (list of named tuples)
print(grid.grid_scores_)
os.system('mkdir -p performance_figures')
with open('performance_figures/grid_search_mse_parameter_set_{}_step{}.txt'.format(par_group, scaledown), 'w') as output:
  for line in grid.grid_scores_:
    output.write(str(line)+'\n')
