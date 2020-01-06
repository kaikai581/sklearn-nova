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
C_range = np.logspace(-3, 3, 7)
gamma_range = np.logspace(-3, 3, 7)
param_grid = dict(gamma=gamma_range, C=C_range)

# prepare grid
grid = GridSearchCV(SVR(), param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=10)

# retrieve data
scaledown = 30
X = root2array('../grid_output_stride5_offset0.root',
               branches=['recotrklenact','recotrklencat'],
               step=scaledown)
X = X.view(np.float32).reshape(X.shape + (-1,))
y = root2array('../grid_output_stride5_offset0.root', branches='trueemu',
               step=scaledown)

# fit the grid with data
grid.fit(X, y)

# view the complete results (list of named tuples)
print(grid.grid_scores_)
os.system('mkdir -p performance_figures')
with open('performance_figures/grid_search_mae_combined.txt', 'w') as output:
  for line in grid.grid_scores_:
    output.write(str(line)+'\n')
