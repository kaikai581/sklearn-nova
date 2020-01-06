#!/usr/bin/env python

# This estimator treat muons ending in active region or muon catcher in one
# shot.

from __future__ import print_function

from root_numpy import root2array
from sklearn.externals import joblib
from sklearn.svm import SVR
import numpy as np
import os

scaledown = 30
X = root2array('../grid_output_stride5_offset0.root',
               branches=['recotrklenact','recotrklencat'],
               step=scaledown)
X = X.view(np.float32).reshape(X.shape + (-1,))
y = root2array('../grid_output_stride5_offset0.root', branches='trueemu',
               step=scaledown)

svr_rbf = SVR(kernel='rbf', C=0.01, gamma=0.001, verbose=True)
svr_rbf.fit(X, y)

# save the model
os.system('mkdir -p models')
joblib.dump(svr_rbf, 'models/muon_energy_estimator_combined.pkl')
