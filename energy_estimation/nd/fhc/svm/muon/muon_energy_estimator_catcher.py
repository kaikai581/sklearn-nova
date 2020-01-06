#!/usr/bin/env python

# Compare with muons contained in the active volume,
# energy regression for muons stopping in the muon catcher uses
# an addition feature, the track length in the muon catcher.

from __future__ import print_function

from root_numpy import root2array
from sklearn.externals import joblib
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import os

scaledown = 20
X = root2array('../grid_output_stride5_offset0.root',
               branches=['recotrklenact','recotrklencat'],
               selection='mustopz>=1275', step=scaledown)
X = X.view(np.float32).reshape(X.shape + (-1,))
y = root2array('../grid_output_stride5_offset0.root', branches='trueemu',
               selection='mustopz>=1275', step=scaledown)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1, verbose=True)
svr_rbf.fit(X, y)

# save the model
os.system('mkdir -p models')
joblib.dump(svr_rbf, 'models/muon_energy_estimator_catcher.pkl')
