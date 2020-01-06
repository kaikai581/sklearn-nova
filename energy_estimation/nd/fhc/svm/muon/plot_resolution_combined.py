#!/usr/bin/env python

from __future__ import print_function

from root_numpy import root2array, array2root
from sklearn.externals import joblib
import numpy as np
import os

svr = joblib.load('models/muon_energy_estimator_combined.pkl')

# retrieve test data
scaledown = 30
X = root2array('../grid_output_stride5_offset1.root',
               branches=['recotrklenact','recotrklencat'],
               selection='mustopz<1275',
               step=scaledown)
X = X.view(np.float32).reshape(X.shape + (-1,))
y = root2array('../grid_output_stride5_offset1.root', branches='trueemu',
               selection='mustopz<1275',
               step=scaledown)
yest = svr.predict(X)

res = (yest-y)/y
res = res.view(dtype=[('resolution',float)])
os.system('mkdir -p output_root_files')
array2root(res, 'output_root_files/resolution_combined.root', 'tr')
