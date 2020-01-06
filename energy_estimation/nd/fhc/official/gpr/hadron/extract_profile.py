#!/usr/bin/env python

from __future__ import print_function

from ROOT import *
from sklearn.externals import joblib
import numpy as np
import os

# retrieve training data
f = TFile('../../make_profile/hadronccProfile.root')
g = f.Get('1')
X = np.asarray(g.GetX()).reshape(-1,1)
y = np.asarray(g.GetY())
ey = []
for i in range(len(X)):
  ey = np.append(ey, [g.GetErrorY(i)])
# save to file
os.system('mkdir -p profiled_data')
joblib.dump(X, 'profiled_data/hadronic_cale_active.pkl')
joblib.dump(y, 'profiled_data/hadronic_truee_active.pkl')
joblib.dump(ey, 'profiled_data/hadronic_dtruee_active.pkl')
