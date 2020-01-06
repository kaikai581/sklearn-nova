#!/usr/bin/env python

from __future__ import print_function

from ROOT import *
from sklearn.externals import joblib
import numpy as np
import os

# retrieve training data
f = TFile('../../make_profile/muonProfile.root')
g = f.Get('1')
X = np.asarray(g.GetX())
y = np.asarray(g.GetY())
idx = (X>0.1) & (X<15)
X = X[idx].reshape(-1,1)
y = y[idx]
ey = []
for i in range(len(idx)):
  if idx[i] == True:
    ey = np.append(ey, [g.GetErrorY(i)])

# save to file
os.system('mkdir -p profiled_data')
joblib.dump(X, 'profiled_data/muon_trklen_active.pkl')
joblib.dump(y, 'profiled_data/muon_truee_active.pkl')
joblib.dump(ey, 'profiled_data/muon_dtruee_active.pkl')
