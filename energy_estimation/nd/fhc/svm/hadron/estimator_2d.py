#!/usr/bin/env python

from __future__ import print_function

from array import array
from ROOT import *
from root_numpy import root2array
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.svm import SVR
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

# parse command line arguments
parser = argparse.ArgumentParser(description='Train a hadronic energy SVM with specified parameters!')
parser.add_argument('-c','--cpar',type=str,default='0.01')
parser.add_argument('-g','--gamma',type=str,default='0.01')
parser.add_argument('-s','--step',type=int,default='50')
args = parser.parse_args()

# specified parameters
Cpar = args.cpar
gpar = args.gamma
scaledown = args.step

# retrieve training data and official reco hadronic energy for comparison
X = root2array('../grid_output_stride2_offset0.root',
               branches=['calehad', 'cvnchargedpion'],
               selection='mustopz<1275',
               step=scaledown)
X = X.view(np.float32).reshape(X.shape + (-1,))
recoemu_official = root2array('../grid_output_stride2_offset0.root', branches='recoemu',
                              selection='mustopz<1275',
                              step=scaledown)
trueenu = root2array('../grid_output_stride2_offset0.root', branches='trueenu',
                     selection='mustopz<1275',
                     step=scaledown)
y = trueenu - recoemu_official
yoff = root2array('../grid_output_stride2_offset0.root', branches='recoehad',
                  selection='mustopz<1275',
                  step=scaledown)

# rescale the regressors
scaler = preprocessing.StandardScaler().fit(X)

# train svm with standardized regressors
svr_rbf = SVR(kernel='rbf', C=float(Cpar), gamma=float(gpar), verbose=True)
svr_rbf.fit(scaler.transform(X), y)

# save the model
os.system('mkdir -p models')
modelpn = 'models/hadronic_2d_energy_estimator_active_c{}g{}step{}.pkl'.format(Cpar, gpar, scaledown)
joblib.dump(svr_rbf, modelpn)

# estimate reco value
yest = svr_rbf.predict(scaler.transform(X))
rest = (yest-y)/y
roff = (yoff-y)/y

# save root file
os.system('mkdir -p output_root_files')
toutf = TFile('output_root_files/resolution_2d_c{}g{}step{}.root'.format(Cpar, gpar, scaledown), 'recreate')
tr = TTree( 'tr', 'resolution tree' )
r1 = array( 'f', [ 0. ] )
r2 = array( 'f', [ 0. ] )
svmehad = array( 'f', [ 0. ] )
offehad = array( 'f', [ 0. ] )
trueehad = array( 'f', [ 0. ] )
tr.Branch( 'rest', r1, 'rest/F' )
tr.Branch( 'roff', r2, 'roff/F' )
tr.Branch('svmehad', svmehad, 'svmehad/F')
tr.Branch('offehad', offehad, 'offehad/F')
tr.Branch('trueehad', trueehad, 'trueehad/F')
for i in range(len(rest)):
  r1[0] = rest[i]
  r2[0] = roff[i]
  svmehad[0] = yest[i]
  offehad[0] = yoff[i]
  trueehad[0] = y[i]
  tr.Fill()
tr.Write()
toutf.Close()
