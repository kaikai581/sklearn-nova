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
import os

# parse command line arguments
parser = argparse.ArgumentParser(description='Train a muon energy SVM with specified parameters and with outlier removed data.')
parser.add_argument('-c','--cpar',type=str,default='0.01')
parser.add_argument('-g','--gamma',type=str,default='1.0')
parser.add_argument('-s','--step',type=int,default='50')
args = parser.parse_args()

# specified parameters
Cpar = args.cpar
gpar = args.gamma
scaledown = args.step

# retrieve training data
X = joblib.load('outlier_removed_data/muon_trklen_active_step{}neighbor50.pkl'.format(scaledown))
y = joblib.load('outlier_removed_data/muon_truee_active_step{}neighbor50.pkl'.format(scaledown))

# rescale the regressors
scaler = preprocessing.StandardScaler().fit(X)

# fit the model
svr_rbf = SVR(kernel='rbf', C=float(Cpar), gamma=float(gpar), verbose=True)
svr_rbf.fit(scaler.transform(X), y)

# save the model
os.system('mkdir -p models')
modelpn = 'models/outlier_removed_estimator_active_c{}g{}step{}.pkl'.format(Cpar, gpar, scaledown)
joblib.dump(svr_rbf, modelpn)

# retrieve test data
Xtest = root2array('../no_truecc_cut_stride2_offset1.root',
                   branches='recotrklenact',
                   selection='mustopz<1275&&isnumucc==1',
                   step=scaledown).reshape(-1,1)
ytest = root2array('../no_truecc_cut_stride2_offset1.root', branches='trueemu',
                   selection='mustopz<1275&&isnumucc==1',
                   step=scaledown)
yofftest = root2array('../no_truecc_cut_stride2_offset1.root', branches='recoemu',
                      selection='mustopz<1275&&isnumucc==1',
                      step=scaledown)

# estimate resolution on test data
yest = svr_rbf.predict(scaler.transform(Xtest))
rest = (yest-ytest)/ytest
roff = (yofftest-ytest)/ytest

# save root file
os.system('mkdir -p output_root_files')
toutf = TFile('output_root_files/outlier_removed_resolution_active_c{}g{}step{}.root'.format(Cpar, gpar, scaledown), 'recreate')
tr = TTree( 'tr', 'resolution tree' )
r1 = array( 'f', [ 0. ] )
r2 = array( 'f', [ 0. ] )
svmemu = array( 'f', [ 0. ] )
offemu = array( 'f', [ 0. ] )
trueemu = array( 'f', [ 0. ] )
tr.Branch( 'rest', r1, 'rest/F' )
tr.Branch( 'roff', r2, 'roff/F' )
tr.Branch('svmemu', svmemu, 'svmemu/F')
tr.Branch('offemu', offemu, 'offemu/F')
tr.Branch('trueemu', trueemu, 'trueemu/F')
for i in range(len(rest)):
  r1[0] = rest[i]
  r2[0] = roff[i]
  svmemu[0] = yest[i]
  offemu[0] = yofftest[i]
  trueemu[0] = ytest[i]
  tr.Fill()
tr.Write()
toutf.Close()
