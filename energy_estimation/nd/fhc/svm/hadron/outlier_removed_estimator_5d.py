#!/usr/bin/env python

from __future__ import print_function

from array import array
from ROOT import *
from root_numpy import root2array
from scipy.stats import tstd, skew, kurtosis
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

# retrieve outlier removed training data
X = joblib.load('outlier_removed_data/hadron_regressors_5d_step{}neighbor50.pkl'.format(scaledown))
y = joblib.load('outlier_removed_data/hadron_target_5d_step{}neighbor50.pkl'.format(scaledown))

# rescale the regressors
scaler = preprocessing.StandardScaler().fit(X)

# train svm with standardized regressors
svr_rbf = SVR(kernel='rbf', C=float(Cpar), gamma=float(gpar), verbose=True)
svr_rbf.fit(scaler.transform(X), y)

# save the model
os.system('mkdir -p models')
modelpn = 'models/outlier_removed_5d_estimator_c{}g{}step{}.pkl'.format(Cpar, gpar, scaledown)
joblib.dump(svr_rbf, modelpn)

# retrieve test data
Xtest = root2array('../no_truecc_cut_stride2_offset1.root',
                   branches=['calehad', 'cvnpi0', 'cvnchargedpion', 'cvnneutron', 'cvnproton'],
                   selection='mustopz<1275&&isnumucc==1',
                   step=scaledown)
Xtest = Xtest.view(np.float32).reshape(Xtest.shape + (-1,))
recoemu_official = root2array('../no_truecc_cut_stride2_offset1.root', branches='recoemu',
                              selection='mustopz<1275&&isnumucc==1',
                              step=scaledown)
trueenu = root2array('../no_truecc_cut_stride2_offset1.root', branches='trueenu',
                     selection='mustopz<1275&&isnumucc==1',
                     step=scaledown)
ytest = trueenu - recoemu_official
yofftest = root2array('../no_truecc_cut_stride2_offset1.root', branches='recoehad',
                      selection='mustopz<1275&&isnumucc==1',
                      step=scaledown)

# estimate resolution on test data
yest = svr_rbf.predict(scaler.transform(Xtest))
rest = (yest-ytest)/ytest
roff = (yofftest-ytest)/ytest

# save root file
os.system('mkdir -p output_root_files')
toutf = TFile('output_root_files/outlier_removed_resolution_5d_c{}g{}step{}.root'.format(Cpar, gpar, scaledown), 'recreate')
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
  offehad[0] = yofftest[i]
  trueehad[0] = ytest[i]
  tr.Fill()
tr.Write()
toutf.Close()

# print out the statistics
os.system('mkdir -p performance_figures')
with open('performance_figures/outlier_removed_5d_c{}g{}step{}.txt'.format(Cpar, gpar, scaledown), 'w') as outf:
  outf.write(str(np.mean(rest))+'\n')
  outf.write(str(tstd(rest))+'\n')
  outf.write(str(skew(rest))+'\n')
  outf.write(str(kurtosis(rest))+'\n')
