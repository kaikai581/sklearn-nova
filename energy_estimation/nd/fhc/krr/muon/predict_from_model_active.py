#!/usr/bin/env python

import argparse
import numpy as np
import os, sys
sys.path.append(os.getcwd())
from fit_model_active import fit_krr

from array import array
from ROOT import *
from root_numpy import root2array
from scipy.stats import tstd, skew, kurtosis
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.kernel_ridge import KernelRidge

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-a','--apar',type=str,default='1.0')
parser.add_argument('-g','--gamma',type=str,default='1.0')
parser.add_argument('-n','--number_of_events',type=int,default='40000')
parser.add_argument('-s','--scaledown',type=int,default='100')
args = parser.parse_args()

# specified parameters
apar = args.apar
nevt = args.number_of_events
gpar = args.gamma
scaledown = args.scaledown

# load model
if not os.path.isfile('models/muon_energy_estimator_active_a{}g{}nevt{}.pkl'.format(apar, gpar, nevt)):
  #~ os.system('time ./fit_model_active.py -a {} -g {} -n {}'.format(apar, gpar, nevt))
  fit_krr(apar, gpar, nevt)
krr = joblib.load('models/muon_energy_estimator_active_a{}g{}nevt{}.pkl'.format(apar, gpar, nevt))
scaler = joblib.load('models/regressor_scaler_active_a{}g{}nevt{}.pkl'.format(apar, gpar, nevt))

# retrieve test data
Xtest = root2array('../../svm/no_truecc_cut_stride2_offset0.root',
                   branches='recotrklenact',
                   selection='mustopz<1275&&isnumucc==1',
                   step=scaledown).reshape(-1,1)
ytest = root2array('../../svm/no_truecc_cut_stride2_offset0.root', branches='trueemu',
                   selection='mustopz<1275&&isnumucc==1',
                   step=scaledown)
yofftest = root2array('../../svm/no_truecc_cut_stride2_offset0.root', branches='recoemu',
                      selection='mustopz<1275&&isnumucc==1',
                      step=scaledown)

# estimate resolution on test data
yest = krr.predict(scaler.transform(Xtest))
rest = (yest-ytest)/ytest
roff = (yofftest-ytest)/ytest

# save root file
os.system('mkdir -p output_root_files')
toutf = TFile('output_root_files/resolution_active_a{}g{}nevt{}step{}.root'.format(apar, gpar, nevt, scaledown), 'recreate')
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

# print out the statistics
os.system('mkdir -p performance_figures')
with open('performance_figures/active_a{}g{}step{}.txt'.format(apar, gpar, scaledown), 'w') as outf:
  outf.write(str(np.mean(rest))+'\n')
  outf.write(str(tstd(rest))+'\n')
  outf.write(str(skew(rest))+'\n')
  outf.write(str(kurtosis(rest))+'\n')
