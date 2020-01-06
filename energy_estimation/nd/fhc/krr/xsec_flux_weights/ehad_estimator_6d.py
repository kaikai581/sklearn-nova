#!/usr/bin/env python

"""
This script weights each point with inverse square root of its true hadronic
energy as well as the 2018 official cross section and flux weights when fitting.
This is equivalent to modifying the KRR loss function.
Six predictors are used, namely, calibrated hadronic energy, number of prongs,
and CVN final state paricle scores for proton, neutron, pi0, and charged pi.
"""

from __future__ import print_function
print(__doc__)

from array import array
from matplotlib.colors import LogNorm
from ROOT import *
from root_numpy import root2array
from scipy.spatial.distance import pdist
from scipy.stats import tstd, skew, kurtosis
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.kernel_ridge import KernelRidge
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def fit_with_subdata(apar, gpar_opt, scaledown, offset):
  # retrieve training data and official reco hadronic energy for comparison
  cut = 'mustopz<1275&&isnumucc==1'
  X = root2array('../training_data.root',
                 branches=['calehad', 'cvnpi0', 'cvnchargedpion', 'cvnneutron', 'cvnproton', 'npng'],
                 selection=cut,
                 step=scaledown, start=offset)
  X = X.view(np.float32).reshape(X.shape + (-1,))
  recoemu_official = root2array('../training_data.root', branches='recoemu',
                                selection='mustopz<1275&&isnumucc==1',
                                step=scaledown, start=offset)
  trueenu = root2array('../training_data.root', branches='trueenu',
                       selection=cut,
                       step=scaledown, start=offset)
  y = trueenu - recoemu_official
  yoff = root2array('../training_data.root', branches='recoehad',
                    selection=cut,
                    step=scaledown, start=offset)
  offwei = root2array('../training_data.root', branches='weight',
                      selection=cut,
                      step=scaledown, start=offset)
  
  # rescale the regressors
  scaler = preprocessing.StandardScaler().fit(X)
  
  # calculate the mean pairwise squared distance between regressors
  Xstd = scaler.transform(X)
  mean_squared_dist = np.mean(pdist(Xstd, 'sqeuclidean'))
  gpar = None
  if gpar_opt == 'default': gpar = None
  elif gpar_opt == 'msd': gpar = float('{0:.3g}'.format(1./mean_squared_dist))
  else: gpar = float(gpar_opt)
  
  # save the scaler
  os.system('mkdir -p models/6d')
  joblib.dump(scaler, 'models/6d/sample_weighted_hadronic_6d_scaler_a{}g{}step{}offset{}.pkl'.format(apar, gpar, scaledown, offset))
  
  # train krr with standardized regressors
  # Since the metric we use to judge if things work well is the
  # "relative" resolution, while the loss function used is just the mean square
  # loss, we have to weight the sample.
  # make an array for sample weights
  swei = np.copy(y)
  swei[y != 0] = 1./np.sqrt(np.abs(swei[y != 0]))
  swei[y == 0.] = 1.
  krr = KernelRidge(kernel='rbf', alpha=float(apar), gamma=gpar)
  krr.fit(scaler.transform(X), y, swei*offwei)
  
  # save the model
  modelpn = 'models/6d/sample_weighted_hadronic_6d_energy_estimator_a{}g{}step{}offset{}.pkl'.format(apar, gpar, scaledown, offset)
  joblib.dump(krr, modelpn)
  
  # estimate reco value
  yest = krr.predict(scaler.transform(X))
  rest = (yest-y)/y
  roff = (yoff-y)/y
  
  # save root file
  os.system('mkdir -p output_root_files/6d')
  toutf = TFile('output_root_files/6d/sample_weighted_resolution_6d_a{}g{}step{}offset{}.root'.format(apar, gpar, scaledown, offset), 'recreate')
  tr = TTree( 'tr', 'resolution tree' )
  r1 = array( 'f', [ 0. ] )
  r2 = array( 'f', [ 0. ] )
  krrehad = array( 'f', [ 0. ] )
  offehad = array( 'f', [ 0. ] )
  trueehad = array( 'f', [ 0. ] )
  tr.Branch( 'rest', r1, 'rest/F' )
  tr.Branch( 'roff', r2, 'roff/F' )
  tr.Branch('krrehad', krrehad, 'krrehad/F')
  tr.Branch('offehad', offehad, 'offehad/F')
  tr.Branch('trueehad', trueehad, 'trueehad/F')
  for i in range(len(rest)):
    r1[0] = rest[i]
    r2[0] = roff[i]
    krrehad[0] = yest[i]
    offehad[0] = yoff[i]
    trueehad[0] = y[i]
    tr.Fill()
  tr.Write()
  toutf.Close()
  
  # print out the statistics
  os.system('mkdir -p performance_figures/6d')
  with open('performance_figures/6d/sample_weighted_6d_a{}g{}step{}offset{}.txt'.format(apar, gpar, scaledown, offset), 'w') as outf:
    outf.write(str(np.mean(rest))+'\n')
    outf.write(str(tstd(rest))+'\n')
    outf.write(str(skew(rest))+'\n')
    outf.write(str(kurtosis(rest))+'\n')


if __name__ == '__main__':
  
  # parse command line arguments
  parser = argparse.ArgumentParser(description='Train a hadronic energy KRR with specified parameters and weighted sample.')
  parser.add_argument('-a','--alpha',type=str,default='0.01')
  parser.add_argument('-g','--gamma',type=str,default='default')
  parser.add_argument('-s','--step',type=int,default='200')
  parser.add_argument('-o','--offset',type=int,default='0')
  args = parser.parse_args()
  
  # specified parameters
  apar = args.alpha
  gpar = args.gamma
  scaledown = args.step
  offset = args.offset

  # fit model with arguments
  fit_with_subdata(apar, gpar, scaledown, offset)
