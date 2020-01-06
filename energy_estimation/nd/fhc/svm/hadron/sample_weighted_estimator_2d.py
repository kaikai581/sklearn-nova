#!/usr/bin/env python

"""
This script weights each point with inverse square root of its true hadronic
energy when fitting. This is equivalent to modifying the SVM loss function.
Two predictors are used, namely calibrated hadronic energy, and one swappable
varisble.
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
from sklearn.svm import SVR
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def fit_with_subdata(var, cpar, gpar, scaledown, offset):
  # retrieve training data and official reco hadronic energy for comparison
  cut = 'mustopz<1275&&isnumucc==1'
  X = root2array('../training_data.root',
                 branches=['calehad', var],
                 selection=cut,
                 step=scaledown, start=offset)
  X = X.view(np.float32).reshape(X.shape + (-1,))
  recoemu_official = root2array('../training_data.root', branches='recoemu',
                                selection=cut,
                                step=scaledown, start=offset)
  trueenu = root2array('../training_data.root', branches='trueenu',
                       selection=cut,
                       step=scaledown, start=offset)
  y = trueenu - recoemu_official
  yoff = root2array('../training_data.root', branches='recoehad',
                    selection=cut,
                    step=scaledown, start=offset)

  # rescale the regressors
  scaler = preprocessing.StandardScaler().fit(X)
  
  # calculate the mean pairwise squared distance between regressors
  Xstd = scaler.transform(X)
  if gpar == 'auto':
    mean_squared_dist = np.mean(pdist(Xstd, 'sqeuclidean'))
    gpar = '{0:.3g}'.format(1./mean_squared_dist)
  
  # save the scaler
  os.system('mkdir -p models/2d')
  joblib.dump(scaler, 'models/2d/sample_weighted_hadronic_scaler_{}_c{}g{}step{}offset{}.pkl'.format(var, cpar, gpar, scaledown, offset))
  
  # make an array for sample weights
  swei = np.copy(y)
  #~ swei[y != 0] = 1./np.sqrt(np.abs(swei[y != 0]))
  swei[y != 0] = 1./np.abs(swei[y != 0])
  swei[y == 0.] = 1.
  
  # train svm with standardized regressors
  svr = SVR(kernel='rbf', C=float(cpar), gamma=float(gpar), verbose=True)
  y_pred = svr.fit(Xstd, y, swei).predict(Xstd)
  
  # save the model
  joblib.dump(svr, 'models/2d/sample_weighted_hadronic_energy_estimator_{}_c{}g{}step{}offset{}.pkl'.format(var, cpar, gpar, scaledown, offset))
  
  # make plots
  #~ plt.figure(1)
  #~ xbin = np.linspace(-.05,2,80)
  #~ ybin = np.linspace(-.2,5,80)
  #~ plt.hist2d(X,y,[xbin,ybin], norm=LogNorm())
  #~ plt.colorbar()
  #~ plt.scatter(X, y_pred, s=2, c='red', alpha=0.5)
  
  # save plots
  #~ os.system('mkdir -p plots/2d')
  #~ plt.savefig('plots/2d/sample_weighted_estimator_overlaid_on_data_{}_c{}g{}step{}offset{}.pdf'.format(var, cpar, gpar, scaledown, offset))
  
  # estimate various reco values
  yest = y_pred
  rest = (yest-y)/y
  roff = (yoff-y)/y
  
  # save root file
  os.system('mkdir -p output_root_files/2d')
  toutf = TFile('output_root_files/2d/sample_weighted_resolution_{}_c{}g{}step{}offset{}.root'.format(var, cpar, gpar, scaledown, offset), 'recreate')
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


if __name__ == '__main__':
  # list of second variable
  varlist = ['cvnpi0', 'cvnchargedpion', 'cvnneutron', 'cvnproton', 'npng']
  
  # parse command line arguments
  parser = argparse.ArgumentParser(description='Hadronic energy SVM with sample weights.')
  parser.add_argument('-c','--cpar',type=str,default='100')
  parser.add_argument('-g','--gpar',type=str,default='auto')
  parser.add_argument('-s','--step',type=int,default='500')
  parser.add_argument('-o','--offset',type=int,default='0')
  parser.add_argument('-v','--variable',type=int,default='0')
  args = parser.parse_args()
  
  # specified parameters
  cpar = args.cpar
  gpar = args.gpar
  scaledown = args.step
  offset = args.offset
  var = varlist[args.variable]
  
  # fit model with arguments
  fit_with_subdata(var, cpar, gpar, scaledown, offset)
