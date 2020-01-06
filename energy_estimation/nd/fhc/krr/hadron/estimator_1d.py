#!/usr/bin/env python

from __future__ import print_function

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

def fit_with_subdata(apar, gpar, scaledown, offset):
  # retrieve training data and official reco hadronic energy for comparison
  X = root2array('../training_data.root',
                 branches='calehad',
                 selection='mustopz<1275&&isnumucc==1',
                 step=scaledown, start=offset).reshape(-1,1)
  recoemu_official = root2array('../training_data.root', branches='recoemu',
                                selection='mustopz<1275&&isnumucc==1',
                                step=scaledown, start=offset)
  trueenu = root2array('../training_data.root', branches='trueenu',
                       selection='mustopz<1275&&isnumucc==1',
                       step=scaledown, start=offset)
  y = trueenu - recoemu_official
  yoff = root2array('../training_data.root', branches='recoehad',
                    selection='mustopz<1275&&isnumucc==1',
                    step=scaledown, start=offset)
  
  # rescale the regressors
  scaler = preprocessing.StandardScaler().fit(X)
  
  # calculate the mean pairwise squared distance between regressors
  Xstd = scaler.transform(X)
  mean_squared_dist = np.mean(pdist(Xstd, 'sqeuclidean'))
  gpar = '{0:.3g}'.format(1./mean_squared_dist)
  
  # save the scaler
  os.system('mkdir -p models/1d')
  joblib.dump(scaler, 'models/1d/hadronic_1d_scaler_a{}g{}step{}offset{}.pkl'.format(apar, gpar, scaledown, offset))
  
  # train svm with standardized regressors
  krr = KernelRidge(kernel='rbf', alpha=float(apar), gamma=float(gpar))
  krr.fit(scaler.transform(X), y)
  
  # save the model
  modelpn = 'models/1d/hadronic_1d_energy_estimator_a{}g{}step{}offset{}.pkl'.format(apar, gpar, scaledown, offset)
  joblib.dump(krr, modelpn)
  
  # estimate reco value
  yest = krr.predict(scaler.transform(X))
  rest = (yest-y)/y
  roff = (yoff-y)/y
  
  # save root file
  os.system('mkdir -p output_root_files/1d')
  toutf = TFile('output_root_files/1d/resolution_1d_a{}g{}step{}offset{}.root'.format(apar, gpar, scaledown, offset), 'recreate')
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
  
  # make plots
  figkrr = plt.figure(1)
  xbin = np.linspace(-.05,2,80)
  ybin = np.linspace(-.2,5,80)
  plt.hist2d(X.flatten(),y,[xbin,ybin], norm=LogNorm())
  plt.colorbar()
  plt.scatter(X.flatten(), yest, s=2, c='red', alpha=0.5)
  plt.xlabel('visible hadronic energy (GeV)')
  plt.ylabel('true hadronic energy (GeV)')
  plt.title(r'KRR $\alpha = {}, \gamma = {}$'.format(apar, gpar))
  figprod4 = plt.figure(2)
  plt.hist2d(X.flatten(),y,[xbin,ybin], norm=LogNorm())
  plt.colorbar()
  plt.scatter(X.flatten(), yoff, s=2, c='red', alpha=0.5)
  plt.xlabel('visible hadronic energy (GeV)')
  plt.ylabel('true hadronic energy (GeV)')
  plt.title('prod4')
  figsbs = plt.figure(figsize=(12,5))
  plt.subplot(121)
  plt.hist2d(X.flatten(),y,[xbin,ybin], norm=LogNorm())
  plt.colorbar()
  plt.scatter(X.flatten(), yoff, s=2, c='red', alpha=0.5)
  plt.xlabel('visible hadronic energy (GeV)')
  plt.ylabel('true hadronic energy (GeV)')
  plt.title('prod4')
  plt.subplot(122)
  plt.hist2d(X.flatten(),y,[xbin,ybin], norm=LogNorm())
  plt.colorbar()
  plt.scatter(X.flatten(), yest, s=2, c='red', alpha=0.5)
  plt.xlabel('visible hadronic energy (GeV)')
  plt.ylabel('true hadronic energy (GeV)')
  plt.title(r'KRR $\alpha = {}, \gamma = {}$'.format(apar, gpar))
  
  # save plots
  os.system('mkdir -p plots/1d')
  figkrr.savefig('plots/1d/estimator_overlaid_on_data_a{}g{}step{}offset{}.pdf'.format(apar, gpar, scaledown, offset))
  figprod4.savefig('plots/1d/estimator_overlaid_on_data_prod4_step{}offset{}.pdf'.format(scaledown, offset))
  figsbs.savefig('plots/1d/prod4_krr_side_by_side_a{}g{}step{}offset{}.pdf'.format(apar, gpar, scaledown, offset))
  
  # print out the statistics
  os.system('mkdir -p performance_figures/1d')
  with open('performance_figures/1d/1d_a{}g{}step{}offset{}.txt'.format(apar, gpar, scaledown, offset), 'w') as outf:
    outf.write(str(np.mean(rest))+'\n')
    outf.write(str(tstd(rest))+'\n')
    outf.write(str(skew(rest))+'\n')
    outf.write(str(kurtosis(rest))+'\n')


if __name__ == '__main__':
  # parse command line arguments
  parser = argparse.ArgumentParser(description='Train a hadronic energy KRR with specified parameters!')
  parser.add_argument('-a','--alpha',type=str,default='0.01')
  parser.add_argument('-g','--gamma',type=str,default='0.01')
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
