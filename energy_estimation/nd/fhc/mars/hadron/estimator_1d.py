#!/usr/bin/env python

from __future__ import print_function

from array import array
from pyearth import Earth
from ROOT import *
from root_numpy import root2array
from scipy.spatial.distance import pdist
from scipy.stats import tstd, skew, kurtosis
from sklearn.externals import joblib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def fit_with_subdata(scaledown, offset):
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
  
  # train svm with standardized regressors
  mars = Earth()
  mars.fit(X, y)
  
  # save the model
  os.system('mkdir -p models/1d')
  modelpn = 'models/1d/hadronic_1d_energy_estimator_step{}offset{}.pkl'.format(scaledown, offset)
  joblib.dump(mars, modelpn)
  
  # estimate reco value
  yest = mars.predict(X)
  rest = (yest-y)/y
  roff = (yoff-y)/y
  
  # save root file
  os.system('mkdir -p output_root_files/1d')
  toutf = TFile('output_root_files/1d/resolution_1d_step{}offset{}.root'.format(scaledown, offset), 'recreate')
  tr = TTree( 'tr', 'resolution tree' )
  r1 = array( 'f', [ 0. ] )
  r2 = array( 'f', [ 0. ] )
  marsehad = array( 'f', [ 0. ] )
  offehad = array( 'f', [ 0. ] )
  trueehad = array( 'f', [ 0. ] )
  tr.Branch( 'rest', r1, 'rest/F' )
  tr.Branch( 'roff', r2, 'roff/F' )
  tr.Branch('marsehad', marsehad, 'marsehad/F')
  tr.Branch('offehad', offehad, 'offehad/F')
  tr.Branch('trueehad', trueehad, 'trueehad/F')
  for i in range(len(rest)):
    r1[0] = rest[i]
    r2[0] = roff[i]
    marsehad[0] = yest[i]
    offehad[0] = yoff[i]
    trueehad[0] = y[i]
    tr.Fill()
  tr.Write()
  toutf.Close()
  
  # print out the statistics
  os.system('mkdir -p performance_figures/1d')
  with open('performance_figures/1d/1d_step{}offset{}.txt'.format(scaledown, offset), 'w') as outf:
    outf.write(str(np.mean(rest))+'\n')
    outf.write(str(tstd(rest))+'\n')
    outf.write(str(skew(rest))+'\n')
    outf.write(str(kurtosis(rest))+'\n')


if __name__ == '__main__':
  # parse command line arguments
  parser = argparse.ArgumentParser(description='Train a hadronic energy with MARS model!')
  parser.add_argument('-s','--step',type=int,default='200')
  parser.add_argument('-o','--offset',type=int,default='0')
  args = parser.parse_args()
  
  # specified parameters
  scaledown = args.step
  offset = args.offset

  # fit model with arguments
  fit_with_subdata(scaledown, offset)
