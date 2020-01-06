#!/usr/bin/env python

"""
This script compares svm regression with different kernels by plotting the
regression curves on top of data.
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

def make_plot(X, y, y_pred, fig_num):
  plt.figure(fig_num)
  xbin = np.linspace(-.05,2,80)
  ybin = np.linspace(-.2,5,80)
  plt.hist2d(X,y,[xbin,ybin], norm=LogNorm())
  plt.colorbar()
  plt.scatter(X, y_pred, s=2, c='red', alpha=0.5)
  plt.savefig('estimator_overlaid_on_data_{}.png'.format(fig_num))

def fit_with_subdata(scaledown, offset, kernel):
  # retrieve training data and official reco hadronic energy for comparison
  X = root2array('../training_data.root',
                 branches='calehad',
                 selection='mustopz<1275&&isnumucc==1',
                 step=scaledown, start=offset)
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
  # experiment: swap X and y
  ytemp = y
  y = X
  X = ytemp

  # rescale the regressors
  scaler = preprocessing.StandardScaler().fit(X.reshape(-1,1))
  
  # calculate the mean pairwise squared distance between regressors
  Xstd = scaler.transform(X.reshape(-1,1))
  mean_squared_dist = np.mean(pdist(Xstd, 'sqeuclidean'))
  gpar = '{0:.3g}'.format(1./mean_squared_dist)
  
  # save the scaler
  #~ os.system('mkdir -p models/1d')
  #~ joblib.dump(scaler, 'models/1d/hadronic_1d_scaler_step{}offset{}.pkl'.format(scaledown, offset))
  
  # train svm with standardized regressors
  svr = None
  if kernel == 0:
    svr = SVR(kernel='linear', C=1e3, verbose=True)
  elif kernel == 1:
    svr = SVR(kernel='poly', C=1e3, degree=2, verbose=True)
  elif kernel == 2:
    svr = SVR(kernel='rbf', C=1e1, gamma=float(gpar)*1e-2, verbose=True)
  else:
    svr = SVR(kernel='sigmoid', C=10, gamma=0.01, verbose=True, coef0=0.)
  y_pred = svr.fit(scaler.transform(X.reshape(-1,1)), y).predict(scaler.transform(X.reshape(-1,1)))
  #~ y_lin = svr_lin.fit(scaler.transform(X), y).predict(X)
  #~ y_poly = svr_poly.fit(scaler.transform(X), y).predict(X)
  #~ y_rbf = svr_rbf.fit(scaler.transform(X), y).predict(X)
  #~ y_sigmoid = svr_sigmoid.fit(scaler.transform(X), y).predict(X)
  
  # make plots
  #~ make_plot(X, y, y_lin, 1)
  #~ make_plot(X, y, y_poly, 2)
  #~ make_plot(X, y, y_rbf, 3)
  #~ make_plot(X, y, y_sigmoid, 4)
  plt.figure(1)
  #~ xbin = np.linspace(-.05,2,80)
  #~ ybin = np.linspace(-.2,5,80)
  ybin = np.linspace(-.05,2,80)
  xbin = np.linspace(-.2,5,80)
  plt.hist2d(X,y,[xbin,ybin], norm=LogNorm())
  plt.colorbar()
  plt.scatter(X, y_pred, s=2, c='red', alpha=0.5)
  plt.savefig('estimator_overlaid_on_data_{}.pdf'.format(kernel))


if __name__ == '__main__':
  # parse command line arguments
  parser = argparse.ArgumentParser(description='Hadronic energy SVM with different kernels.')
  parser.add_argument('-s','--step',type=int,default='500')
  parser.add_argument('-o','--offset',type=int,default='0')
  parser.add_argument('-k','--kernel',type=int,default='0')
  args = parser.parse_args()
  
  # specified parameters
  scaledown = args.step
  offset = args.offset
  kernel = args.kernel
  
  # fit model with arguments
  fit_with_subdata(scaledown, offset, kernel)
