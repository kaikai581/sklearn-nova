#!/usr/bin/env python

from __future__ import print_function
from array import array
from matplotlib.colors import LogNorm
from ROOT import *
from root_numpy import root2array
from sklearn.externals import joblib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def make_regressor(pi0, pipm, n, p):
  res = X.reshape(-1,1)
  res = np.insert(res,1,pi0,axis=1)
  res = np.insert(res,2,pipm,axis=1)
  res = np.insert(res,3,n,axis=1)
  res = np.insert(res,4,p,axis=1)
  return res

# parse command line arguments
parser = argparse.ArgumentParser(description='Train a hadronic energy KRR with specified parameters!')
parser.add_argument('-a','--alpha',type=str,default='0.01')
parser.add_argument('-g','--gamma',type=str,default='None')
parser.add_argument('-s','--step',type=int,default='100')
parser.add_argument('-o','--offset',type=int,default='0')
args = parser.parse_args()

# specified parameters
apar = args.alpha
gpar = args.gamma
scaledown = args.step
offset = args.offset

# check if models are already there
modelpath = 'models/5d/sample_weighted_hadronic_5d_energy_estimator_a0.01gNonestep100offset0.pkl'
scalerpath = 'models/5d/sample_weighted_hadronic_5d_scaler_a0.01gNonestep100offset0.pkl'

# load estimator and scaler
krr = joblib.load(modelpath)
scaler = joblib.load(scalerpath)

# retrieve training data and official reco hadronic energy for comparison
cut = 'mustopz<1275&&isnumucc==1'
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
wei = root2array('../training_data.root', branches='weight',
                 selection=cut,
                 step=scaledown, start=offset)

# make plots
figsbs = plt.figure(figsize=(12,5))
plt.subplot(121)
xbin = np.linspace(-.05,2,80)
ybin = np.linspace(-.2,5,80)
plt.hist2d(X.flatten(),y,[xbin,ybin], norm=LogNorm(), weights=wei)
plt.colorbar()
plt.scatter(X.flatten(), yoff, s=2, c='red', alpha=0.5)
plt.xlabel('visible hadronic energy (GeV)')
plt.ylabel('true hadronic energy (GeV)')
plt.title('prod4')
plt.subplot(122)
plt.hist2d(X.flatten(),y,[xbin,ybin], norm=LogNorm(), weights=wei)
plt.colorbar()
y1 = krr.predict(scaler.transform(make_regressor(0.09951,0.4111,0.3446,0.732)))
y2 = krr.predict(scaler.transform(make_regressor(0.09951,0.4111,0.3446,0.732)))
y3 = krr.predict(scaler.transform(make_regressor(0.09951,0.4111,0.3446,0.732)))
plt.scatter(X.flatten(), y1, s=2, c='red', alpha=0.5, label=r'$\pi^0=0.09951\pi^\pm=0.411n=0.3446p=0.732$')
plt.scatter(X.flatten(), y2, s=2, c='cyan', alpha=0.5, label=r'$\pi^0=\pi^\pm=n=p=0.7$')
plt.scatter(X.flatten(), y3, s=2, c='blue', alpha=0.5, label=r'$\pi^0=\pi^\pm=0.3,n=p=0.7$')
plt.xlabel('visible hadronic energy (GeV)')
plt.ylabel('true hadronic energy (GeV)')
plt.title(r'KRR $\alpha = {}, \gamma = {}$'.format(apar, gpar))
plt.legend()
#~ plt.show()

os.system('mkdir -p plots/5d')
plt.savefig('plots/5d/prod4_krr_5d_side_by_side_a{}g{}step{}offset{}.pdf'.format(apar, gpar, scaledown, offset))
