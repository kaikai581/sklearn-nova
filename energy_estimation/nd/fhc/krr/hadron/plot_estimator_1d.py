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

# parse command line arguments
parser = argparse.ArgumentParser(description='Train a hadronic energy KRR with specified parameters!')
parser.add_argument('-a','--alpha',type=str,default='0.01')
parser.add_argument('-g','--gamma',type=str,default='0.01')
parser.add_argument('-s','--step',type=int,default='20')
parser.add_argument('-o','--offset',type=int,default='0')
args = parser.parse_args()

# specified parameters
apar = args.alpha
gpar = args.gamma
scaledown = args.step
offset = args.offset

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

# plot 2d histogram
#~ f = TFile('../training_data.root')
#~ tr = f.Get('tr')
#~ h2d = TH2F('h2d','',50,-.05,2,50,-.2,5)
#~ tr.Project('h2d','trueenu-recoemu:calehad','mustopz<1275&&isnumucc==1')
#~ h2d.Draw('colz')
#~ h2d.SetStats(0)
#~ gPad.SetLogz(1)
xbin = np.linspace(-.05,2,80)
ybin = np.linspace(-.2,5,80)
plt.hist2d(X,y,[xbin,ybin], norm=LogNorm())
plt.colorbar()

# plot the krr estimator
#~ n = len(X)
#~ xx, yy = array( 'd' ), array( 'd' )
#~ krr = joblib.load('models/1d/hadronic_1d_energy_estimator_a0.01g0.5step200offset0.pkl')
#~ scaler = joblib.load('models/1d/hadronic_1d_scaler_a0.01g0.5step200offset0.pkl')
#~ for i in range( n ):
  #~ xx.append(X[i])
  #~ yy.append( krr.predict(scaler.transform(X[i])) )
#~ gr = TGraph( n, xx, yy )
#~ gr.Draw('P')
krr = joblib.load('models/1d/hadronic_1d_energy_estimator_a0.01g0.5step200offset0.pkl')
scaler = joblib.load('models/1d/hadronic_1d_scaler_a0.01g0.5step200offset0.pkl')
plt.scatter(X, krr.predict(scaler.transform(X.reshape(-1,1))), s=2, c='red', alpha=0.5)
plt.savefig('estimator_overlaid_on_data.png')

#~ raw_input('press enter')
