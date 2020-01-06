#!/usr/bin/env python

# SVM regression for hadronic energy. The regressor will be the hadronic
# calibrated energy and other features that indicate the "hardness"
# of the interaction.

from __future__ import print_function

from ROOT import *
from root_numpy import root2array
from sklearn.externals import joblib
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os

# retrieve data, scaled down by factor of 20
scaledown = 20
Xhad = root2array('../grid_output_stride5_offset0.root',
                  branches=['calehad', 'cvnchargedpion'],
                  selection='mustopz<1275',
                  step=scaledown)
Xmu = root2array('../grid_output_stride5_offset0.root',
                 branches='recotrklenact',
                 selection='mustopz<1275',
                 step=scaledown).reshape(-1,1)
ynu = root2array('../grid_output_stride5_offset0.root', branches='trueenu',
               selection='mustopz<1275', step=scaledown)
svr_mu = joblib.load('../muon/models/muon_energy_estimator_active.pkl')
recoemu = svr_mu.predict(Xmu)
yhad = ynu - recoemu

hfit = TH2F('hfit','',100,0,2,100,0,5)
for i in range(len(Xhad)):
  hfit.Fill(Xhad['calehad'][i], yhad[i])
hfit.Draw('colz')

raw_input('Press enter to continue.')  
