#!/usr/bin/env python

"""
Plot official reco and true ehad histograms with official weights.
Also plot KRR reco ehad overlaid.
"""

from __future__ import print_function
print(__doc__)

from ROOT import *
from sklearn.externals import joblib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def weighted_moments(data,freq):
  totw = freq.sum()
  mean = (data*freq).sum()/totw
  std  = np.sqrt((freq*(data-mean)**2).sum()/totw)
  skew = (freq*((data-mean)/std)**3).sum()/totw
  kurt = (freq*((data-mean)/std)**4).sum()/totw - 3
  
  return mean, std, skew, kurt


# parse command line arguments
parser = argparse.ArgumentParser(description='Train a hadronic energy KRR with specified parameters and weighted sample.')
parser.add_argument('-n','--number_of_files',type=int,default='100')
args = parser.parse_args()

# specified parameters
nfiles = args.number_of_files

y_raw = np.empty(1)
y_off = np.empty(1)
y_krr = np.empty(1)
wei = np.empty(1)
intmode = np.empty(1)
for i in range(nfiles):
  y_raw = np.append(y_raw, joblib.load('partial_output/5d/sample_weighted_hadronic_true_a0.01gNonestep100offset{}.pkl'.format(i)))
  y_off = np.append(y_off, joblib.load('partial_output/5d/sample_weighted_hadronic_official_a0.01gNonestep100offset{}.pkl'.format(i)))
  y_krr = np.append(y_krr, joblib.load('partial_output/5d/sample_weighted_hadronic_krr_a0.01gNonestep100offset{}.pkl'.format(i)))
  wei = np.append(wei, joblib.load('partial_output/5d/sample_weighted_hadronic_weight_a0.01gNonestep100offset{}.pkl'.format(i)))
  intmode = np.append(intmode, joblib.load('partial_output/5d/sample_weighted_intmode_a0.01gNonestep100offset{}.pkl'.format(i)))

y = y_raw[y_raw > 0]
y_off = y_off[y_raw > 0]
y_krr = y_krr[y_raw > 0]
wei = wei[y_raw > 0]
intmode = intmode[y_raw > 0]

os.system('mkdir -p plots/5d')

# plot official resolution by mode
figres = plt.figure(2)
res_off = (y_off-y)/y
res_krr = (y_krr-y)/y
modes = [0,1,2,3,10]
colors = ['red','blue','green','yellow','purple']
labels = ['QE','Res','DIS','Coh','MEC']
for i in range(len(modes)):
  n, bins, patches = plt.hist(res_off[intmode == modes[i]], bins=np.linspace(-2,2,200), linewidth=2, histtype='step', weights=wei[intmode == modes[i]], color=colors[i], label=labels[i])
plt.xlim(-1,1)
plt.legend()
plt.title('prod4 hadronic energy resolution')
plt.xlabel('(reco-true)/true')
plt.axvline(x=0., color='green', linestyle='--')
figres.savefig('plots/5d/off_ehad_resolution_by_mode.pdf')

# plot krr resolution by mode
figres = plt.figure(3)
for i in range(len(modes)):
  n, bins, patches = plt.hist(res_krr[intmode == modes[i]], bins=np.linspace(-2,2,200), linewidth=2, histtype='step', weights=wei[intmode == modes[i]], color=colors[i], label=labels[i])
plt.xlim(-1,1)
plt.legend()
plt.title('KRR hadronic energy resolution')
plt.xlabel('(reco-true)/true')
plt.axvline(x=0., color='green', linestyle='--')
figres.savefig('plots/5d/krr_ehad_resolution_by_mode.pdf')

