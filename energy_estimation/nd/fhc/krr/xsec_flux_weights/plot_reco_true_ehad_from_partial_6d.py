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


#~ # helper function for calculating statistic moments of distributions
#~ def integrator(f,data,freq):
  #~ diffs = np.roll(data,-1)-data
  #~ return (f(data[:-1])*freq[:-1]*diffs[:-1]).sum()

#~ # function for returning weighted moments
#~ def weighted_moments(data,freq):
  #~ freq_norm = freq/integrator(lambda x:1,data,freq)
  
  #~ exp_x = integrator(lambda x:x,data,freq_norm)
  #~ exp_x2 = integrator(lambda x:x**2,data,freq_norm)
  #~ exp_x4 = integrator(lambda x:x**4,data,freq_norm)
  
  #~ mean = exp_x
  #~ rms  = integrator(lambda x: ((x-exp_x)/std)**2,data,freq_norm)
  #~ kurt = integrator(lambda x: ((x-exp_x)/std)**4,data,freq_norm)
  #~ skew = integrator(lambda x: ((x-exp_x)/std)**3,data,freq_norm)
  
  #~ return mean, rms, skew, kurt

def weighted_moments(data,freq):
  totw = freq.sum()
  mean = (data*freq).sum()/totw
  std  = np.sqrt((freq*(data-mean)**2).sum()/totw)
  skew = (freq*((data-mean)/std)**3).sum()/totw
  kurt = (freq*((data-mean)/std)**4).sum()/totw - 3
  
  return mean, std, skew, kurt


# parse command line arguments
parser = argparse.ArgumentParser(description='Train a hadronic energy KRR with specified parameters and weighted sample.')
parser.add_argument('-n','--number_of_files',type=int,default='10')
args = parser.parse_args()

# specified parameters
nfiles = args.number_of_files

y = np.empty(1)
y_off = np.empty(1)
y_krr = np.empty(1)
wei = np.empty(1)
for i in range(nfiles):
  y = np.append(y, joblib.load('partial_output/6d/sample_weighted_hadronic_true_a0.01gNonestep100offset{}.pkl'.format(i)))
  y_off = np.append(y_off, joblib.load('partial_output/6d/sample_weighted_hadronic_official_a0.01gNonestep100offset{}.pkl'.format(i)))
  y_krr = np.append(y_krr, joblib.load('partial_output/6d/sample_weighted_hadronic_krr_a0.01gNonestep100offset{}.pkl'.format(i)))
  wei = np.append(wei, joblib.load('partial_output/6d/sample_weighted_hadronic_weight_a0.01gNonestep100offset{}.pkl'.format(i)))

os.system('mkdir -p plots/6d')
# plot true ehad with reco overlaid
figehad = plt.figure(1)
n, bins, patches = plt.hist(y, bins=np.linspace(-1,10,550), histtype='stepfilled', weights=wei, color='yellow', label='true')
plt.hist(y_off, bins=bins, histtype='step', color='red', linewidth=2, weights=wei, label='prod4')
plt.hist(y_krr, bins=bins, histtype='step', color='blue', linewidth=2, weights=wei, label='shallow learning')
plt.xlim(0,2)
plt.legend()
plt.title('hadronic energy spectra')
plt.xlabel('hadronic energy (GeV)')
figehad.savefig('plots/6d/ehad_spectra.pdf')

# plot resolution
figres = plt.figure(2)
res_off = (y_off-y)/y
res_krr = (y_krr-y)/y
n, bins, patches = plt.hist(res_off, bins=np.linspace(-2,2,200), linewidth=2, histtype='step', weights=wei, color='red', label='prod4')
plt.hist(res_krr, bins=bins, histtype='step', color='blue', linewidth=2, weights=wei, label='shallow learning')
plt.xlim(-1,1)
plt.legend()
plt.title('hadronic energy resolution')
plt.xlabel('(reco-true)/true')
plt.axvline(x=0., color='green', linestyle='--')
figres.savefig('plots/6d/ehad_resolution.pdf')

# plot side by side
figsbs = plt.figure(figsize=(12,5))
plt.subplot(121)
n, bins, patches = plt.hist(y, bins=np.linspace(-1,10,550), histtype='stepfilled', weights=wei, color='yellow', label='true')
plt.hist(y_off, bins=bins, histtype='step', color='red', linewidth=2, weights=wei, label='prod4')
plt.hist(y_krr, bins=bins, histtype='step', color='blue', linewidth=2, weights=wei, label='shallow learning')
plt.xlim(0,2)
plt.legend()
plt.title('hadronic energy spectra')
plt.xlabel('hadronic energy (GeV)')
plt.subplot(122)
plt.hist(res_off, bins=np.linspace(-2,2,200), linewidth=2, histtype='step', weights=wei, color='red', label='prod4')
plt.hist(res_krr, bins=bins, histtype='step', color='blue', linewidth=2, weights=wei, label='shallow learning')
plt.xlim(-1,1)
plt.legend()
plt.title('hadronic energy resolution')
plt.xlabel('(reco-true)/true')
plt.axvline(x=0., color='green', linestyle='--')
figsbs.savefig('plots/6d/ehad_spec_res_side_by_side.pdf')
figsbs.savefig('plots/6d/ehad_spec_res_side_by_side.png')
#~ plt.show()

# print out the statistic moments of resolution histograms
#~ m1, m2, m3, m4 = weighted_moments(res_off, wei)
#~ print(m1, m2, m3, m4)
#~ m1, m2, m3, m4 = weighted_moments(res_krr, wei)
#~ print(m1, m2, m3, m4)

# use ROOT to double check...
noff, bins, patches = plt.hist(res_off, bins=np.linspace(-1,1,100), linewidth=2, histtype='step', weights=wei, color='red', label='prod4')
nkrr, bins, patches = plt.hist(res_krr, bins=bins, histtype='step', color='blue', linewidth=2, weights=wei, label='shallow learning')
h = TH1F('h','',len(bins),bins[0],bins[-1])
for i in range(1,len(noff)+1):
  h.SetBinContent(i, noff[i-1])
print(h.GetMean(), h.GetRMS(), h.GetSkewness(), h.GetKurtosis(), h.Integral())
#~ h.Draw()
for i in range(1,len(nkrr)+1):
  h.SetBinContent(i, nkrr[i-1])
print(h.GetMean(), h.GetRMS(), h.GetSkewness(), h.GetKurtosis(), h.Integral())
#~ raw_input('press enter')
