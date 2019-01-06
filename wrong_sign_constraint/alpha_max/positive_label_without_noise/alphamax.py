#!/usr/bin/env python

from __future__ import print_function
from scipy.optimize import minimize

import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import pickle

def kappa(x, xmin, xmax):
    if x > xmin and x < xmax:
        return 1./(xmax-xmin)
    return 0

def f1(x):
    vomega = h_score_fhc[0]
    bin_bounds = h_score_fhc[1]
    bin_size = bin_bounds[1] - bin_bounds[0]
    vomega = [x*bin_size for x in vomega]
    result = 0
    bin_bounds_idx = 0
    for omega in vomega:
        xmin = bin_bounds[bin_bounds_idx]
        xmax = bin_bounds[bin_bounds_idx+1]
        bin_bounds_idx += 1
        result += omega*kappa(x, xmin, xmax)
    return result

def h1(x, vbeta):
    vomega = h_score_rhc[0]
    bin_bounds = h_score_rhc[1]
    bin_size = bin_bounds[1] - bin_bounds[0]
    vomega = [x*bin_size for x in vomega]
    numerator = 0
    denominator = 0
    bin_bounds_idx = 0
    for beta, omega in zip(vbeta, vomega):
        xmin = bin_bounds[bin_bounds_idx]
        xmax = bin_bounds[bin_bounds_idx+1]
        bin_bounds_idx += 1
        numerator += beta*omega*kappa(x, xmin, xmax)
        denominator += beta*omega
    return numerator/denominator

def h(x, vbeta):
    vomega = h_score_rhc[0]
    bin_bounds = h_score_rhc[1]
    bin_size = bin_bounds[1] - bin_bounds[0]
    vomega = [x*bin_size for x in vomega]
    alpha = 0
    numerator_h0 = 0
    bin_bounds_idx = 0
    for beta, omega in zip(vbeta, vomega):
        xmin = bin_bounds[bin_bounds_idx]
        xmax = bin_bounds[bin_bounds_idx+1]
        bin_bounds_idx += 1
        alpha += beta*omega
        numerator_h0 += (1-beta)*omega*kappa(x, xmin, xmax)
    return alpha*f1(x)+numerator_h0

def L(vbeta):
    result1 = 0
    result2 = 0
    n1_used = 0
    n_used = 0
    score_fhc_short = score_fhc.sample(10000, random_state=0)
    score_rhc_short = score_rhc.sample(10000, random_state=0)
    for i in range(len(score_fhc_short)):
        x = score_fhc.iloc[i]
        h1_val = h1(x, vbeta)
        if h1_val > 0:
            n1_used += 1
            result1 += math.log(h1_val)
    for i in range(len(score_rhc_short)):
        x = score_rhc.iloc[i]
        h_val = h(x, vbeta)
        if h_val > 0:
            n_used += 1
            result2 += math.log(h_val)
    print(result1, result2)
    return -result1-result2

def vbeta_optimization():
    # vbeta_init = np.full(len(h_score_fhc[0]), )
    return

# load predictions
prediction_in = 'bdt_scores.h5'
y_fhc_test = pd.read_hdf(prediction_in, 'ten_vars_depth_10_samme/fhc_test_score')
y_rhc_test = pd.read_hdf(prediction_in, 'ten_vars_depth_10_samme/rhc_test_score')
pdg_fhc_test = pd.read_hdf(prediction_in, 'fhc_test_truepdg')
pdg_rhc_test = pd.read_hdf(prediction_in, 'rhc_test_truepdg')
# concatenate the tables
fhc_test = pd.concat([y_fhc_test, pdg_fhc_test], axis=1)
rhc_test = pd.concat([y_rhc_test, pdg_rhc_test], axis=1)

# process command line arguments
parser = argparse.ArgumentParser(description='Command line options.')
parser.add_argument('-n', '--nevents', type=int, default=1000000)
args = parser.parse_args()
nevents = min(len(fhc_test), len(rhc_test), args.nevents)

# select only specified number of events
fhc_test = fhc_test.sample(nevents, random_state=1)
rhc_test = rhc_test.sample(nevents, random_state=1)
score_fhc = fhc_test['bdt_score']
score_rhc = rhc_test['bdt_score']

# make histograms
bins = np.linspace(-1,1,101)
h_score_fhc = plt.hist(score_fhc.values, bins=bins, density=True, histtype='step')
h_score_rhc = plt.hist(score_rhc.values, bins=bins, density=True, histtype='step')
# plt.show()

# test the code
vbeta = np.full(len(h_score_fhc[0]), .85)
vbeta[len(vbeta)//2:] = .05
print(vbeta)
print(L(vbeta))

# try out scipy optimization
vbeta_optimization()