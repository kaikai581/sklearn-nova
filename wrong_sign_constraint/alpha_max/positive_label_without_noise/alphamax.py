#!/usr/bin/env python

from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

def kappa(x, xmin, xmax):
    if x > xmin and x < xmax:
        return 1./(xmax-xmin)
    else:
        return 0

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
plt.show()