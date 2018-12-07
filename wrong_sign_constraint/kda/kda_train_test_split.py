#!/usr/bin/env python

from __future__ import print_function

from skkda.base import KernelDiscriminantAnalysis

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib

# command line argument
parser = argparse.ArgumentParser(description='Command line argument parser for this script.')
parser.add_argument('-r', '--ntrain', type=int, default=10000, help='Number of training samples.')
parser.add_argument('-e', '--ntest', type=int, default=10000, help='Number of test samples.')
args = parser.parse_args()
ntrain = args.ntrain
ntest = args.ntest

# retrieve data for training and testing
infpn = '../data/train_test_split.h5'
X_train = pd.read_hdf(infpn, 'X_train').loc[:ntrain,:].values
y_train = pd.read_hdf(infpn, 'y_train').loc[:ntrain].values.flatten()
y_train = np.apply_along_axis(lambda x: (x+1)/2, 0, y_train).astype('int')
X_test = pd.read_hdf(infpn, 'X_test').loc[:ntest,:].values
y_test = pd.read_hdf(infpn, 'y_test').loc[:ntest].values.flatten()
y_test = np.apply_along_axis(lambda x: (x+1)/2, 0, y_test).astype('int')

# linear discriminative analysis
gamma = 0.1
# fit the model and save
os.system('mkdir -p models')
model_pn = 'models/kda_train{}_gamma{}.pkl'.format(ntrain, gamma)
kda = None
if not os.path.isfile(model_pn):
    # Create and fit an AdaBoosted decision tree
    kda = KernelDiscriminantAnalysis(gamma=gamma)
    kda.fit(X_train, y_train)
    joblib.dump(kda, model_pn)
else:
    kda = joblib.load(model_pn)

X_r = kda.transform(X_test)

outfn = ['first_var.pdf', 'second_var.pdf']
for i in [0, 1]:
    x_min = np.amin(X_r[:,i])
    x_max = np.amax(X_r[:,i])
    bins = np.linspace(x_min, x_max, num=100)

    plt.figure(i)
    h, bins2, p = plt.hist(X_r[y_test == 1][:,i], color='navy', alpha=.6, lw=2, label=r'$\bar{\nu}_\mu$', bins=bins, histtype='step')
    plt.hist(X_r[y_test == 0][:,i], color='turquoise', alpha=.5, lw=2, label=r'$\nu_\mu$', bins=bins)
    plt.hist(X_r[:,i], color='red', alpha=.8, lw=2, label='total', bins=bins, histtype='step')
    plt.legend()
    plt.savefig(outfn[i])