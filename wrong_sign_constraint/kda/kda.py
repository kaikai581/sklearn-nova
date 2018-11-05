#!/usr/bin/env python

from __future__ import print_function

from skkda.base import KernelDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import wget

# command line argument
parser = argparse.ArgumentParser(description='Command line argument parser for this script.')
parser.add_argument('-n', '--nsamples', type=int, default=10000, help='Number of training samples.')
args = parser.parse_args()
nsamples = args.nsamples

# define dataframe's column names
col_names = ['run','subrun','cycle','evt','subevt','hade','hadnhit','cosnumi',
'cce','mue','nmichels','numuhadefrac','recow','bpfbestmuondedxll',
'bpfbestmuonchi2t','muonid','remid','xseccvwgt2018','ppfxfluxcvwgt','trueywgt',
'truey','antinumubdt','truepdg']

infn = 'nd_rhc_wrong_sign_reweight_y.txt'
# Check if file exists. If it does not, check it out.
if not os.path.isfile(infn):
    url = r'https://p-def3.pcloud.com/cBZ0aOSDHZ8FDRKHZamINZZT6ebI7Z2ZZRwVZkZ5aPowZKZW7ZlkZt7Zp0ZJ7Za0Zh5ZMkZ5XZf7ZfXZgXZoXZJQDM7ZVi0BrnLL4n7GWFXwRHP5xHILaQe7/nd_rhc_wrong_sign_reweight_y.txt'
    wget.download(url)

# read data into a dataframe
df = pd.read_csv(infn, index_col=False, names=col_names, sep=' ', header=None)
df = df[(df['truepdg'] != -5) & (df['cce'] < 15) & (df['bpfbestmuonchi2t'] < 20) & (df['bpfbestmuondedxll'] > -8) & (df['hadnhit'] < 250) & (df['recow'] > 0) & (df['hade'] < 5) & (abs(df['truepdg']) == 14)]
# convert integer values into float
df['hadnhit'] = pd.to_numeric(df['hadnhit'], errors='coerce')
df['nmichels'] = pd.to_numeric(df['nmichels'], errors='coerce')
# after all the cuts, use reset_index to rebuild the row index
df_train = df.reset_index(drop=True).head(nsamples)

# features included
fnames = ['hade','hadnhit','cosnumi', 'cce','mue','nmichels','numuhadefrac','recow','bpfbestmuondedxll','bpfbestmuonchi2t']

# save features as a numpy array
X = df_train.loc[:, fnames].values
y = df_train.loc[:, 'truepdg'].values
y = y.astype('int')
yenc = np.copy(y)
for i in range(len(y)):
    if y[i] == -14: yenc[i] = 0
    else: yenc[i] = 1
# Standardizing the features
X = StandardScaler().fit_transform(X)

# linear discriminative analysis
kda = KernelDiscriminantAnalysis(gamma=1)
X_r = kda.fit(X, yenc).transform(X)
# kda.fit(X, yenc)
# X_r = kda.transform(df.loc[:, fnames].values)
print(X_r[:,1][yenc == 0], len(X_r[:,1][yenc == 0]))
print(X_r[:,1][yenc == 1], len(X_r[:,1][yenc == 1]))
print(X_r)

outfn = ['first_var.pdf', 'second_var.pdf']
for i in [0, 1]:
    x_min = np.amin(X_r[:,i])
    x_max = np.amax(X_r[:,i])
    bins = np.linspace(x_min, x_max, num=100)

    plt.figure(i)
    h, bins2, p = plt.hist(X_r[yenc == 0][:,i], color='navy', alpha=.6, lw=2, label=r'$\bar{\nu}_\mu$', bins=bins, histtype='step')
    plt.hist(X_r[yenc == 1][:,i], color='turquoise', alpha=.5, lw=2, label=r'$\nu_\mu$', bins=bins)
    plt.hist(X_r[:,i], color='red', alpha=.8, lw=2, label='total', bins=bins, histtype='step')
    plt.legend()
    plt.savefig(outfn[i])
