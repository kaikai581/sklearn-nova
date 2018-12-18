#!/usr/bin/env python

from __future__ import print_function
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os, sys

# command line argument
parser = argparse.ArgumentParser(description='Command line argument parser for this script.')
parser.add_argument('-c','--horn_current',default='rhc')
parser.add_argument('-t','--test_fraction',default=0.6)
args = parser.parse_args()
polarity = args.horn_current
test_fraction = args.test_fraction

# horn current safeguard
if polarity not in ['fhc', 'rhc']:
    print('Horn current not valid: {}'.format(polarity))
    sys.exit()

# prepare input data
data_path = '../data'
data_fn = '{}_wrong_sign_bdt_variables.h5'.format(polarity)
data_pathname = os.path.join(data_path, data_fn)
if not os.path.isfile(data_pathname):
    print('Input file {} does not exist.'.format(data_pathname))
    sys.exit()

# load input data and make basic cuts
df = pd.read_hdf(data_pathname)
df = df[(df['truepdg'] != -5) & (df['cce'] < 15) & (df['bpfbestmuonchi2t'] < 20) & (df['bpfbestmuondedxll'] > -8) & (df['hadnhit'] < 250) & (df['recow'] > 0) & (df['hade'] < 5) & (abs(df['truepdg']) == 14)]
# convert integer values into float
df['hadnhit'] = pd.to_numeric(df['hadnhit'], errors='coerce')
df['nmichels'] = pd.to_numeric(df['nmichels'], errors='coerce')
# after all the cuts, use reset_index to rebuild the row index
df = df.reset_index(drop=True)

# features included
fnames = ['hade','hadnhit','cosnumi', 'cce','mue','nmichels','numuhadefrac','recow','bpfbestmuondedxll','bpfbestmuonchi2t']

# BDT wants [-1, 1] label
X = df.loc[:, fnames].values
y = df.loc[:, 'truepdg'].values
y = y.astype('int')
yenc = np.copy(y)
yenc = np.apply_along_axis(lambda x: (x/14+1)/2, 0, yenc).astype('int')

# training and test split
X_train, X_test, y_train, y_test = train_test_split(X, yenc, test_size=test_fraction, random_state=42)

# transform numpy ndarrays into dataframes
X_train = pd.DataFrame(data=X_train, index=None, columns=fnames)
y_train = pd.DataFrame(data=y_train, index=None, columns=['truepdg'])
X_test = pd.DataFrame(data=X_test, index=None, columns=fnames)
y_test = pd.DataFrame(data=y_test, index=None, columns=['truepdg'])

# save to file
outpfn = '../data/confident_learning_{}_train_test_split_{}percent_train.h5'.format(polarity, int(100*(1-test_fraction)))
X_train.to_hdf(outpfn, 'X_train', complevel=9, complib='bzip2')
y_train.to_hdf(outpfn, 'y_train', complevel=9, complib='bzip2')
X_test.to_hdf(outpfn, 'X_test', complevel=9, complib='bzip2')
y_test.to_hdf(outpfn, 'y_test', complevel=9, complib='bzip2')