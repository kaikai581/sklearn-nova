#!/usr/bin/env python

from __future__ import print_function
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os, sys

# prepare input data
data_path = '../data'
data_fn = 'wrong_sign_bdt_variables.h5'
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
yenc = np.apply_along_axis(lambda x: x/-14, 0, yenc).astype('int')

# training and test split
X_train, X_test, y_train, y_test = train_test_split(X, yenc, test_size=0.8, random_state=42)
# even out the two classes
nnumu = len(y_train[y_train == -1])
numubar_ctr = 0
for idx in range(len(y_train)):
    if y_train[idx] == 1:
        numubar_ctr += 1
        if numubar_ctr > nnumu:
            break
X_remain = X_train[idx:,:]
y_remain = y_train[idx:]
y_remain_sel = (y_remain == -1)
X_remain = X_remain[y_remain_sel]
y_remain = y_remain[y_remain_sel]
X_train = np.concatenate((X_train[:idx,:], X_remain), axis=0)
y_train = np.concatenate((y_train[:idx], y_remain), axis=0)

# transform numpy ndarrays into dataframes
X_train = pd.DataFrame(data=X_train, index=None, columns=fnames)
y_train = pd.DataFrame(data=y_train, index=None, columns=['truepdg'])
X_test = pd.DataFrame(data=X_test, index=None, columns=fnames)
y_test = pd.DataFrame(data=y_test, index=None, columns=['truepdg'])

# save to file
X_train.to_hdf('../data/train_test_split.h5', 'X_train', complevel=9, complib='bzip2')
y_train.to_hdf('../data/train_test_split.h5', 'y_train', complevel=9, complib='bzip2')
X_test.to_hdf('../data/train_test_split.h5', 'X_test', complevel=9, complib='bzip2')
y_test.to_hdf('../data/train_test_split.h5', 'y_test', complevel=9, complib='bzip2')