#!/usr/bin/env python

from __future__ import print_function

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd

# load data and do some basic cut
df = pd.read_hdf('../data/wrong_sign_bdt_variables.h5', 'rhc_dataframe')
df = df[(df['truepdg'] != -5) & (df['cce'] < 15) & (df['bpfbestmuonchi2t'] < 20) & (df['bpfbestmuondedxll'] > -8) & (df['hadnhit'] < 250) & (df['recow'] > 0) & (df['hade'] < 5) & (abs(df['truepdg']) == 14)]
# convert integer values into float
df['hadnhit'] = pd.to_numeric(df['hadnhit'], errors='coerce')
df['nmichels'] = pd.to_numeric(df['nmichels'], errors='coerce')
# after all the cuts, use reset_index to rebuild the row index
# df = df.reset_index(drop=True).head(100000)
df = df.reset_index(drop=True)

# features included
fnames = ['hade','hadnhit','cosnumi', 'cce','mue','nmichels','numuhadefrac','recow','bpfbestmuondedxll', 'bpfbestmuonchi2t']

# save features as a numpy array
X = df.loc[:, fnames].values
y = df.loc[:, 'truepdg'].values
y = y.astype('int')
# Standardizing the features
# X = StandardScaler().fit_transform(X)

# linear discriminative analysis
lda = LinearDiscriminantAnalysis()
X_r = lda.fit(X, y).transform(X)

plt.figure()
colors = ['navy', 'turquoise']
bins = None
# for color, i, target_name in zip(colors, [-14, 14], [r'$\bar{\nu}_mu',r'\nu_\mu']]):
h, bins, p = plt.hist(X_r[y == -14], color='navy', alpha=.6, lw=2, label=r'$\bar{\nu}_\mu$', bins=100, histtype='step')
plt.hist(X_r[y == 14], color='turquoise', alpha=.99, lw=2, label=r'$\nu_\mu$', bins=bins)
plt.hist(X_r, color='red', alpha=.8, lw=2, label='total', bins=bins, histtype='step')
plt.legend()
plt.show()

# print(finalDf.shape)