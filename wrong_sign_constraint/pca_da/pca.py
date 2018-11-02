#!/usr/bin/env python

from __future__ import print_function

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd

# load data and do some basic cut
df = pd.read_hdf('../data/wrong_sign_bdt_variables.h5', 'rhc_dataframe')
df = df[(df['truepdg'] != -5) & (df['cce'] < 15) & (df['bpfbestmuonchi2t'] < 20) & (df['bpfbestmuondedxll'] > -8) & (df['hadnhit'] < 250) & (df['recow'] > 0) & (df['hade'] < 5)]
# convert integer values into float
df['hadnhit'] = pd.to_numeric(df['hadnhit'], errors='coerce')
df['nmichels'] = pd.to_numeric(df['nmichels'], errors='coerce')
# after all the cuts, use reset_index to rebuild the row index
df = df.reset_index(drop=True).head(10000)

# features included
fnames = ['hade','hadnhit','cosnumi', 'cce','mue','nmichels','numuhadefrac','recow','bpfbestmuondedxll', 'bpfbestmuonchi2t']

# save features as a numpy array
X = df.loc[:, fnames].values
# Standardizing the features
X = StandardScaler().fit_transform(X)

# print(X, X.shape)
pca = PCA(n_components=10)
# pca = KernelPCA(n_components=10, kernel='poly')
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8', 'principal component 9', 'principal component 10'])
# attach the true pdg value
finalDf = pd.concat([principalDf[['principal component 1', 'principal component 2']], df[['truepdg']]], axis = 1)
print(finalDf)

# plot results
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [-14, 14]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['truepdg'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 20)
ax.legend(targets)
plt.show()