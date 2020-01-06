#!/usr/bin/env python

from __future__ import print_function

from root_numpy import root2array
from sklearn.externals import joblib
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import ticker
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

# parse command line arguments
# I found it useful that a rule of thumb is step*n_neighbor = 2500.
parser = argparse.ArgumentParser(description='Train a hadronic energy SVM with specified parameters!')
parser.add_argument('-c','--n_neighbors',type=int,default='50')
parser.add_argument('-s','--step',type=int,default='50')
args = parser.parse_args()

# specified parameters
scaledown = args.step
nneighbors = args.n_neighbors

# retrieve training data
X = root2array('../no_truecc_cut_stride2_offset0.root',
               branches=['calehad', 'cvnpi0', 'cvnchargedpion', 'cvnneutron', 'cvnproton'],
               selection='mustopz<1275&&remidtrkismuon==1&&isnumucc==1',
               step=scaledown)
X = X.view(np.float32).reshape(X.shape + (-1,))
recoemu_official = root2array('../no_truecc_cut_stride2_offset0.root', branches='recoemu',
                              selection='mustopz<1275&&remidtrkismuon==1&&isnumucc==1',
                              step=scaledown)
trueenu = root2array('../no_truecc_cut_stride2_offset0.root', branches='trueenu',
                     selection='mustopz<1275&&remidtrkismuon==1&&isnumucc==1',
                     step=scaledown)
y = trueenu - recoemu_official
Xy = np.insert(X, 5, y, axis=1)

# fit the model
clf = LocalOutlierFactor(n_neighbors = nneighbors)
y_pred = clf.fit_predict(Xy)

#~ # plot the level sets of the decision function
#~ xx, yy = np.meshgrid(np.linspace(0, 15, 150), np.linspace(0, 15, 150))
#~ Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
#~ Z = Z.reshape(xx.shape)

#~ # level curve plot with original distribution
#~ plt.figure(1)
#~ plt.subplot(1, 2, 1)
#~ plt.title("Local Outlier Factor (LOF)")
#~ plt.contourf(xx, yy, -Z, locator=ticker.LogLocator(), cmap=plt.cm.Blues_r)
#~ a = plt.scatter(X, y, c='white',
                #~ edgecolor='k', s=20)
#~ plt.axis('tight')
#~ plt.xlim((0, 15))
#~ plt.ylim((0, 15))
#~ # make another plot with outliers removed
#~ plt.subplot(1, 2, 2)
#~ plt.title("Local Outlier Factor (LOF)")
#~ plt.contourf(xx, yy, -Z, locator=ticker.LogLocator(), cmap=plt.cm.Blues_r)
#~ b = plt.scatter(X[y_pred > 0], y[y_pred > 0], c='white',
                #~ edgecolor='k', s=20)
#~ plt.axis('tight')
#~ plt.xlim((0, 15))
#~ plt.ylim((0, 15))

#~ # histogram of inliers and outliers
#~ plt.figure(2)
#~ plt.hist(y_pred, histtype='step')

# save filtered data
os.system('mkdir -p outlier_removed_data')
Xpn = 'outlier_removed_data/hadron_regressors_5d_step{}neighbor{}.pkl'.format(scaledown, nneighbors)
ypn = 'outlier_removed_data/hadron_target_5d_step{}neighbor{}.pkl'.format(scaledown, nneighbors)
joblib.dump(X[y_pred > 0], Xpn)
joblib.dump(y[y_pred > 0], ypn)

#~ plt.show()
