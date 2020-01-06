#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

#~ from root_numpy import root2array
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# retrieve training data
#~ X = root2array('../no_truecc_cut_stride2_offset0.root',
               #~ branches='recotrklenact',
               #~ selection='mustopz<1275&&isnumucc==1',
               #~ step=scaledown).reshape(-1,1)
#~ y = root2array('../no_truecc_cut_stride2_offset0.root',
               #~ branches='trueemu',
               #~ selection='mustopz<1275&&isnumucc==1',
               #~ step=scaledown)
scaledown = 50
X = joblib.load('../../svm/muon/outlier_removed_data/muon_trklen_active_step{}neighbor50.pkl'.format(scaledown))
y = joblib.load('../../svm/muon/outlier_removed_data/muon_truee_active_step{}neighbor50.pkl'.format(scaledown))

# rescale the regressors
scaler = preprocessing.StandardScaler().fit(X)

# fit the model
gp = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=1)
Xnorm = scaler.transform(X)
gp.fit(Xnorm, y)

# get prediction
y_pred = gp.predict(Xnorm)

# plot
fig = plt.figure()
np.histogram2d(y, X)
