#!/usr/bin/env python

from __future__ import print_function

from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import os

# retrieve training data
X = joblib.load('profiled_data/muon_trklen_active.pkl')
y = joblib.load('profiled_data/muon_truee_active.pkl')
ey = joblib.load('profiled_data/muon_dtruee_active.pkl')

# Instanciate a Gaussian Process model
kernel = RBF(10, (1e-2, 1e2))

# Instanciate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, alpha=(ey / y) ** 2,
                              n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
x = np.linspace(0,13,200).reshape(-1,1)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure()
#~ plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.errorbar(X.ravel(), y, ey, fmt='r.', markersize=1, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(0, 3)
plt.legend(loc='upper left')

os.system('mkdir -p plots')
plt.savefig('plots/muon_energy_estimator_active.pdf')

# save model
os.system('mkdir -p models')
joblib.dump(gp, 'models/muon_energy_estimator_active.pkl')
