#!/usr/bin/env python

from __future__ import print_function
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#from root_numpy import root2array
from sklearn.externals import joblib
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import os

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax = Axes3D(fig)

#scaledown = 30
#X = root2array('../grid_output_stride5_offset0.root',
#               branches=['recotrklenact','recotrklencat'],
#               step=scaledown)
#X = X.view(np.float32).reshape(X.shape + (-1,))

#print(np.amax(X[:,0]),np.amin(X[:,0]),np.amax(X[:,1]),np.amin(X[:,1]))

maxact = 12.5877*1.05
maxcat = 2.69112*1.05
# Make data.
X = np.arange(0, maxact, maxact/50)
Y = np.arange(0, maxcat, maxcat/50)
xx, yy = np.meshgrid(X, Y)
svr = joblib.load('models/muon_energy_estimator_combined.pkl')
Z = svr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=6., azim=225)
ax.set_xlabel('track length in active (m)')
ax.set_ylabel('track length in catcher (m)')
ax.set_zlabel('muon total energy (GeV)')
#plt.show()
os.system('mkdir -p plots')
plt.savefig('plots/muon_energy_estimator_combined.pdf')
