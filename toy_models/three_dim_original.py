#!/usr/bin/env python

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return 0.3*x + 0.2*y

# deterministic random number generation
np.random.seed(seed=0)
xs = np.linspace(0,9,100)
ys = np.random.uniform(0,9,100)
es = np.random.randn(100)
# equation: z = 0.3x + 0.2y
zs = f(xs, ys) + 0.4*es

xx, yy = np.meshgrid(np.linspace(0,9,10), np.linspace(0,9,10))
zz = f(xx, yy)

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(xx, yy, zz, alpha=.2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.text(-2, 5, 5.5, r'$z_i=0.3x_i+0.2y_i+\epsilon_i, \epsilon_i\in\mathcal{N}(0,0.16)$', color='red', fontsize=12)
ax.text(-2, 5, 4.75, r'$f(x_i,y_i)=0.3x_i+0.2y_i$', color='red', fontsize=12)
ax.scatter(xs, ys, zs, color='green')

ax2 = fig.add_subplot(1,2,2)
dd = (f(xs, ys)-zs)/zs
plt.hist(dd, bins=np.linspace(-1,1,10))
ax2.set_xlabel(r'$\frac{f(x_i,y_i)-z_i}{z_i}$', fontsize=15)
ax2.set_title(r'resolution, $RMS={:3.2f}$'.format(np.std(dd)))

plt.tight_layout()
plt.show()