#!/usr/bin/env python

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,2,1)
ax.scatter(xs, zs, color='green')
# Generated linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(xs,zs)
line = slope*xs + intercept
plt.plot(xs, line)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_title('low dimensional projection')
ax.text(1,4,r'$g(x)={:3.2f}x+{:3.2f}$'.format(slope, intercept), color='red', fontsize=12)

ax2 = fig.add_subplot(1,2,2)
dd = (line-zs)/zs
plt.hist(dd, bins=np.linspace(-1,1,10))
ax2.set_xlabel(r'$\frac{g(x_i)-z_i}{z_i}$', fontsize=15)
ax2.set_title(r'resolution, $RMS={:3.2f}$'.format(np.std(dd)))

plt.tight_layout()
# plt.show()
plt.savefig('illustration2d.png')