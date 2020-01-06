#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import os

C_ranges = np.logspace(-3, 3, 7)
gamma_ranges = np.logspace(-4, 2, 7)

for i in range(len(C_ranges)):
  for j in range(len(gamma_ranges)):
    os.system('time ./muon_energy_estimator_active.py -c {} -g {}'.format(C_ranges[i], gamma_ranges[j]))
