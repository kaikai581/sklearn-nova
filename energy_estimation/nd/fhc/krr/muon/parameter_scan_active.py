#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import os

a_ranges = np.logspace(-3, -1, 3)
gamma_ranges = np.logspace(-4, 2, 7)

for i in range(len(a_ranges)):
  for j in range(len(gamma_ranges)):
    os.system('time ./predict_from_model_active.py -a {} -g {} -n 20000'.format(a_ranges[i], gamma_ranges[j]))
