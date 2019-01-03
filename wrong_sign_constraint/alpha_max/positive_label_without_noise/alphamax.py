#!/usr/bin/env python

from __future__ import print_function

import matplotlib.pyplot as plt
import os
import pickle

# load histograms
model_pn = 'models/ten_vars_depth_10_samme.pkl'
model_name = os.path.splitext(os.path.basename(model_pn))[0]
in_hist_pn = 'histograms/{}.pkl'.format(model_name)
with open(in_hist_pn, 'rb') as f:
    label_hist, mixture_hist = pickle.load(f)

print(sum(label_hist[0])*(label_hist[1][1]-label_hist[1][0]))