#!/usr/bin/env python

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
import pickle

from sklearn.externals import joblib

# load train data
infn = 'train_test_split.h5'
df_fhc_test = pd.read_hdf(infn, 'fhc_test')
df_rhc_test = pd.read_hdf(infn, 'rhc_test')

# load trained model
model_pn = 'models/ten_vars_depth_10_samme.pkl'
if os.path.isfile(model_pn):
    bdt = joblib.load(model_pn)
else:
    print('Specified model does not exist.')
    sys.exit()

# make prediction
y_fhc_test = bdt.decision_function(df_fhc_test.drop(columns='truepdg'))
y_fhc_test = pd.DataFrame(y_fhc_test, columns=['bdt_score'])
y_rhc_test = bdt.decision_function(df_rhc_test.drop(columns='truepdg'))
y_rhc_test = pd.DataFrame(y_rhc_test, columns=['bdt_score'])
plt.subplot(211)
bins = np.linspace(-1,1,101)
label_hist = plt.hist(y_fhc_test.head(1000000).values,
        bins=bins,
        alpha=.5,
        edgecolor='k'
            #  histtype='step',
        , density=True
        )
plt.subplot(212)
mixture_hist = plt.hist(y_rhc_test.head(1000000).values,
        bins=bins,
        alpha=.5,
        edgecolor='k'
            #  histtype='step',
        , density=True
        )

# save the density decision functions
model_name = os.path.splitext(os.path.basename(model_pn))[0]
output_pn = 'histograms/{}.pkl'.format(model_name)
os.system('mkdir -p histograms')
with open(output_pn, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([label_hist, mixture_hist], f)

# save the predicted scores
outfn = 'bdt_scores.h5'
y_fhc_test.to_hdf(outfn, '{}/fhc_test_score'.format(model_name), complevel=9, complib='bzip2')
y_rhc_test.to_hdf(outfn, '{}/rhc_test_score'.format(model_name), complevel=9, complib='bzip2')

# plt.show()