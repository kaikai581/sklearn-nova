#!/usr/bin/env python

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

# prepare data
data_path = '../data'
data_fn_fhc = 'fhc_train_test_split.h5'
data_fn_rhc = 'rhc_train_test_split.h5'
data_pathname_fhc = os.path.join(data_path, data_fn_fhc)
data_pathname_rhc = os.path.join(data_path, data_fn_rhc)
for data_pathname in [data_pathname_fhc, data_pathname_rhc]:
    if not os.path.isfile(data_pathname):
        print('Input file {} does not exist.'.format(data_pathname))
        sys.exit()
# retrieve feature dataframes
X_fhc = pd.read_hdf(data_pathname_fhc, 'X_train')
X_fhc['polarity'] = 1
X_rhc = pd.read_hdf(data_pathname_rhc, 'X_train')
X_rhc['polarity'] = -1
# even out number of training samples
nfhc = len(X_fhc)
nrhc = len(X_rhc)
if nfhc > nrhc:
    X_fhc = X_fhc.sample(n=nrhc)
else:
    X_rhc = X_rhc.sample(n=nfhc)
X_train = pd.concat([X_fhc, X_rhc])
# shuffle training data
X_train = shuffle(X_train)
# prepare features and labels
y_train = X_train['polarity']
X_train = X_train.drop(columns='polarity')

# fit and save the model
os.system('mkdir -p models')
model_pn = 'models/polarity_as_labels.pkl'
# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                            algorithm="SAMME",
                            n_estimators=200)
bdt.fit(X_train, y_train)
joblib.dump(bdt, model_pn)