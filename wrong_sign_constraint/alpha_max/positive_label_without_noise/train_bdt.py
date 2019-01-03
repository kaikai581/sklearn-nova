#!/usr/bin/env python

from __future__ import print_function
import os
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier

# load train data
infn = 'train_test_split.h5'
df_fhc_train = pd.read_hdf(infn, 'fhc_train')
df_rhc_train = pd.read_hdf(infn, 'rhc_train')

# remove truepdg column, add bdt target labels
# concatenate two tables and shuffle
df_fhc_train = df_fhc_train.drop(columns='truepdg')
df_fhc_train['target'] = 1
df_rhc_train = df_rhc_train.drop(columns='truepdg')
df_rhc_train['target'] = -1
df_train = pd.concat([df_fhc_train, df_rhc_train])
df_train = df_train.sample(len(df_train),random_state=0)

# prepare target and regressor
y_train = df_train['target']
X_train = df_train.drop(columns='target')

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                         algorithm="SAMME",
                         n_estimators=100)

# fit the model and save
os.system('mkdir -p models')
model_pn = 'models/ten_vars_depth_10_samme.pkl'
if not os.path.isfile(model_pn):
    bdt.fit(X_train, y_train)
    joblib.dump(bdt, model_pn)
else:
    bdt = joblib.load(model_pn)