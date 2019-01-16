#!/usr/bin/env python

from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

# command line parser
parser = argparse.ArgumentParser(description='The command line parser.')
parser.add_argument('-n', '--ntrain', help='number of events for training', type=int, default=10000)
args = parser.parse_args()
ntrain = args.ntrain

# load train data
df_fhc_train = pd.read_hdf('../data/fhc_train_test_split.h5', 'X_train')
df_fhc_train['polarity'] = 1
df_rhc_train = pd.read_hdf('../data/rhc_train_test_split.h5', 'X_train')
df_rhc_train['polarity'] = -1

# load test data
df_fhc_test = pd.read_hdf('../data/fhc_train_test_split.h5', 'X_test')
df_fhc_test['polarity'] = 1
df_rhc_test = pd.read_hdf('../data/rhc_train_test_split.h5', 'X_test')
df_rhc_test['polarity'] = -1

# select only ntrain events for training and testing
ntrain = min(len(df_fhc_train), len(df_rhc_train), ntrain)
df_fhc_train = df_fhc_train.sample(ntrain, random_state=0)
df_rhc_train = df_rhc_train.sample(ntrain, random_state=0)
df_fhc_test = df_fhc_test.sample(ntrain, random_state=0)
df_rhc_test = df_rhc_test.sample(ntrain, random_state=0)

# combine the tables and shuffle
X_train = pd.concat([df_fhc_train, df_rhc_train], axis=0)
X_train = X_train.sample(len(X_train), random_state=0)
X_test = pd.concat([df_fhc_test, df_rhc_test], axis=0)
X_test = X_test.sample(len(X_test), random_state=0)

# separate target
y_train = X_train['polarity']
X_train = X_train.drop(columns='polarity')
y_test = X_test['polarity']
X_test = X_test.drop(columns='polarity')

# run through auc scores with different depths
# code from https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3
max_depths = np.linspace(1, 32, 32, endpoint=True)

train_results = []
test_results = []
for max_depth in max_depths:
   dt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth), algorithm="SAMME", n_estimators=100)
   dt.fit(X_train, y_train)

   train_pred = dt.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous train results
   train_results.append(roc_auc)

   y_pred = dt.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous test results
   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('Tree depth')
# plt.show()
plt.savefig('plots/auc.pdf')
