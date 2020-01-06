#!/usr/bin/env python

from __future__ import print_function

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

col_names = [
  'run',
  'subrun',
  'cycle',
  'evt',
  'subevt',
  'isnumucc',
  'trueenu',
  'trueemu',
  'recoemu',
  'recotrklenact',
  'recotrklencat',
  'mustopz',
  'trueehad',
  'recoehad',
  'calehad',
  'recoq2',
  'npng',
  'remidtrkismuon',
  'cvnelectron',
  'cvnmuon',
  'cvnpi0',
  'cvnchargedpion',
  'cvnneutron',
  'cvnproton',
  'weight',
  'intmode'
]
df = pd.read_csv('../make_training_and_test_datasets/grid_output/nd_p6_standard_numucc_selection_stride1_offset0.1_of_200.txt',
                 delimiter='\s+',
                 names=col_names)

X = df.as_matrix(['cvnpi0','cvnchargedpion','cvnneutron','cvnproton'])
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.mean_)
print(pca.components_)
print(pca.components_[0])
print(pca.explained_variance_)
