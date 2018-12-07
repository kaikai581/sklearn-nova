#!/usr/bin/env python

# Refer to Abahilash's file at /nova/app/users/ayallapp/Ws_S18-10-12/test_train/nhit/TMVAClassification_All.C
# "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20"

from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier

# prepare data
data_path = '../data'
data_fn = 'train_test_split.h5'
data_pathname = os.path.join(data_path, data_fn)
if not os.path.isfile(data_pathname):
    print('Input file {} does not exist.'.format(data_pathname))
    sys.exit()

# load input data
X_train = pd.read_hdf(data_pathname, 'X_train')
y_train = pd.read_hdf(data_pathname, 'y_train').values.flatten()

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=200)

# fit the model and save
os.system('mkdir -p models')
model_pn = 'models/ten_vars_depth_3.pkl'
bdt = None
if not os.path.isfile(model_pn):
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             algorithm="SAMME",
                             n_estimators=200)
    bdt.fit(X_train, y_train)
    joblib.dump(bdt, model_pn)
else:
    bdt = joblib.load(model_pn)


plot_colors = "br"
plot_step = 0.02
class_names = "12"

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X_train)
print(twoclass_output)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(111)
for i, n, c in zip([1,-1], class_names, plot_colors):
    print(y_train == i)
    plt.hist(twoclass_output[y_train == i],
             bins=25,
             range=plot_range,
             facecolor=c,
             label='component {}'.format(n),
             alpha=.5,
             edgecolor='k'
            #  histtype='step',
             )
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
# plt.show()
plt.savefig('bdt_score.pdf')