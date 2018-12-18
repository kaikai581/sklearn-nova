#!/usr/bin/env python

from __future__ import print_function
from cleanlab.latent_estimation import estimate_py_noise_matrices_and_cv_pred_proba
from cleanlab.pruning import get_noise_indices
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import pickle

train_dataset = '../data/confident_learning_fhc_rhc_train.h5'
result_latent_vars = 'results/latent_variables.pkl'

def make_training_dataset():
    ''' Preprocess input data if training data does not exist and save.'''

    datasets = {
        'fhc': '../data/confident_learning_fhc_train_test_split_40percent_train.h5',
        'rhc': '../data/confident_learning_rhc_train_test_split_40percent_train.h5'
    }
    X_train = dict()
    y_train = dict()
    for p in ['fhc', 'rhc']:
        X_train[p] = pd.read_hdf(datasets[p], 'X_train')
        X_train[p]['polarity'] = 1 if p == 'fhc' else 0
        y_train[p] = pd.read_hdf(datasets[p], 'y_train')
        X_train[p] = pd.concat([X_train[p], y_train[p]], axis=1)
    n_use = min(len(X_train['fhc']), len(X_train['rhc']))
    X_train = pd.concat([X_train['fhc'].head(n_use), X_train['rhc'].head(n_use)]).sample(frac=1, random_state=0).reset_index(drop=True)
    train_true_labels = X_train['truepdg']
    train_labels_with_errors = X_train['polarity']
    X_train = X_train.iloc[:,:10]
    # save to file
    X_train.to_hdf(train_dataset, 'X_train', complevel=9, complib='bzip2')
    train_true_labels.to_hdf(train_dataset, 'train_true_labels', complevel=9, complib='bzip2')
    train_labels_with_errors.to_hdf(train_dataset, 'train_labels_with_errors', complevel=9, complib='bzip2')
    
    

if __name__ == '__main__':

    # load data
    if not os.path.isfile(train_dataset):
        make_training_dataset()
    else:
        X_train = pd.read_hdf(train_dataset, 'X_train')
        train_true_labels = pd.read_hdf(train_dataset, 'train_true_labels')
        train_labels_with_errors = pd.read_hdf(train_dataset, 'train_labels_with_errors')
    
    # build models and estimate latent variables
    if not os.path.isfile(result_latent_vars):
        # start training
        est_py, est_nm, est_inv, confident_joint, psx = estimate_py_noise_matrices_and_cv_pred_proba(
            X=X_train.values,
            s=train_labels_with_errors,
            # clf=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        )

        # save results
        if not os.path.exists('results'):
            os.makedirs('results')
        with open(result_latent_vars, 'wb') as output:
            pickle.dump(est_py, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(est_nm, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(est_inv, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(confident_joint, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(psx, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(result_latent_vars, 'rb') as inf:
            est_py = pickle.load(inf)
            est_nm = pickle.load(inf)
            est_inv = pickle.load(inf)
            confident_joint = pickle.load(inf)
            psx = pickle.load(inf)
    
    # print flipped labels
    label_errors = get_noise_indices(
        s=train_labels_with_errors, # required
        psx=psx, # required
        inverse_noise_matrix=est_inv, # not required, include to avoid recomputing
        confident_joint=confident_joint, # not required, include to avoid recomputing
    )
    print(pd.concat([train_labels_with_errors, train_true_labels, pd.DataFrame(data=label_errors, columns=['flipped_label'])], axis=1))