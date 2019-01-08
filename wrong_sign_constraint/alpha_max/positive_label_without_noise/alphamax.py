#!/usr/bin/env python

from __future__ import print_function
from scipy.optimize import minimize

import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import pickle
import scipy

def kappa(x, xmin, xmax):
    if x > xmin and x < xmax:
        return 1./(xmax-xmin)
    return 0

def f1(x):
    result = 0
    bin_bounds_idx = 0
    for omega in vomega_fhc:
        xmin = bin_bounds[bin_bounds_idx]
        xmax = bin_bounds[bin_bounds_idx+1]
        bin_bounds_idx += 1
        result += omega*kappa(x, xmin, xmax)
    return result

def h1(x, vbeta):
    numerator = 0
    denominator = 0
    bin_bounds_idx = 0
    for beta, omega in zip(vbeta, vomega_rhc):
        xmin = bin_bounds[bin_bounds_idx]
        xmax = bin_bounds[bin_bounds_idx+1]
        bin_bounds_idx += 1
        numerator += beta*omega*kappa(x, xmin, xmax)
        denominator += beta*omega
    return numerator/denominator

def h(x, vbeta):
    alpha = 0
    numerator_h0 = 0
    bin_bounds_idx = 0
    for beta, omega in zip(vbeta, vomega_rhc):
        xmin = bin_bounds[bin_bounds_idx]
        xmax = bin_bounds[bin_bounds_idx+1]
        bin_bounds_idx += 1
        alpha += beta*omega
        numerator_h0 += (1-beta)*omega*kappa(x, xmin, xmax)
    return alpha*f1(x)+numerator_h0

def L(vbeta):
    result1 = 0
    result2 = 0
    n1_used = 0
    n_used = 0
    score_fhc_short = score_fhc.sample(nlikelihood, random_state=0)
    score_rhc_short = score_rhc.sample(nlikelihood, random_state=0)
    for i in range(len(score_fhc_short)):
        x = score_fhc.iloc[i]
        h1_val = h1(x, vbeta)
        if h1_val > 0:
            n1_used += 1
            result1 += math.log(h1_val)
    for i in range(len(score_rhc_short)):
        x = score_rhc.iloc[i]
        h_val = h(x, vbeta)
        if h_val > 0:
            n_used += 1
            result2 += math.log(h_val)
    print(result1, result2)
    return -result1-result2

def vbeta_optimization():
    c = 0.5
    nbeta = len(h_score_fhc[0])
    vbeta_init = np.full(nbeta, c/nbeta)
    # define equality constraint
    eq_cons = dict()
    eq_cons['type'] = 'eq'
    eq_cons['fun'] = lambda x: level_surface_fun(x, c)
    eq_cons['jac'] = level_surface_jac
    res = minimize(L, vbeta_init, method='SLSQP', constraints=[eq_cons], bounds=vbeta_bounds)
    print(res.x)
    return

def level_surface_fun(vbeta, c):
    result = 0
    for i in range(len(vomega_rhc)):
        result += vbeta[i] * vomega_rhc[i]
    result -= c
    return np.array([result])

def level_surface_jac(vbeta):
    return np.array(vomega_rhc)

# main script
if __name__ == '__main__':
    # load predictions
    prediction_in = 'bdt_scores.h5'
    y_fhc_test = pd.read_hdf(prediction_in, 'ten_vars_depth_10_samme/fhc_test_score')
    y_rhc_test = pd.read_hdf(prediction_in, 'ten_vars_depth_10_samme/rhc_test_score')
    pdg_fhc_test = pd.read_hdf(prediction_in, 'fhc_test_truepdg')
    pdg_rhc_test = pd.read_hdf(prediction_in, 'rhc_test_truepdg')
    # concatenate the tables
    fhc_test = pd.concat([y_fhc_test, pdg_fhc_test], axis=1)
    rhc_test = pd.concat([y_rhc_test, pdg_rhc_test], axis=1)

    # process command line arguments
    parser = argparse.ArgumentParser(description='Command line options.')
    parser.add_argument('-n', '--nevents', type=int, default=1000000, help='Number of events used for producing histograms.')
    parser.add_argument('-l', '--nlikelihood', type=int, default=10000, help='Number of events used for forming likelihood functions.')
    args = parser.parse_args()
    nevents = min(len(fhc_test), len(rhc_test), args.nevents)
    nlikelihood = min(nevents, args.nlikelihood)

    # select only specified number of events to make histograms
    fhc_test = fhc_test.sample(nevents, random_state=1)
    rhc_test = rhc_test.sample(nevents, random_state=1)
    score_fhc = fhc_test['bdt_score']
    score_rhc = rhc_test['bdt_score']

    # make histograms
    bins = np.linspace(-1,1,101)
    h_score_fhc = plt.hist(score_fhc.values, bins=bins, density=True, histtype='step')
    h_score_rhc = plt.hist(score_rhc.values, bins=bins, density=True, histtype='step')
    # plt.show()

    # define some variables for convenience
    # normalize omega vector so that they sum up to 1
    vomega_fhc = h_score_fhc[0]
    vomega_rhc = h_score_rhc[0]
    bin_bounds = h_score_rhc[1]
    bin_size = bin_bounds[1] - bin_bounds[0]
    vomega_fhc = [x*bin_size for x in vomega_fhc]
    vomega_rhc = [x*bin_size for x in vomega_rhc]
    vbeta_bounds = scipy.optimize.Bounds(np.zeros(len(vomega_fhc)), np.ones(len(vomega_fhc)), keep_feasible=True)

    # test the code
    # vbeta = np.full(len(h_score_fhc[0]), .85)
    # vbeta[len(vbeta)//3:] = .05
    # print(vbeta)
    # print(L(vbeta))
    # print(level_surface_fun(vbeta, .5))
    # print(level_surface_jac(vbeta))
    # print(scipy.optimize.check_grad(lambda x: level_surface_fun(x, 0.3), level_surface_jac, vbeta))

    # try out scipy optimization
    vbeta_optimization()