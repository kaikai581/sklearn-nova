#!/usr/bin/env python

from __future__ import print_function
import argparse
import os.path
import pandas as pd
import scipy.stats
import sys

all_data_pn = '../hdf5/wrong_sign_bdt_variables.h5'
subsample_pn = '../hdf5/check_rcn_subsample.h5'

def closeness_score(nutype, varname, metric_fuc):
    fhc_sample = pd.read_hdf(subsample_pn, 'fhc_'+nutype)
    rhc_sample = pd.read_hdf(subsample_pn, 'rhc_'+nutype)
    return metric_fuc(fhc_sample[varname].values, rhc_sample[varname].values)

def make_subsample(polarity, pdg, tablename):

    if not os.path.isfile(all_data_pn):
        print('Input data does not exist.')
        sys.exit()
    
    df_subsample = df_all[(df_all['polarity'] == polarity) & (df_all['truepdg'] == pdg)]
    nevts = nevents
    nevts = min(nevts, len(df_subsample))
    df_subsample = df_subsample.sample(n=nevts, random_state=0)
    df_subsample.to_hdf(subsample_pn, tablename, complevel=9, complib='bzip2')

if __name__ == '__main__':

    # command line argument
    parser = argparse.ArgumentParser(description='Command line argument parser')
    parser.add_argument('-n', '--nevents', type=int, default=100000, help='Number of events used for closeness tests.')
    args = parser.parse_args()
    nevents = args.nevents

    # If subsample file does not exist, create one.
    if not os.path.isfile(subsample_pn):
        # retrieve data frame
        df_all = pd.read_hdf('../hdf5/wrong_sign_bdt_variables.h5', 'fhc_rhc')
        make_subsample(1, 14, 'fhc_numu')
        make_subsample(1, -14, 'fhc_numubar')
        make_subsample(-1, 14, 'rhc_numu')
        make_subsample(-1, -14, 'rhc_numubar')
    
    varnames = ['hade','hadnhit','cosnumi','cce','mue','nmichels','numuhadefrac','recow','bpfbestmuondedxll',
                 'bpfbestmuonchi2t','orphcale', 'nhit', 'hadcalcnhit', 'muonhit']
    nutypes = ['numu','numubar']
    scores = {'numu': dict(), 'numubar': dict()}
    # Calculate closeness scores for variables.
    for nutype in nutypes:
        for varname in varnames:
            # scores[nutype][varname] = closeness_score(nutype, varname, scipy.stats.wasserstein_distance)
            scores[nutype][varname] = closeness_score(nutype, varname, scipy.stats.energy_distance)
            print('{} {} {}'.format(nutype, varname, scores[nutype][varname]))