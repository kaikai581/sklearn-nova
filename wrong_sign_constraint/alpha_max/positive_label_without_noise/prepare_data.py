#!/usr/bin/env python

from __future__ import print_function
import pandas as pd

# split dataframe into train and test
def train_test_split(df):
    df_train = df.sample(n=1000000,random_state=0)
    df_test = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    return df_train, df_test

# box cut for cleaning up obvious reconstruction failure
def variable_cleanup(df):
    df = df[(df['truepdg'] != -5) & (df['cce'] < 15) & (df['bpfbestmuonchi2t'] < 20) & (df['bpfbestmuondedxll'] > -8) & (df['hadnhit'] < 250) & (df['recow'] > 0) & (df['hade'] < 5) & (abs(df['truepdg']) == 14)]
    # convert integer values into float
    df.loc[:,'hadnhit'] = df['hadnhit'].astype(float)
    df.loc[:,'nmichels'] = df['nmichels'].astype(float)
    # after all the cuts, use reset_index to rebuild the row index
    df = df.reset_index(drop=True)
    # features included
    fnames = ['hade','hadnhit','cosnumi', 'cce','mue','nmichels','numuhadefrac','recow','bpfbestmuondedxll','bpfbestmuonchi2t','truepdg']
    df = df.loc[:, fnames]
    return df

fhc_in = '../../data/fhc_wrong_sign_bdt_variables.h5'
rhc_in = '../../data/rhc_wrong_sign_bdt_variables.h5'

# read in the dataframes
df_fhc = pd.read_hdf(fhc_in)
df_rhc = pd.read_hdf(rhc_in)
# clean up variables
df_fhc = variable_cleanup(df_fhc)
df_rhc = variable_cleanup(df_rhc)

# Now, split data into four tables.
# Two training data and two test data.
# First, remove label noise: allow only numu in fhc.
df_fhc = df_fhc[df_fhc['truepdg'] == 14].reset_index(drop=True)

# take one million from each dataset and remove them from the original tables
df_fhc_train, df_fhc_test = train_test_split(df_fhc)
df_rhc_train, df_rhc_test = train_test_split(df_rhc)

# save the four dataframes to file
outfn = 'train_test_split.h5'
df_fhc_train.to_hdf(outfn, 'fhc_train', complevel=9, complib='bzip2')
df_fhc_test.to_hdf(outfn, 'fhc_test', complevel=9, complib='bzip2')
df_rhc_train.to_hdf(outfn, 'rhc_train', complevel=9, complib='bzip2')
df_rhc_test.to_hdf(outfn, 'rhc_test', complevel=9, complib='bzip2')