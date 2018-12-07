#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os, sys
import pandas as pd

# command line argument
parser = argparse.ArgumentParser(description='Command line argument parser for this script.')
parser.add_argument('-c','--horn_current',default='rhc')
args = parser.parse_args()
polarity = args.horn_current

# horn current safeguard
if polarity not in ['fhc', 'rhc']:
    print('Horn current not valid: {}'.format(polarity))
    sys.exit()

data_dir = '/pnfs/nova/scratch/users/slin/cafana_on_grid_output/dropbox'
df_list = []

col_names = ['run','subrun','cycle','evt','subevt','hade','hadnhit','cosnumi',
'cce','mue','nmichels','numuhadefrac','recow','bpfbestmuondedxll',
'bpfbestmuonchi2t','muonid','remid','xseccvwgt2018','ppfxfluxcvwgt','trueywgt',
'truey','antinumubdt','truepdg']

fn_wild = 'nd_fhc_wrong_sign_bdt_variables_mc*.txt'
if polarity == 'rhc':
    fn_wild = 'nd_rhc_wrong_sign_reweight_y*.txt'

for fn in glob.glob('{}/{}'.format(data_dir, fn_wild)):
    df_list.append(pd.read_csv(fn, index_col=False, names=col_names, sep=' ', header=None))

comb_df = pd.concat(df_list, ignore_index=True)
comb_df.to_hdf('../data/{}_wrong_sign_bdt_variables.h5'.format(polarity), '{}_dataframe'.format(polarity), complevel=9, complib='bzip2')

print(comb_df.head())
