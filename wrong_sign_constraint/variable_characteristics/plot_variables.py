#!/usr/bin/env python

from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def plot_one_var(var_idx):
    vn = var_names[var_idx]
    var_data = dict()
    for pdg in [-14,-12,12,14]:
        var_data[pdg] = []
    # for index, row in df.iterrows():
    #     var_data[row['truepdg']].append(row[vn])
    # print(var_data[-14], var_data[14])
    # h1 = df[df['truepdg'] == 14].hist(column=vn, grid=False, bins=40)
    # h2 = df[df['truepdg'] == -14].hist(column=vn, grid=False, bins=40)
    numu_label = r'$\nu_\mu$'
    numubar_label = r'$\bar{\nu}_\mu$'
    if is_blog:
        numubar_label = 'component 1'
        numu_label = 'component 2'
    h1, bins, patches = ax[var_idx].hist(df[vn][df['truepdg'] == -14], bins=50, alpha=0.9, label=numubar_label, histtype='step')
    ax[var_idx].hist(df[vn][df['truepdg'] == 14], bins=bins, alpha=0.5, label=numu_label)

    # manually change legend location
    leg_loc = 'best'
    if vn in ['bpfbestmuondedxll', 'cosnumi']:
        leg_loc = 'upper left'
    if is_blog:
        ax[var_idx].legend(loc=leg_loc, prop={'size': 6})
    else:
        ax[var_idx].legend(loc=leg_loc)

    # if in blog mode, use dummy x labels
    if(not is_blog):
        ax[var_idx].set_xlabel(vn)
    else:
        ax[var_idx].set_xlabel('feature '+str(var_idx+1))

# command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--blog', dest='blog', action='store_true')
parser.set_defaults(blog=False)
args = parser.parse_args()
is_blog = args.blog

df = pd.read_hdf('../data/wrong_sign_bdt_variables.h5', 'rhc_dataframe')
# remove records with pdg value -5, which is introduced by the CAFAna Var kTruePDG
# df = df[(df['truepdg'] != -5) & (df['cce'] < 15)].truncate(after=1000)
df = df[(df['truepdg'] != -5) & (df['cce'] < 15) & (df['bpfbestmuonchi2t'] < 20) & (df['bpfbestmuondedxll'] > -8) & (df['hadnhit'] < 250) & (df['recow'] > 0) & (df['hade'] < 5)]
# have to convert integer values to float to make reasonable histograms
df['hadnhit'] = pd.to_numeric(df['hadnhit'], errors='coerce')
df['nmichels'] = pd.to_numeric(df['nmichels'], errors='coerce')
print(df.shape)
var_names = ['hade','hadnhit','cosnumi', 'cce','mue','nmichels','numuhadefrac','recow','bpfbestmuondedxll', 'bpfbestmuonchi2t']
# var_names = ['bpfbestmuondedxll']

# plot container
axes_collection = dict()
for vn in var_names:
    axes_collection[vn] = []

# make figure
fig, ax = plt.subplots(2, 5, figsize=(12,5))
ax = ax.ravel()

# loop over all variables
for i in range(len(var_names)):
    # axes_collection[vn].append(plot_one_var(vn))
    plot_one_var(i)

plt.tight_layout()
# plt.show()
if is_blog:
    fig.savefig('sig_bkg_dists_censored.pdf')
else:
    fig.savefig('sig_bkg_dists.pdf')