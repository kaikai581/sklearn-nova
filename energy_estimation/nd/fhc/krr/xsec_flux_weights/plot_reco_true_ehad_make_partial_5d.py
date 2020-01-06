#!/usr/bin/env python

"""
Plot official reco and true ehad histograms with official weights.
This plot can be compared with p. 32(42) of this document:
https://nova-docdbcert.fnal.gov/cgi-bin/cert/RetrieveFile?docid=27221&filename=2018-02-28_prod4_nd_fhc_energy_estimator.pdf&version=1

To avoid memory error, do in batch:
$ for i in `seq {0,99}`; do time ./plot_reco_true_ehad_make_partial_5d.py -s 100 -o ${i}; done
"""

from __future__ import print_function
print(__doc__)

from root_numpy import root2array
from sklearn.externals import joblib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def make_prediction(pscaler, pmodel, X):
  print('start making prediction...')
  # load scaler
  scaler = joblib.load(pscaler)
  # load model
  krr = joblib.load(pmodel)
  # make prediction
  return krr.predict(scaler.transform(X))


def make_plot(apar, gpar, scaledown, offset, modelstep, modeloffset):
  # retrieve training data and official reco hadronic energy for comparison
  #~ cut = 'mustopz<1275&&isnumucc==1&&trueenu-recoemu>1e-3'
  cut = 'isnumucc==1'
  # ~ X = root2array('../training_data.root',
                 # ~ branches='calehad',
                 # ~ selection=cut,
                 # ~ step=scaledown, start=offset).reshape(-1,1)
  recoemu_official = root2array('../training_data.root', branches='recoemu',
                                selection=cut,
                                step=scaledown, start=offset)
  trueenu = root2array('../training_data.root', branches='trueenu',
                       selection=cut,
                       step=scaledown, start=offset)
  y = trueenu - recoemu_official
  yoff = root2array('../training_data.root', branches='recoehad',
                    selection=cut,
                    step=scaledown, start=offset)
  offwei = root2array('../training_data.root', branches='weight',
                      selection=cut,
                      step=scaledown, start=offset)
  X5d = root2array('../training_data.root',
                   branches=['calehad', 'cvnpi0', 'cvnchargedpion', 'cvnneutron', 'cvnproton'],
                   selection=cut,
                   step=scaledown, start=offset)
  X5d = X5d.view(np.float32).reshape(X5d.shape + (-1,))
  
  # interaction mode
  intmode = root2array('../training_data.root', branches='intmode',
                      selection=cut,
                      step=scaledown, start=offset)
  
  # plot true ehad with reco overlaid
  n, bins, patches = plt.hist(y, bins=np.linspace(-1,10,550), histtype='step', weights=offwei)
  plt.hist(yoff, bins=bins, histtype='step', color='red', weights=offwei)
  
  # if a fitted model exists, overlay my predicted histogram as well
  y_pred = None
  pscaler = 'models/5d/sample_weighted_hadronic_5d_scaler_a{}g{}step{}offset{}.pkl'.format(apar, gpar, modelstep, modeloffset)
  pmodel = 'models/5d/sample_weighted_hadronic_5d_energy_estimator_a{}g{}step{}offset{}.pkl'.format(apar, gpar, modelstep, modeloffset)
  if os.path.isfile(pscaler):
    if os.path.isfile(pmodel):
      y_pred = make_prediction(pscaler, pmodel, X5d)
      plt.hist(y_pred, bins=bins, histtype='step', color='green', weights=offwei)
  
  # ~ plt.xlim(0,2)
  # ~ plt.show()
  # save the three histograms
  os.system('mkdir -p partial_output/5d')
  joblib.dump(y, 'partial_output/5d/sample_weighted_hadronic_true_a{}g{}step{}offset{}.pkl'.format(apar, gpar, scaledown, offset))
  joblib.dump(yoff, 'partial_output/5d/sample_weighted_hadronic_official_a{}g{}step{}offset{}.pkl'.format(apar, gpar, scaledown, offset))
  joblib.dump(y_pred, 'partial_output/5d/sample_weighted_hadronic_krr_a{}g{}step{}offset{}.pkl'.format(apar, gpar, scaledown, offset))
  joblib.dump(offwei, 'partial_output/5d/sample_weighted_hadronic_weight_a{}g{}step{}offset{}.pkl'.format(apar, gpar, scaledown, offset))
  joblib.dump(intmode, 'partial_output/5d/sample_weighted_intmode_a{}g{}step{}offset{}.pkl'.format(apar, gpar, scaledown, offset))

if __name__ == '__main__':
  
  # parse command line arguments
  parser = argparse.ArgumentParser(description='Train a hadronic energy KRR with specified parameters and weighted sample.')
  parser.add_argument('-a','--alpha',type=str,default='0.01')
  parser.add_argument('-g','--gamma',type=str,default='None')
  parser.add_argument('-s','--step',type=int,default='1')
  parser.add_argument('-m','--model_step',type=int,default='100')
  parser.add_argument('-o','--offset',type=int,default='0')
  parser.add_argument('-f','--model_offset',type=int,default='0')
  args = parser.parse_args()
  
  # specified parameters
  apar = args.alpha
  gpar = args.gamma
  scaledown = args.step
  offset = args.offset
  modelstep = args.model_step
  modeloffset = args.model_offset

  # plot true, official reco, and my reco ehad
  make_plot(apar, gpar, scaledown, offset, modelstep, modeloffset)
