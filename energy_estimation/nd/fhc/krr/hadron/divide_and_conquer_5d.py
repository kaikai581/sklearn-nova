#!/usr/bin/env python

from __future__ import print_function

from estimator_5d import fit_with_subdata

import argparse
import timeit

# parse command line arguments
parser = argparse.ArgumentParser(description='Train a hadronic energy KRR with specified parameters!')
parser.add_argument('-a','--alpha',type=str,default='0.01')
parser.add_argument('-s','--step',type=int,default='200')
args = parser.parse_args()

# specified parameters
apar = args.alpha
scaledown = args.step

for i in range(scaledown):
  start_time = timeit.default_timer()
  fit_with_subdata(apar, 1, scaledown, i)
  print('dataset {} complete'.format(i))
  print('time spent {}'.format(timeit.default_timer() - start_time))
