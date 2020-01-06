#!/usr/bin/env python

from __future__ import print_function

from ROOT import *
import glob

toutf = TFile('grid_output.root', 'recreate')
tr = TTree('tr', 'data for energy reconstruction')
for name in glob.glob('grid_output/*.txt'):
  tr.ReadFile(name, 'run/I:subrun/I:cycle/I:evt/I:subevt/I:trueenu/F:trueemu/F:recotrklenact/F:recotrklencat/F:mustopz/F:trueehad/F:calehad/F:recoq2/F:cvnchargedpion/F:npng/I', ' ')
tr.Write()
