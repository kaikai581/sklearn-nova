#!/usr/bin/env python

from __future__ import print_function
from ROOT import *

tinf = TFile('infile.root')
tr = tinf.Get('tr')

mode_val = [0,1,2,3,10]
mode_label = ['QE','Res','DIS','Coh','MEC']

# For each interaction mode, make the true hadronic energy vs. visible hadronic
# energy plot.
toutf = TFile('output.root','recreate')

# first, save the overall plot
h = TH2F('hAll','All',100,0,2,100,0,5)
tr.Project('hAll','trueenu-recoemu:calehad','mustopz<1275&&isnumucc==1&&calehad>0')
h.Write()

# second, mode by mode
for i in range(len(mode_val)):
  hname = 'h_'+mode_label[i]
  h = TH2F(hname,mode_label[i],100,0,2,100,0,5)
  tr.Project(hname,'trueenu-recoemu:calehad','mustopz<1275&&isnumucc==1&&calehad>0&&intmode=='+str(mode_val[i]))
  h.Write()

# third, 1d histograms in thin slices
slc = [
'calehad>0&&calehad<.1',
'calehad>1&&calehad<1.1'
]
slc_label = [
'0<visHadE<0.1',
'1<visHadE<1.1'
]
for j in range(len(slc)):
  for i in range(len(mode_val)):
    hname = 'h_'+mode_label[i]+'_'+str(j)
    h = TH1F(hname,mode_label[i]+' '+slc_label[j],50,0,5)
    tr.Project(hname,'trueenu-recoemu','mustopz<1275&&isnumucc==1&&calehad>0&&intmode=={}&&{}'.format(mode_val[i],slc[j]))
    h.Write()

# forth, final state scores broken down by modes
fsp = [
'cvnpi0',
'cvnchargedpion',
'cvnneutron',
'cvnproton'
]
for j in range(len(fsp)):
  for i in range(len(mode_val)):
    hname = 'h_'+mode_label[i]+'_'+fsp[j]
    h = TH1F(hname,mode_label[i]+' '+fsp[j],100,0,1)
    tr.Project(hname,fsp[j],'mustopz<1275&&isnumucc==1&&calehad>0&&intmode=={}'.format(mode_val[i]))
    h.Write()

toutf.Close()
