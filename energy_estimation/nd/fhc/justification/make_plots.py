#!/usr/bin/env python

from __future__ import print_function
from ROOT import *
import os

def plot2d(nc,mode):
  c2.cd(nc)
  gPad.SetLogz(1)
  h2d[nc] = tinf.Get(mode)
  h2d[nc].SetStats(0)
  h2d[nc].GetXaxis().SetTitle('visible hadronic energy (GeV)')
  h2d[nc].GetXaxis().CenterTitle()
  h2d[nc].GetYaxis().SetTitle('true E_{#nu} - reco E_{#mu} (GeV)')
  h2d[nc].GetYaxis().CenterTitle()
  h2d[nc].Draw('colz')

def plot_thin_slice(nc,slc):
  c3.cd(nc)
  gPad.SetLogy(1)
  modes = ['QE','Res','DIS','Coh','MEC']
  colors = [kBlue, kRed, kGreen, kMagenta, kYellow]
  umin = 0
  umax = 5
  if slc == 0: umax = 1.5
  for i in range(len(modes)):
    mode = 'h_' + modes[i] + '_' + str(slc)
    hmode[slc][mode] = tinf.Get(mode)
    hmode[slc][mode].SetStats(0)
    hmode[slc][mode].SetTitle('{} GeV < Ehadvis < {} GeV'.format(slc,slc+0.1))
    hmode[slc][mode].GetXaxis().SetTitle('true E_{#nu} - reco E_{#mu} (GeV)')
    hmode[slc][mode].GetXaxis().CenterTitle()
    hmode[slc][mode].SetLineColor(colors[i])
    hmode[slc][mode].SetLineWidth(2)
    hmode[slc][mode].GetXaxis().SetRangeUser(umin, umax)
  dominant_mode = 'QE'
  if slc == 1: dominant_mode = 'DIS'
  rest_modes = list(modes)
  rest_modes.remove(dominant_mode)
  dominant_mode = 'h_' + dominant_mode + '_' + str(slc)
  hmode[slc][dominant_mode].Draw()
  for mode in rest_modes:
    mode = 'h_' + mode + '_' + str(slc)
    hmode[slc][mode].Draw('same')
  # deal with legends
  legslc[slc] = TLegend(.7,.6,.88,.88)
  legslc[slc].SetBorderSize(0)
  for mode in modes:
    hname = 'h_' + mode + '_' + str(slc)
    legslc[slc].AddEntry(hmode[slc][hname],mode,'L')
  legslc[slc].Draw()

def plot_fs_score(varidx):
  c4.cd(varidx+1)
  modes = ['QE','Res','DIS','Coh','MEC']
  colors = [kBlue, kRed, kGreen, kMagenta, kYellow]
  var = fs_id[varidx]
  hs[var] = THStack(fs_id[varidx],'')
  for i in range(len(modes)):
    hname = 'h_{}_{}'.format(modes[i], var)
    hfsscore[var][modes[i]] = tinf.Get(hname)
    hfsscore[var][modes[i]].SetStats(0)
    hfsscore[var][modes[i]].GetXaxis().SetTitle(fs_label[varidx])
    hfsscore[var][modes[i]].GetXaxis().CenterTitle()
    hfsscore[var][modes[i]].SetLineColor(colors[i])
    hfsscore[var][modes[i]].SetLineWidth(2)
    hs[var].Add(hfsscore[var][modes[i]])
  hs[var].Draw("nostack")
  hs[var].GetXaxis().SetTitle(fs_label[varidx])
  hs[var].GetXaxis().CenterTitle()
  legslc[0].Draw()

# root file with all histograms
tinf = TFile('output.root')

# make an output directory
os.system('mkdir -p plots')

# make the plot of overall regression plot
c1 = TCanvas('c1','c1',500,500)
hall = tinf.Get('hAll')
gPad.SetLogz(1)
hall.SetStats(0)
hall.GetXaxis().SetTitle('visible hadronic energy (GeV)')
hall.GetXaxis().CenterTitle()
hall.GetYaxis().SetTitle('true E_{#nu} - reco E_{#mu} (GeV)')
hall.GetYaxis().CenterTitle()
hall.Draw('colz')
c1.SaveAs('plots/2d_all.pdf')

# make the plot of regression plots broken down with interaction mode
c2 = TCanvas('c2','c2',900,600)
c2.Divide(3,2)
c2.cd(1)
gPad.SetLogz(1)
hall.Draw('colz')
h2d = dict()
plot2d(2,'h_QE')
plot2d(3,'h_Res')
plot2d(4,'h_DIS')
plot2d(5,'h_Coh')
plot2d(6,'h_MEC')
c2.SaveAs('plots/2d_all_modes.pdf')

# make the plot of thin slice
c3 = TCanvas('c3','c3',800,400)
c3.Divide(2,1)
hmode = dict()
hmode[0] = dict()
hmode[1] = dict()
legslc = dict()
legslc[0] = dict()
legslc[1] = dict()
plot_thin_slice(1,0)
plot_thin_slice(2,1)
c3.SaveAs('plots/thin_slices.pdf')

# make the plot of final state particle scores
c4 = TCanvas('c4','c4',500,500)
c4.Divide(2,2)
hfsscore = dict()
hs = dict()
fs_id = [
'cvnpi0',
'cvnchargedpion',
'cvnneutron',
'cvnproton'
]
fs_label = [
'CVN #pi^{0} score',
'CVN #pi^{#pm} score',
'CVN neutron score',
'CVN proton score'
]
for i in range(len(fs_id)):
  hfsscore[fs_id[i]] = dict()
  plot_fs_score(i)
c4.SaveAs('plots/fs_particle_scores.pdf')
