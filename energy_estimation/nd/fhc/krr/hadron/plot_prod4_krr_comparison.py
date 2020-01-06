#!/usr/bin/env python

from __future__ import print_function
from ROOT import *
import argparse
import os

def max_histogram(hlist):
  hmax = hlist[0]
  cmax = 0
  for hist in hlist:
    curmax = hist.GetMaximum()
    if curmax > cmax:
      cmax = curmax
      hmax = hist
  return hmax

# parse command line arguments
parser = argparse.ArgumentParser(description='Train a hadronic energy KRR with specified parameters!')
parser.add_argument('-f','--input_file',type=str,default='output_root_files/1d/resolution_1d_a0.5g0.5step200offset0.root')
args = parser.parse_args()

# specified parameters
fn = args.input_file

f = TFile(fn)
tr = f.Get('tr')
canv = TCanvas('c1','c1',1200,500)
canv.Divide(2,1)

# energy spectra
canv.cd(1)
hsp0 = TH1F('hsp0','',50,0,2)
hsp1 = TH1F('hsp1','hadronic energy spectra',50,0,2)
hsp2 = TH1F('hsp2','',50,0,2)
tr.Project('hsp0','trueehad')
tr.Project('hsp1','offehad')
tr.Project('hsp2','krrehad')
hsp1.SetLineWidth(2)
hsp1.SetLineColor(kRed)
hsp1.SetStats(0)
hsp1.GetXaxis().SetTitle('hadronic energy (GeV)')
hsp1.Draw()
hsp0.SetLineColor(kYellow)
hsp0.SetFillColor(kYellow)
hsp0.Draw('same')
hsp2.SetLineWidth(2)
hsp2.SetLineColor(kBlue)
hsp2.Draw('same')
hsp1.Draw('same')
# legend
l1 = TLegend(.7,.7,.88,.88)
l1.SetBorderSize(0)
l1.AddEntry(hsp0,'true','F')
l1.AddEntry(hsp1,'prod4','L')
l1.AddEntry(hsp2,'KRR','L')
l1.Draw()

gStyle.SetOptStat(110001111)
# energy resolution
canv.cd(2)
prod4 = TH1F('prod4','hadronic energy resolution',50,-1,1)
krr = TH1F('krr','hadronic energy resolution',50,-1,1)
tr.Project('prod4','roff','roff>-1&&roff<1')
tr.Project('krr','rest','rest>-1&&rest<1')
prod4.SetLineWidth(2)
prod4.SetLineColor(kRed)
prod4.GetXaxis().SetTitle('(reco-true)/true')
prod4.Draw()
gPad.Update()
tps0 = prod4.FindObject('stats')
tps0.SetTextColor(kRed)
tps0.SetLineColor(kRed)
X1 = tps0.GetX1NDC()
X2 = tps0.GetX2NDC()
Y1 = tps0.GetY1NDC()
Y2 = tps0.GetY2NDC()
krr.SetLineWidth(2)
krr.SetLineColor(kBlue)
krr.Draw()
gPad.Update()
tps1 = krr.FindObject('stats')
tps1.SetTextColor(kBlue);
tps1.SetLineColor(kBlue);
tps1.SetX1NDC(X1);
tps1.SetX2NDC(X2);
tps1.SetY1NDC(Y1-(Y2-Y1));
tps1.SetY2NDC(Y1);
hmax = max_histogram([prod4, krr])
hmax.GetXaxis().SetTitle('(reco-true)/true')
hmax.Draw()
prod4.Draw('same')
krr.Draw('same')
tps0.Draw('same')
tps1.Draw('same')

fign = 'plots' + fn.rstrip('.root').lstrip('output_root_files') + '.pdf'
figpath = os.path.dirname(fign)
os.system('mkdir -p ' + figpath)
canv.SaveAs(fign)

#~ raw_input('press enter')
