//Run using: root -b -q -l MakeProfileMuEND.C

//This script is where I make the graph from 2d plots that I shall fit. I also then
//make nice versions of the graph and 2d plot with fit lines on it.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstring>
#include "TFile.h"
#include "TH2D.h"
#include "TLine.h"
#include "TObject.h"
#include "TArray.h"
#include "TArrayF.h"
#include "TGraphAsymmErrors.h"


void MakeProfileMuEND()
{
  std::cout<<"Hiya!"<<std::endl;

  static const int nn = 150; //Number of points on graph - should be equal to or smaller than number of bins in x
  float xxx[nn];
  float exxl[nn]; 
  float exxh[nn];
  float yyx[nn];
  float eyxl[nn];
  float eyxh[nn];
  
  gStyle->SetOptStat(0);
  TFile* file = new TFile("./2DPlotsForFittingND.root","READ");

  TH1D* chiValues   = new TH1D("chiValue",";Chi squared per NDF;Fits",150, 0.0, 5.0);

  TH2D* hist  = (TH2D*)file->Get("MuonE_hist_active");

  TCanvas* c7 = new TCanvas("c7","c7");
  hist->GetXaxis()->CenterTitle();
  hist->GetYaxis()->CenterTitle();
  hist->Draw("colz");
  gPad->Update();
  gPad->SetRightMargin(0.13);
  TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c7->Print("muonMother.pdf");
  
  for(int iw = 1; iw <= hist->GetNbinsX(); ++iw){
 
    // make projection in Y view
    TH1D* py       = hist->ProjectionY("py",iw,iw,"");
    int maxbin     = py->GetMaximumBin();        //Bin in slice with most entries
    double maxcont = py->GetBinContent(maxbin);  //Height of the bin with most entries

    if (maxcont == 0){continue; }    //If no entries in the slice, don't make a graph point

    double totalEntries = py->GetEntries();
    if (totalEntries < 30){continue; }    //If not many entries in the slice, don't make a graph point

    double modex = py->GetBinCenter(maxbin); //Center location of bin with most entries

    //Find the location of width at half max
    int yyhighbin = maxbin+1;
    int yylowbin  = maxbin-1;
    // find upper bin of half max
    for(int ihigh = maxbin+1; ihigh <= nn; ++ihigh){
      yyhighbin = ihigh;
      if(py->GetBinContent(yyhighbin) <= maxcont/2.0){ break; }
    }
    // find lower bin of half max
    for(int ilow = maxbin-1; ilow > 0; --ilow){
      yylowbin = ilow;
      if(py->GetBinContent(yylowbin) <= maxcont/2.0){ break; }
    }
    
    double multiplier = 1.5; //This defines how many times wider than width at half max I want to fit over
    int veryHighBin = (yyhighbin - maxbin)*multiplier+maxbin;
    int veryLowBin  = maxbin - multiplier*(maxbin-yylowbin);
    
    double lowSide  = modex-multiplier*(modex-py->GetBinCenter(yylowbin));
    double highSide = multiplier*(py->GetBinCenter(yyhighbin)-modex)+modex;
    
    xxx[iw-1] = (hist->GetXaxis()->GetBinCenter(iw)); //Center of x bin position set as graph point
    
    TF1 *f1 = new TF1("customGaus","gaus",0,5);
    py->Fit("customGaus","OQ","",lowSide,highSide); //O and Q mean do not plot result, quiet mode 
    double mean = f1->GetParameter(1);
    double error = f1->GetParError(1);
    double chi = f1->GetChisquare();
    double ndf = f1->GetNDF();

    if (ndf > 0) {chiValues->Fill(chi/ndf);}
    else if (chi > 0.000001){chiValues->Fill(chi/ndf);}

    if ((chi/ndf) > 10) {std::cout<<" We have chi / ndf of : "<<chi/ndf<<" in the bin: "<<iw<<std::endl;}

    yyx[iw-1] = mean; //Set mean of gaussian fit as y location of graph point

    //For now, setting the x error bars as zero. Can change if want to denote variable bin widths.
    exxl[iw-1] = 0; //effVwx->GetBinWidth(iw)/2.0;
    exxh[iw-1] = 0; //effVwx->GetBinWidth(iw)/2.0;
    //Setting the error on the mean reported by the fit as error in y
    eyxl[iw-1] = error;
    eyxh[iw-1] = error;
 
  }//End of loop over bins in 2D histogram

  //Making graph from points
  TGraphAsymmErrors* customProfile = new TGraphAsymmErrors(nn,xxx,yyx,exxl,exxh,eyxl,eyxh);
  
  customProfile->GetXaxis()->SetNoExponent(kTRUE);
  customProfile->GetXaxis()->SetTitle("Reco muon track length (m)");
  customProfile->GetYaxis()->SetTitle("True muon energy (GeV)");
  customProfile->GetXaxis()->CenterTitle();
  customProfile->GetYaxis()->CenterTitle();
  customProfile->SetMarkerStyle(6);
  customProfile->GetXaxis()->SetLimits(0,15);
  customProfile->GetHistogram()->SetMaximum(5);
  customProfile->GetHistogram()->SetMinimum(0);
  customProfile->SetTitle("");
  
  //Overlays 2d plot and graph
  TCanvas* c1 = new TCanvas("c1","c1");
  customProfile->SetMarkerColor(18); 
  customProfile->SetLineColor(18);
  hist->Draw("colz"); 
  customProfile->Draw("P"); 
  gPad->Update();
  gPad->SetRightMargin(0.13);
  //TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  //palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c1->Print("muonMotherAndGraph.pdf");

  //Overlays 2d loqz plot and graph
  TCanvas* c2 = new TCanvas("c2","c2");
  c2->SetLogz();
  customProfile->SetMarkerColor(13); 
  customProfile->SetLineColor(13);
  hist->Draw("colz"); 
  customProfile->Draw("P"); 
  gPad->Update();
  gPad->SetRightMargin(0.13);
  //TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  //palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c2->Print("muonMotherLogzAndGraph.pdf");

  //Just the graph
  TCanvas* c3 = new TCanvas("c3","c3");
  customProfile->SetMarkerColor(1); 
  customProfile->SetLineColor(1);
  customProfile->Draw("AP"); 
  gPad->Update();
  gPad->SetRightMargin(0.13);
  //TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  //palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c3->Print("muonGraph.pdf");

  //Drawing fit over the 2d plot and graph


  //New tuning values rounded for errors
  double stitch1 = 3.34;     // m
  double stitch2 = 5.39;     // m
  double stitch3 = 10.64;     // m
  double offset  = 0.1503;    // GeV
  double slope1  = 0.1910;  // GeV/m
  double slope2  = 0.1987;  // GeV/m
  double slope3  = 0.2039;  // GeV/m
  double slope4  = 0.2159;  // GeV/m
  
  TLine fitLine1 = TLine(0,offset,stitch1,slope1*stitch1+offset);
  fitLine1.SetLineColor(2);
  
  TLine fitLine2 = TLine(stitch1,slope2*stitch1+(slope1-slope2)*stitch1+offset,stitch2,slope2*stitch2+(slope1-slope2)*stitch1+offset);
  fitLine2.SetLineColor(2);
  
  TLine fitLine3 = TLine(stitch2,slope3*stitch2+(slope1-slope2)*stitch1+(slope2-slope3)*stitch2+offset,stitch3,slope3*stitch3+(slope1-slope2)*stitch1+(slope2-slope3)*stitch2+offset);
  fitLine3.SetLineColor(2);

  TLine fitLine4 = TLine(stitch3,slope4*stitch3+(slope1-slope2)*stitch1+(slope2-slope3)*stitch2+(slope3-slope4)*stitch3+offset,1500,slope4*1500+(slope1-slope2)*stitch1+(slope2-slope3)*stitch2+(slope3-slope4)*stitch3+offset);
  fitLine4.SetLineColor(2);
  
  TLine stitchLine1 = TLine(stitch1, 0, stitch1,5.0);
  TLine stitchLine2 = TLine(stitch2, 0, stitch2,5.0);
  TLine stitchLine3 = TLine(stitch3, 0, stitch3,5.0);
  stitchLine1.SetLineColor(2);
  stitchLine1.SetLineStyle(7);
  stitchLine2.SetLineColor(2);
  stitchLine2.SetLineStyle(7);
  stitchLine3.SetLineColor(2);
  stitchLine3.SetLineStyle(7);

  //Graph with fitting line
  TCanvas* c4 = new TCanvas("c4","c4");
  customProfile->Draw("AP"); 
  fitLine1.Draw();
  fitLine2.Draw();
  fitLine3.Draw();
  fitLine4.Draw();
  stitchLine1.Draw();
  stitchLine2.Draw();
  stitchLine3.Draw();
  gPad->Update();
  gPad->SetRightMargin(0.13);
  //TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  //palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c4->Print("fittingLineOnGraph.pdf");

  //2d plot with fitting line
  TCanvas* c5 = new TCanvas("c5","c5");
  hist->Draw("colz");
  fitLine1.Draw();
  fitLine2.Draw();
  fitLine3.Draw();
  fitLine4.Draw();
  stitchLine1.Draw();
  stitchLine2.Draw();
  stitchLine3.Draw();
  gPad->Update();
  gPad->SetRightMargin(0.13);
  //TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  //palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c5->Print("fittingLineOnMother.pdf");

  //2d logz plot with fitting line
  TCanvas* c8 = new TCanvas("c8","c8");
  c8->SetLogz();
  hist->Draw("colz");
  fitLine1.Draw();
  fitLine2.Draw();
  fitLine3.Draw();
  fitLine4.Draw();
  stitchLine1.Draw();
  stitchLine2.Draw();
  stitchLine3.Draw();
  gPad->Update();
  gPad->SetRightMargin(0.13);
  //TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  //palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c8->Print("fittingLineOnMotherLogz.pdf");

  //Plot of chi squared per ndf
  TCanvas* c6 = new TCanvas("c6","c6");
  chiValues->Draw(); 
  c6->Print("graphChiSquaredperNDF.pdf");

  TFile f("muonProfile.root","RECREATE");
  customProfile->Write("1");
  chiValues->Write();

}
