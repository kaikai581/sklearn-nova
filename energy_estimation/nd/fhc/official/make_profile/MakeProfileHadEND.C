//Run using: root -b -q -l MakeProfileHadEND.C

//This script is where I make the graph from 2d plots that I shall fit. I also then
//make nice versions of the graph and 2d plot with fit lines on it.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstring>
#include "TFile.h"
#include "TLine.h"
#include "TH2D.h"
#include "TObject.h"
#include "TArray.h"
#include "TArrayF.h"
#include "TGraphAsymmErrors.h"


void MakeProfileHadEND()
{
  std::cout<<"Hiya!"<<std::endl;

  static const int nn = 115; //Number of points on graph - should be equal to or smaller than number of bins in x
  // static const int nn = 57; //Number of points on graph - should be equal to or smaller than number of bins in x
  float xxx[nn];
  float exxl[nn]; 
  float exxh[nn];
  float yyx[nn];
  float eyxl[nn];
  float eyxh[nn];
  
  gStyle->SetOptStat(0);
  TFile* file = new TFile("./2DPlotsForFittingND.root","READ");

  TH1D* chiValues   = new TH1D("chiValue",";Chi squared per NDF;Fits",100, 0.0, 10.0);

  TH2D* hist  = (TH2D*)file->Get("HadE_hist_DIS");

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
  c7->Print("hadronccMother.pdf");

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
    if (iw>100){multiplier = 10.0;}
    if (iw<3){multiplier = 0.5;}
    if ((iw<20)&&(iw>2)){multiplier = 1.0;}
    int veryHighBin = (yyhighbin - maxbin)*multiplier+maxbin;
    int veryLowBin  = maxbin - multiplier*(maxbin-yylowbin);
    
    double lowSide  = modex-multiplier*(modex-py->GetBinCenter(yylowbin));
    double highSide = multiplier*(py->GetBinCenter(yyhighbin)-modex)+modex;
    
    xxx[iw-1] = (hist->GetXaxis()->GetBinCenter(iw)); //Center of x bin position set as graph point
    
    TF1 *f1 = new TF1("customGaus","gaus",0,5);
    py->Fit("customGaus","OQ","",lowSide,highSide); //O and Q mean: do not plot result, quiet mode 
    double mean = f1->GetParameter(1);
    double error = f1->GetParError(1);
    double chi = f1->GetChisquare();
    double ndf = f1->GetNDF();

    if (ndf > 0) {chiValues->Fill(chi/ndf);}
    else if (chi > 0.000001){chiValues->Fill(chi/ndf);}

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
  customProfile->GetXaxis()->SetTitle("Visible Hadronic E (GeV)");
  customProfile->GetYaxis()->SetTitle("True Hadronic E (GeV)");
  customProfile->GetXaxis()->CenterTitle();
  customProfile->GetYaxis()->CenterTitle();
  customProfile->SetMarkerStyle(6);
  customProfile->GetXaxis()->SetLimits(0,2.0);
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
  c1->Print("hadronccMotherAndGraph.pdf");

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
  c2->Print("hadronccMotherLogzAndGraph.pdf");

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
  c3->Print("hadronccGraph.pdf");

  //Drawing fit over the 2d plot and graph

  //Old Tune Values
  //double stitch1 = 0.225007;     // GeV
  //double offset  = 0.202837;    // GeV
  //double slope1  = 1.21606;  // unitless
  //double slope2  = 1.56778;  // unitless

  //New Tune Values
  //double stitch1 = 0.0145409;     // GeV
  //double offset  = 0.0607282;    // GeV
  //double slope1  = 0.642951;  // unitless
  //double slope2  = 2.07193;  // unitless

  //New Tune Values with Error Rounding
  double stitch1 = 0.015;     // GeV
  double offset  = 0.061;    // GeV
  double slope1  = 0.64;  // unitless
  double slope2  = 2.072;  // unitless

  TLine fitLine1 = TLine(0,offset,stitch1,slope1*stitch1+offset);
  fitLine1.SetLineColor(2);

  TLine fitLine2 = TLine(stitch1,slope2*stitch1+(slope1-slope2)*stitch1+offset,2.0,slope2*2.0+(slope1-slope2)*stitch1+offset);
  fitLine2.SetLineColor(2);


  TLine stitchLine1 = TLine(stitch1, 0, stitch1,5.0);
  stitchLine1.SetLineColor(2);
  stitchLine1.SetLineStyle(7);

  
  //Graph with fitting line
  TCanvas* c4 = new TCanvas("c4","c4");
  customProfile->Draw("AP"); 
  fitLine1.Draw();
  fitLine2.Draw();
  stitchLine1.Draw();
  gPad->Update();
  gPad->SetRightMargin(0.13);
  //TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  //palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c4->Print("fittingLineOnGraphhadcc.pdf");

  //2d plot with fitting line
  TCanvas* c5 = new TCanvas("c5","c5");
  hist->Draw("colz");
  fitLine1.Draw();
  fitLine2.Draw();
  stitchLine1.Draw();
  gPad->Update();
  gPad->SetRightMargin(0.13);
  //TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  //palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c5->Print("fittingLineOnMotherhadcc.pdf");

  //Plot of chi squared per ndf
  TCanvas* c6 = new TCanvas("c6","c6");
  chiValues->Draw(); 
  c6->Print("graphChiSquaredperNDFhadcc.pdf");

  //2d plot with fitting line on logz
  TCanvas* c8 = new TCanvas("c8","c8");
  c8->SetLogz();
  hist->Draw("colz");
  fitLine1.Draw();
  fitLine2.Draw();
  stitchLine1.Draw();
  gPad->Update();
  gPad->SetRightMargin(0.13);
  //TPaletteAxis *palette1 = (TPaletteAxis*)hist->GetListOfFunctions()->FindObject("palette");
  //palette1->SetBBoxCenterX(625);
  gPad->Modified();
  gPad->Update();
  c8->Print("fittingLineOnMotherLogzhadcc.pdf");

  TFile f("hadronccProfile.root","RECREATE");
  customProfile->Write("1");
  chiValues->Write();
  
  
 
}



