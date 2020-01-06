// Fills spectra for systematic error band
#ifdef __CINT__
void GetHistsND(int stride = 100000, int offset = 0)
{
  std::cout << "Sorry, you must run in compiled mode" << std::endl;
}
#else

#include "CAFAna/Core/Binning.h"
#include "CAFAna/Core/Loaders.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"
#include "CAFAna/Core/Var.h"
#include "CAFAna/Cuts/Cuts.h"
#include "CAFAna/Cuts/NumuCuts.h"
#include "CAFAna/Cuts/NumuCuts2018.h"
#include "CAFAna/Cuts/SpillCuts.h"
#include "CAFAna/Vars/GenieWeights.h"
#include "CAFAna/Vars/NumuVars.h"
#include "CAFAna/Vars/PPFXWeights.h"
#include "NDAna/numucc_inc/NumuCCIncCuts.h"
#include "NDAna/numucc_inc/NumuCCIncVars.h"

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"

using namespace ana;

void GetHistsND(int stride = 100000, int offset = 0)
{

  std::string nd_nonswap = "defname: prod_caf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_period3_v1 with stride "+to_string(stride)+" offset "+to_string(offset);
  SpectrumLoader loader(nd_nonswap);
  loader.SetSpillCut(kStandardSpillCuts);

  struct Plot2D
  {
    std::string name;
    std::string labelx;
    Binning binsx;
    Var varx;
    std::string labely;
    Binning binsy;
    Var vary;
    Cut cut;
  };

  //~ const Cut kTrueEbelow5({"mc.nnu","mc.nu.E"}, 
                  //~ [](const caf::StandardRecord* sr)
                  //~ {
                    //~ if (sr->mc.nnu < 1) return false;
                    //~ else return (sr->mc.nu[0].E <= 5.0);
                  //~ });

  //~ const Cut kHasTrueMuon({"energy.numusimp.mc.truegoodmuon","energy.numusimp.mc.truemuonE"}, 
                  //~ [](const caf::StandardRecord* sr)
                  //~ {
                    //~ if (sr->energy.numusimp.mc.truegoodmuon < 1) return false;
                    //~ else return (sr->energy.numusimp.mc.truemuonE > 0);
                  //~ });

  //~ const Cut kCut = kTrueEbelow5 && kIsNumuCC && kHasTrueMuon && kNumuND;
  const Cut kCut = kNumuCutND2018 && kIsNumuCC;

  //// ------ Now the binnings and variables ------ /////

  const Binning kMuonEnergyBinning = Binning::Simple(150,0,5);
  const Binning kHadEnergyBinning = Binning::Simple(150,0,5);
  const Binning kTrackLengthBinning = Binning::Simple(150, 0, 15);

  std::vector<double> hadronBins;
  double hadronAxis = 0.0;
  for(int i = 0; i < 116; ++i){
    hadronBins.push_back(hadronAxis);
    if (hadronAxis < 1.0){hadronAxis = hadronAxis + 0.01;}
    else if (hadronAxis < 1.5){hadronAxis = hadronAxis + 0.05;}
    else {hadronAxis = hadronAxis + 0.1;}
  }

  const Binning kHadEBinning = Binning::Custom(hadronBins);

  const Var kTrueHadE = kTrueE - kTrueMuonE;

  //~ const Var kTrueCatcherE = SIMPLEVAR(energy.numusimp.mc.truemuoncatcherE);

  //~ const Var kActive1      = SIMPLEVAR(energy.numusimp.ndhadcalactE);
  //~ const Var kActive2      = SIMPLEVAR(energy.numusimp.ndhadtrkactE);

  //~ const Var kTran1        = SIMPLEVAR(energy.numusimp.ndhadcaltranE);
  //~ const Var kTran2        = SIMPLEVAR(energy.numusimp.ndhadtrktranE);

  //~ const Var kCatcher1     = SIMPLEVAR(energy.numusimp.ndhadcalcatE);
  //~ const Var kCatcher2     = SIMPLEVAR(energy.numusimp.ndhadtrkcatE);

  //~ const Var kHadActive    = kActive1 + kActive2;
  //~ const Var kHadTran      = kTran1 + kTran2;
  //~ const Var kHadCatcher   = kCatcher1 + kCatcher2;

  //~ const Var kHadAll = kHadActive + kHadTran + kHadCatcher;

  //~ const Var kTrkCalAct    = SIMPLEVAR(energy.numusimp.ndtrkcalactE);
  //~ const Var kTrkCalTran   = SIMPLEVAR(energy.numusimp.ndtrkcaltranE); 
  //~ const Var kTrkCalCat    = SIMPLEVAR(energy.numusimp.ndtrkcalcatE);

  //~ const Var kTrkLenAct ({"energy.numusimp.ndtrklenact"}, 
                  //~ [](const caf::StandardRecord* sr)
                  //~ {
                    //~ return (sr->energy.numusimp.ndtrklenact / 100); // in m
                  //~ });

  //~ const Var kTrkLenCat ({"energy.numusimp.ndtrklencat"}, 
                  //~ [](const caf::StandardRecord* sr)
                  //~ {
                    //~ return (sr->energy.numusimp.ndtrklencat / 100); // in m
                  //~ });

  //~ const Cut kAllActive = (kTrkCalAct + kTrkCalTran > 0) && (kTrkCalCat == 0);
  //~ const Cut kAllCatcher = (kTrkCalAct + kTrkCalTran == 0) && (kTrkCalCat > 0);
  //~ const Cut kActiveAndCatcher = (kTrkCalAct + kTrkCalTran > 0) && (kTrkCalCat > 0);

  const int kNumPlots2D = 4;

  Plot2D plots2D[kNumPlots2D] = {

    {"MuonE_hist_active","Reco muon track length (m)", kTrackLengthBinning, kTrkLenAct, "True muon energy (GeV)" , kMuonEnergyBinning, kTrueMuonE, kCut && kAllActive},
    {"MuonE_hist_catcher","Reco muon track length (m)", kTrackLengthBinning, kTrkLenCat, "True muon energy (GeV)" , kMuonEnergyBinning, kTrueMuonE, kCut && kAllCatcher},
    {"MuonE_hist_activeAndCatcher","Reco muon track length (m)", kTrackLengthBinning, kTrkLenCat, "True muon energy in catcher (GeV)" , kMuonEnergyBinning, kTrueCatcherE, kCut && kActiveAndCatcher},
    {"HadE_hist_DIS","Visible hadronic energy (GeV)", kHadEBinning, kHadAll, "True hadronic energy (GeV)" , kHadEnergyBinning, kTrueHadE, kCut}
  };

  Spectrum* sSpect2D[kNumPlots2D];

  for(int i = 0; i < kNumPlots2D; i++)
  {
    Plot2D p = plots2D[i];
    // sSpect2D[i] = new Spectrum(p.labelx, p.labely, loader, p.binsx, p.varx, p.binsy, p.vary, p.cut, kNoShift, kTuftsWeightCC * kPPFXFluxCVWgt ); // for miniproduction
    sSpect2D[i] = new Spectrum(p.labelx, p.labely, loader, p.binsx, p.varx, p.binsy, p.vary, p.cut, kNoShift, kXSecCVWgt2018 );
  } // loop over 1D plots

  loader.Go();

  TFile f("2DPlotsForFittingND.root","RECREATE");

  for(int i = 0; i < kNumPlots2D; i++)
  {
    Plot2D p = plots2D[i];
    TH2* h = sSpect2D[i]->ToTH2(sSpect2D[i]->POT());
    h->SetName(p.name.c_str());
    h->Write();
  } // loop over 1D plots

}

#endif
