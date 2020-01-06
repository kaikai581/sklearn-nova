/*
 * This script makes training and test samples for muon and hadron energy
 * estimation.
 */

#ifdef __CINT__
void make_training_and_test_datasets_official_cuts(int stride = 100000, int offset = 0)
{
  std::cout << "Sorry, you must run in compiled mode" << std::endl;
}
#else

// standard library includes
#include <fstream>
#include <iostream>
#include <iterator>

// CAFAna includes
#include "CAFAna/Cuts/Cuts.h"
#include "CAFAna/Cuts/SpillCuts.h"
#include "CAFAna/Cuts/TimingCuts.h"
#include "CAFAna/Cuts/NumuCuts.h"
#include "CAFAna/Cuts/NumuCuts2018.h"
#include "CAFAna/Vars/CVNFinalStates.h"
#include "CAFAna/Vars/NumuVars.h"
#include "CAFAna/Vars/GenieWeights.h"
#include "CAFAna/Vars/PPFXWeights.h"
#include "CAFAna/Core/EventList.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"
#include "CAFAna/Core/GraphDrawer.h"
#include "CAFAna/Vars/NumuVars.h"
#include "CAFAna/Vars/Vars.h"
#include "NDAna/numucc_inc/NumuCCIncCuts.h"
#include "NDAna/numucc_inc/NumuCCIncVars.h"

// Erica's variables
//#include "FitVars.h"

using namespace ana;

// Define my own Vars and Cuts
const Var kRun    = SIMPLEVAR(hdr.run);
const Var kSubRun = SIMPLEVAR(hdr.subrun);
const Var kCycle  = SIMPLEVAR(hdr.cycle);
const Var kEvt    = SIMPLEVAR(hdr.evt);
const Var kSlice  = SIMPLEVAR(hdr.subevt);

const Var kTrueHadE = kTrueE - kTrueMuonE;

const Var kCCFlag([](const caf::SRProxy* sr)
                  {
                    if(sr->mc.nnu == 0) return -1;
                    return int(sr->mc.nu[0].iscc);
                  });

const Var kMuStopZ([](const caf::SRProxy* sr)
                   {
                     if(sr->trk.kalman.ntracks < 1) return -5.f;
                     return sr->trk.kalman.tracks[0].stop.Z();
                   });

const Var kCVNFSElectronScore([](const caf::SRProxy* sr){return CVNFinalStateScore2018(sr, 11);});
const Var kCVNFSMuonScore([](const caf::SRProxy* sr){return CVNFinalStateScore2018(sr, 13);});
const Var kCVNFSPi0Score([](const caf::SRProxy* sr){return CVNFinalStateScore2018(sr, 111);});
const Var kCVNFSChargedPionScore([](const caf::SRProxy* sr){return CVNFinalStateScore2018(sr, 211);});
const Var kCVNFSNeutronScore([](const caf::SRProxy* sr){return CVNFinalStateScore2018(sr, 2112);});

const Var kNPng([](const caf::SRProxy* sr)
                {
                  if(sr->vtx.nelastic == 0) return -1;

                  return (int)sr->vtx.elastic[0].fuzzyk.npng;
                });

const Var kReMIdTrkIsMuon([](const caf::SRProxy* sr)
                          {
                            if(sr->trk.kalman.ntracks < 1) return 0;
                            return (abs(sr->trk.kalman.tracks[0].truth.pdg) == 13? 1:0);
                          });

const Var kIsNumuCCVar([](const caf::SRProxy* sr)
                       {
                         if(sr->mc.nnu == 0) return 0;
                         return (sr->mc.nu[0].iscc && abs(sr->mc.nu[0].pdg) == 14)? 1:0;
                       });

const Cut kRemContND([](const caf::SRProxy* sr)
                     {
                       /// prong based containment condition
                       if( sr->vtx.nelastic < 1 ) return false;
                       /// reconstructed showers all contained
                       for( unsigned int i = 0; i < sr->vtx.elastic[0].fuzzyk.nshwlid; ++i ) {
                         TVector3 start = sr->vtx.elastic[0].fuzzyk.png[i].shwlid.start;
                         TVector3 stop  = sr->vtx.elastic[0].fuzzyk.png[i].shwlid.stop;
                         if( std::min( start.X(), stop.X() ) < -180.0 ) return false;
                         if( std::max( start.X(), stop.X() ) >  180.0 ) return false;
                         if( std::min( start.Y(), stop.Y() ) < -180.0 ) return false;
                         if( std::max( start.Y(), stop.Y() ) >  180.0 ) return false;
                         if( std::min( start.Z(), stop.Z() ) <   20.0 ) return false;
                         if( std::max( start.Z(), stop.Z() ) > 1525.0 ) return false;
                       }
                       
                       /// only primary muon track present in muon catcher
                       if( sr->trk.kalman.ntracks < 1 ) return false;
                       for( unsigned int i = 0; i < sr->trk.kalman.ntracks; ++i ) {
                         if( i == sr->trk.kalman.idxremid ) continue;
                         else if( sr->trk.kalman.tracks[i].start.Z() > 1275 ||
                            sr->trk.kalman.tracks[i].stop.Z()  > 1275 )
                           return false;
                       }
                      
                       return ( sr->trk.kalman.ntracks > sr->trk.kalman.idxremid &&
                                sr->trk.kalman.tracks[0].remcont );
                     });

const Cut kSel = kNumuQuality && kNumuContainND2017 && kIsNumuCC;

const Cut kMuonIdentified([](const caf::SRProxy* sr)
                          {
                            return abs(sr->trk.kalman.tracks[0].truth.pdg) == 13;
                          });

void make_training_and_test_datasets_official_cuts(int stride = 100000, int offset = 0)
{
  string dataset = "defname: prod_caf_R17-11-14-prod4reco.d_nd_genie_nonswap_fhc_nova_v08_full_v1 with stride "+to_string(stride)+" offset "+to_string(offset);
  /// print with number of tracks info
  MakeTextListFile(dataset, {kNumuND}, {"nd_standard_numucc_selection_stride"+to_string(stride)+"_offset"+to_string(offset)+".txt"},
                   {&kRun, &kSubRun, &kCycle, &kEvt, &kSlice, &kIsNumuCCVar, &kTrueE, &kTrueMuonE, &kMuE, &kTrkLenAct, &kTrkLenCat, &kMuStopZ, &kTrueHadE, &kHadE, &kHadAll, &kRecoQ2, &kNPng, &kReMIdTrkIsMuon, &kCVNFSElectronScore, &kCVNFSMuonScore, &kCVNFSPi0Score, &kCVNFSChargedPionScore, &kCVNFSNeutronScore, &kCVNFSProtonScore2018},
                   &kStandardSpillCuts);
}
#endif
