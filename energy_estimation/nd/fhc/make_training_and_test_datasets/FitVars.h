#pragma once
#include "/nova/app/users/essmith/studies/bpfcvn_prong/nu18/macros/cafBPFEnergyEstimator/FitFuncs.h"
#include "CAFAna/Core/Var.h"
#include "CAFAna/Core/MultiVar.h"
#include "CAFAna/Vars/Vars.h"
#include "StandardRecord/Proxy/SRProxy.h"
#include "CAFAna/Vars/CVNProngVars.h"

#include "CAFAna/Vars/BPFEnergyConstants.h"
#include "CAFAna/Vars/BPFEnergyVars.h"

#include <cassert>

namespace ana
{

  bool SelectProngByTruth(const caf::SRProxy *sr, int idx, std::vector<int> pdg)
  {
    if(sr->vtx.nelastic == 0) return false;
    if(sr->vtx.elastic[0].fuzzyk.npng == 0) return false;

    for(unsigned int pdg_idx = 0; pdg_idx < pdg.size(); pdg_idx++){
      if(sr->vtx.elastic[0].fuzzyk.png[idx].truth.pdg == pdg[pdg_idx] && sr->vtx.elastic[0].fuzzyk.png[idx].truth.eff >= 0.7 && sr->vtx.elastic[0].fuzzyk.png[idx].truth.pur >= 0.7) return true;
    }
    return false;
  }

  std::vector<int> GetTrueProngIndices(const caf::SRProxy *sr, std::vector<int> pdg)
    {
      std::vector<int> idx;

      if(sr->vtx.nelastic == 0) return idx;
      if(sr->vtx.elastic[0].fuzzyk.npng == 0) return idx;

      for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png.size(); png_idx++) {
        auto &png = sr->vtx.elastic[0].fuzzyk.png[png_idx];
        if(!SelectProngByTruth(sr, png_idx, pdg)) continue;
        idx.push_back(png_idx);
      }
      return idx;
    }

  // gives prong indices for everything BUT the pdg you ask for 
  std::vector<int> GetGarbageBinIndices(const caf::SRProxy *sr, std::vector<int> pdg)
    { 
      std::vector<int> idx;

      if(sr->vtx.nelastic == 0) return idx;
      if(sr->vtx.elastic[0].fuzzyk.npng == 0) return idx;

      for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png.size(); png_idx++) {
        auto &png = sr->vtx.elastic[0].fuzzyk.png[png_idx];
        if(SelectProngByTruth(sr, png_idx, pdg)) continue;
        idx.push_back(png_idx);
      }
      return idx;
    }

  std::vector<double> GetProngTrueE(const caf::SRProxy *sr, std::vector<int> pdg)
  {
    std::vector<double> trueE;
    std::vector<int> indices = GetTrueProngIndices(sr, pdg);
    if(indices.size()==0) return trueE;

    for(unsigned int idx = 0; idx < indices.size(); idx++){
      auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
      if(png.truth.pdg==2212) trueE.push_back(png.truth.p.E-NameMass["Proton"]); // prKE
      else trueE.push_back(png.truth.p.E);
    }
    return trueE;
  }

  std::vector<double> GetProngCalE(const caf::SRProxy *sr, std::vector<int> pdg)
  {
    std::vector<double> calE;
    std::vector<int> indices = GetTrueProngIndices(sr, pdg);
    if(indices.size()==0) return calE;

    for(unsigned int idx = 0; idx < indices.size(); idx++){
      auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
      if(abs(png.truth.pdg) == 13) calE.push_back(png.calE);
      else calE.push_back(png.weightedCalE);
    }
    return calE;
  }

  std::vector<double> GetProngBPFE(const caf::SRProxy *sr, int pdg)
  {
    std::vector<double> BPFE;
    std::vector<int> indices = GetTrueProngIndices(sr, {pdg});
    if(indices.size()==0) return BPFE;

    double E = -5.0;

    for(unsigned int idx = 0; idx < indices.size(); idx++){
      auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
      for(unsigned int bpf_idx = 0; bpf_idx < png.bpf.size(); bpf_idx++) {
        if(png.bpf[bpf_idx].pdg != png.truth.pdg || png.bpf[bpf_idx].energy == -5.0) continue;
        if(png.truth.pdg==2212) E = png.bpf[bpf_idx].energy-NameMass["Proton"];
        else E = png.bpf[bpf_idx].energy;
      } 
      if(E==-5.0) continue;
      BPFE.push_back(E);
    }
    return BPFE;
  }
  
  std::vector<double> GetProngTrueEwBPFtrk(const caf::SRProxy *sr, int pdg)
  {
    std::vector<double> trueE;
    std::vector<int> indices = GetTrueProngIndices(sr, {pdg});
    if(indices.size()==0) return trueE;

    double E = -5.0;

    for(unsigned int idx = 0; idx < indices.size(); idx++){
      auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
      for(unsigned int bpf_idx = 0; bpf_idx < png.bpf.size(); bpf_idx++){
        if(png.bpf[bpf_idx].pdg != png.truth.pdg || png.bpf[bpf_idx].energy == -5.0) continue;
        if(pdg==2212) E = png.truth.p.E-NameMass["Proton"];
        else E = png.truth.p.E;
      }
      if(E==-5.0) continue;
      trueE.push_back(E);
    }
    return trueE;
  }

  std::vector<double> GetProngCalEwBPFtrk(const caf::SRProxy *sr, int pdg)
  {
    std::vector<double> calE;
    std::vector<int> indices = GetTrueProngIndices(sr, {pdg});
    if(indices.size()==0) return calE;

    double E = -5.0;

    for(unsigned int idx = 0; idx < indices.size(); idx++){
      auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
      for(unsigned int bpf_idx = 0; bpf_idx < png.bpf.size(); bpf_idx++){
        if(png.bpf[bpf_idx].pdg != png.truth.pdg || png.bpf[bpf_idx].energy == -5.0) continue;
        if(abs(png.truth.pdg) == 13) calE.push_back(png.calE - png.bpf[bpf_idx].overlapE);
        else E = png.weightedCalE;
      }
      if(E==-5.0) continue;
      calE.push_back(E);
    }
    return calE;
  }

// ----------------------------------------------------------
//               Truth Selected Prong Indices
// ----------------------------------------------------------

//  const MultiVar kTrueMuonIdx([](const caf::SRProxy *sr) {return GetTrueProngIndices(sr, {13});});
//  const MultiVar kTrueProtonIdx([](const caf::SRProxy *sr) {return GetTrueProngIndices(sr, {2212});});
//  const MultiVar kTruePionIdx([](const caf::SRProxy *sr) {return GetTrueProngIndices(sr, {211});});
//  const MultiVar kTrueEMIdx([](const caf::SRProxy *sr) {return GetTrueProngIndices(sr, {22, 11, 111});});
//  const MultiVar kTrueNeutronIdx([](const caf::SRProxy *sr) {return GetTrueProngIndices(sr, {2112});});

// ----------------------------------------------------------
//                  Truth Selected Prongs
// ----------------------------------------------------------

  // --- True Energy, all --- 

  const MultiVar kTrueMuonTrueE([](const caf::SRProxy *sr) {return GetProngTrueE(sr, {13});});
  const MultiVar kTrueProtonTrueKE([](const caf::SRProxy *sr){return GetProngTrueE(sr, {2212});});
  const MultiVar kTruePionTrueE([](const caf::SRProxy *sr){return GetProngTrueE(sr, {211});});

  // I can't decide if I like the idea of EM and hadE added together or separated by prong so I'll try it both ways
  const MultiVar kTrueEMsepTrueE([](const caf::SRProxy *sr){return GetProngTrueE(sr, {22, 11, 111});});
  const MultiVar kTrueHadsepTrueE([](const caf::SRProxy *sr){return GetProngTrueE(sr, {2212, 211});});

  const Var kTrueEMTrueE([](const caf::SRProxy *sr)
        {
          double trueE = 0;
          std::vector<double> trueEvec = GetProngTrueE(sr, {22, 11, 111});
          if(trueEvec.size()==0) return -5.0;
          for(unsigned int idx=0; idx < trueEvec.size(); idx++){
            trueE+=trueEvec[idx];
          }
          return trueE;
        });

  // not the garbage bin version, which is kTrueNonMuonNonEMTrueE
  const Var kTrueHadTrueE([](const caf::SRProxy *sr)
        {
          double trueE = 0;
          std::vector<double> trueEvec = GetProngTrueE(sr, {2212, 211});
          if(trueEvec.size()==0) return -5.0;
          for(unsigned int idx=0; idx<trueEvec.size(); idx++){
            trueE+=trueEvec[idx];
          }
          return trueE;
        });

  // --- True Energy, BPF ----
  
  const MultiVar kTrueMuonTrueEwBPFtrk([](const caf::SRProxy *sr){return GetProngTrueEwBPFtrk(sr, 13);});
  const MultiVar kTrueProtonTrueKEwBPFtrk([](const caf::SRProxy *sr){return GetProngTrueEwBPFtrk(sr, 2212);});
  const MultiVar kTruePionTrueEwBPFtrk([](const caf::SRProxy *sr){return GetProngTrueEwBPFtrk(sr, 211);});

  // --- Calorimetric Energy, all --- 

  const MultiVar kTrueMuonCalE([](const caf::SRProxy *sr){return GetProngCalE(sr, {13});});
  const MultiVar kTrueProtonCalE([](const caf::SRProxy *sr){return GetProngCalE(sr, {2212});});
  const MultiVar kTruePionCalE([](const caf::SRProxy *sr){return GetProngCalE(sr, {211});});
  const MultiVar kTrueEMsepCalE([](const caf::SRProxy *sr){return GetProngCalE(sr, {22, 11,111});});
  const MultiVar kTrueHadsepCalE([](const caf::SRProxy *sr){return GetProngCalE(sr, {2212, 211});});

  const Var kTrueEMCalE([](const caf::SRProxy *sr)
        {
          double calE = 0;
          std::vector<double> calEvec = GetProngCalE(sr, {22, 11, 111});
          if(calEvec.size()==0) return -5.0;
          for(unsigned int idx=0; idx < calEvec.size(); idx++){
            calE+=calEvec[idx];
          }
          return calE;
        });

  const Var kTrueHadCalE([](const caf::SRProxy *sr)
        {
          double calE = 0;
          std::vector<double> calEvec = GetProngCalE(sr, {2212, 211});
          if(calEvec.size()==0) return -5.0;
          for(unsigned int idx=0; idx < calEvec.size(); idx++){
            calE+=calEvec[idx];
          } 
          return calE;
        });

  // garbage bin -- everything but selected muon
  const Var kTrueNonMuonCalE([](const caf::SRProxy *sr)
        {
          double calE = 0;
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[png_idx];
            if(SelectProngByTruth(sr, png_idx, {13})) continue;
            calE+=png.weightedCalE;
          }

          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            calE+=png.weightedCalE;
          }

          calE+=sr->vtx.elastic[0].fuzzyk.orphCalE;
          return calE;
        });

  // garbage bin -- everything but selected muon and selected EM prongs
  const Var kTrueNonMuonNonEMCalE([](const caf::SRProxy *sr)
        {
          double calE = 0;
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[png_idx];
            if(SelectProngByTruth(sr, png_idx, {13, 22, 11, 111})) continue;
            calE+=png.weightedCalE;
          }

          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            calE+=png.weightedCalE;
          }

          calE+=sr->vtx.elastic[0].fuzzyk.orphCalE;
          return calE;
        });

  // garbage bin -- everything but selected muon, selected EM, selected had
  const Var kTrueOtherCalE([](const caf::SRProxy *sr)
        {
          double calE = 0;
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[png_idx];
            if(SelectProngByTruth(sr, png_idx, {13, 22, 11, 111, 2212, 211})) continue;
            calE += png.weightedCalE;
          }
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            calE += png.weightedCalE;
          }

          calE += sr->vtx.elastic[0].fuzzyk.orphCalE;
          return calE;
        });

  // --- Calorimetric Energy, BPF --- 
  
  const MultiVar kTrueMuonCalEwBPFtrk([](const caf::SRProxy *sr){return GetProngCalEwBPFtrk(sr, 13);});
  const MultiVar kTrueProtonCalEwBPFtrk([](const caf::SRProxy *sr){return GetProngCalEwBPFtrk(sr, 2212);});
  const MultiVar kTruePionCalEwBPFtrk([](const caf::SRProxy *sr){return GetProngCalEwBPFtrk(sr, 211);});

  // --- BPF Energy --- 
  
  const MultiVar kTrueMuonBPFE([](const caf::SRProxy *sr){return GetProngBPFE(sr, 13);});
  const MultiVar kTrueProtonBPFKE([](const caf::SRProxy *sr){return GetProngBPFE(sr, 2212);});
  const MultiVar kTruePionBPFE([](const caf::SRProxy *sr){return GetProngBPFE(sr, 211);});


// ----------------------------------------------------------
//                  Energy Estimator Vars
// ----------------------------------------------------------

//  const Var kTrueMuonEE([](const caf::SRProxy *sr)
//        {
//          double E = 0;
//          std::vector<int> idx = GetTrueProngIndices(sr, {13});
//          for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){
//            double bpf = 0.0;
//            double cal = 0.0;
//            auto &png = sr->vtx.elastic[0].fuzzyk.png[idx[png_idx]];
//
//            // check that BPF energy is valid
//            for(unsigned int bpf_idx=0; bpf_idx < png.bpf.size(); bpf_idx++){
//              if(png.bpf[bpf_idx].pdg != png.truth.pdg || png.bpf[bpf_idx].energy == -5.0) continue;
//              bpf = BPFp0["Muon"] + BPFp1["Muon"]*png.bpf[bpf_idx].energy;
//            }
//            if(bpf == 0.0) {
//              cal = CalEp0["Muon"] + CalEp1["Muon"]*png.weightedCalE;
//              E += cal;
//            }
//            else {
//              cal = CalEwBPFp0["Muon"] + CalEwBPFp1["Muon"]*png.weightedCalE;
//              E+= 0.65*bpf + 0.35*cal;
//            }
//          }
//          return E;
//        });

  const Var kTrueMuonEE([](const caf::SRProxy *sr)
            {    
              double muE = 0.0;

              // get best muon prong index
              std::vector<int> idx = GetTrueProngIndices(sr, {13});
              for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){

                // get the muon calE and BPFE
                // As usual, here is a hardcoded assumption about there being only one vertex...
                double muBPFE = -1.0;
                double muCalE = sr->vtx.elastic[0].fuzzyk.png[png_idx].calE; // need to subtract overlapE in the loop below

                // loop over tracks to make sure we only get the track fit under the muon assumption
                for(unsigned int t = 0; t < sr->vtx.elastic[0].fuzzyk.png[png_idx].bpf.size(); ++t) {
                  // skip this track if it is not the muon assumption
                  if(sr->vtx.elastic[0].fuzzyk.png[png_idx].bpf[t].pdg != 13) continue;
                  muBPFE = sr->vtx.elastic[0].fuzzyk.png[png_idx].bpf[t].energy;
                  muCalE = muCalE - sr->vtx.elastic[0].fuzzyk.png[png_idx].bpf[t].overlapE; // subtract overlapping E from the muon track (to be put in the garbage bin)
                }

                // adjust both E values with spline fits
                double newCalE = useSpline(muonFit_calE_FD_p3, muCalE);
                double newCalEwBPF = useSpline(muonFit_calEwBPF_FD_p3, muCalE);
                double newBPFE = useSpline(muonFit_BPFE_FD_p3, muBPFE);

                // Finally, combine newBPFE and newCalE.
                // If no BPF muon track existed, fall back on calE only.
                if(muBPFE < 0.0) muE += newCalE;
                else             muE += kMuonFitWeight * newBPFE + (1.0 - kMuonFitWeight) * newCalEwBPF;
              }

              return muE;

            });

  
//  const Var kTrueProtonEE([](const caf::SRProxy *sr)
//        {
//          double E = 0;
//          std::vector<int> idx = GetTrueProngIndices(sr, {2212});
//          for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){
//            double bpf = 0.0;
//            double cal = 0.0;
//            auto &png = sr->vtx.elastic[0].fuzzyk.png[idx[png_idx]];
//
//            // check that BPF energy is valid
//            for(unsigned int bpf_idx=0; bpf_idx < png.bpf.size(); bpf_idx++){
//              if(png.bpf[bpf_idx].pdg != png.truth.pdg || png.bpf[bpf_idx].energy == -5.0) continue;
//              bpf = BPFp0["Proton"] + BPFp1["Proton"]*png.bpf[bpf_idx].energy;
//            }
//            if(bpf == 0.0) {
//              cal = CalEp0["Proton"] + CalEp1["Proton"]*png.weightedCalE;
//              E += cal; 
//            }
//            else { 
//              cal = CalEwBPFp0["Proton"] + CalEwBPFp1["Proton"]*png.weightedCalE;
//              E += 0.7*bpf + 0.3*cal;
//            }
//          }
//          return E; 
//        });

  const Var kTrueProtonEE([](const caf::SRProxy *sr) 
        {
          double E = 0.0;
          std::vector<int> idx = GetTrueProngIndices(sr, {2212});
          for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){
            double bpf = 0.0;
            double cal = 0.0;
            auto &png = sr->vtx.elastic[0].fuzzyk.png[idx[png_idx]];
            // check that BPF energy is valid 
            for(unsigned int bpf_idx=0; bpf_idx < png.bpf.size(); bpf_idx++){
              if(png.bpf[bpf_idx].pdg != png.truth.pdg || png.bpf[bpf_idx].energy == -5.0) continue;
              bpf = useSpline(protonFit_BPFE_FD_p3, png.bpf[bpf_idx].energy);
            }
            if(bpf==0.0) E += useSpline(protonFit_calE_FD_p3, png.weightedCalE);
            else {
              cal = useSpline(protonFit_calEwBPF_FD_p3, png.weightedCalE);
              E += kProtonFitWeight*bpf + (1.0 - kProtonFitWeight)*cal;
            }
          }
        return E;
        });


//  const Var kTruePionEE([](const caf::SRProxy *sr)
//        {
//          double E = 0;
//          std::vector<int> idx = GetTrueProngIndices(sr, {211});
//          for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){
//            double bpf = 0.0;
//            double cal = 0.0;
//            auto &png = sr->vtx.elastic[0].fuzzyk.png[idx[png_idx]];
//
//            // check that BPF energy is valid
//            for(unsigned int bpf_idx=0; bpf_idx < png.bpf.size(); bpf_idx++){
//              if(png.bpf[bpf_idx].pdg != png.truth.pdg || png.bpf[bpf_idx].energy == -5.0) continue;
//              bpf = BPFp0["Pion"] + BPFp1["Pion"]*png.bpf[bpf_idx].energy;
//            }
//            if(bpf == 0.0) {
//              cal = CalEp0["Pion"] + CalEp1["Pion"]*png.weightedCalE;
//              E += cal;
//            }
//            else{
//              cal = CalEwBPFp0["Pion"] + CalEwBPFp1["Pion"]*png.weightedCalE;
//              E += 0.55*bpf + 0.45*cal;
//            }
//          }
//          return E; 
//        });

  const Var kTruePionEE([](const caf::SRProxy *sr)
        {
          double E = 0.0;
          std::vector<int> idx = GetTrueProngIndices(sr, {211});
          for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){
            double bpf = 0.0;
            double cal = 0.0;
            auto &png = sr->vtx.elastic[0].fuzzyk.png[idx[png_idx]];
            // check that BPF energy is valid 
            for(unsigned int bpf_idx=0; bpf_idx < png.bpf.size(); bpf_idx++){
              if(png.bpf[bpf_idx].pdg != png.truth.pdg || png.bpf[bpf_idx].energy == -5.0) continue;
              bpf = useSpline(pionFit_BPFE_FD_p3, png.bpf[bpf_idx].energy);
            }
            if(bpf==0.0) E += useSpline(pionFit_calE_FD_p3, png.weightedCalE);
            else {
              cal = useSpline(pionFit_calEwBPF_FD_p3, png.weightedCalE);
              E += kPionFitWeight*bpf + (1.0 - kPionFitWeight)*cal;
            }
          }
        return E;
        });


  const Var kTrueEMsepEE([](const caf::SRProxy *sr)
        {
          double CalE = 0;
          std::vector<int> idx = GetTrueProngIndices(sr, {22, 11, 111});
          for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[idx[png_idx]];
//            CalE+= CalEp0["EMsep"] + CalEp1["EMsep"]*png.weightedCalE;
            CalE += useSpline(EMsepFit_FD_p3, png.weightedCalE);
          }
          return CalE;
        });

  const Var kTrueEMsumEE([](const caf::SRProxy *sr)
        {
          double CalE = 0;
          std::vector<int> idx = GetTrueProngIndices(sr, {22, 11, 111});
          for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[idx[png_idx]];
            CalE+= png.weightedCalE;
          }
//          return (CalEp0["EMsum"] + CalEp1["EMsum"]*CalE);
          return useSpline(EMsumFit_FD_p3, CalE);
        });

  const Var kTrueHadsepEE([](const caf::SRProxy *sr)
        {
          double CalE = 0;
          std::vector<int> idx = GetTrueProngIndices(sr, {2212, 211});
          for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[idx[png_idx]];
//            CalE+= CalEp0["Hadsep"] + CalEp1["Hadsep"]*png.weightedCalE;
            CalE += useSpline(hadsepFit_FD_p3, png.weightedCalE);
          }
          return CalE;
        });

  const Var kTrueHadsumEE([](const caf::SRProxy *sr)
        {
          double CalE = 0;
          std::vector<int> idx = GetTrueProngIndices(sr, {2212, 211});
          for(unsigned int png_idx = 0; png_idx < idx.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[idx[png_idx]];
            CalE+=png.weightedCalE;
          }
//          return (CalEp0["Hadsum"] + CalEp1["Hadsum"]*CalE);
          return useSpline(hadsumFit_FD_p3, CalE);
        });

  // need to add fitted garbage bins

  const Var kNonMuonEE([](const caf::SRProxy *sr)
        {
          double E = 0;
          std::vector<int> indices = GetGarbageBinIndices(sr, {13});

          // get garbage bin calE
          for(unsigned int idx = 0; idx < indices.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            E+=png.weightedCalE;
          }

          std::vector<int> muIdx = GetTrueProngIndices(sr, {13});
          // check if muon has BPF track, if yes, take track overlapE
          for(unsigned int idx = 0; idx< muIdx.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            for(unsigned int t = 0; t < png.bpf.size(); ++t) {
              // skip this track if it is not the muon assumption
              if(png.bpf[t].pdg != 13) continue;
              // if the muon track exists add the trackoverlapE to the garbage bin
              E+=png.bpf[t].overlapE;
            }
          }
          
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            E+=png.weightedCalE;
          }
          E+=sr->vtx.elastic[0].fuzzyk.orphCalE;

//          return (CalEp0["-mu"] + CalEp1["-mu"]*E);
          return useSpline(garbageFit_FD_v1_p3, E);
        });

  const Var kNonMuonEMsepEE([](const caf::SRProxy *sr)
        {
          double E = 0;
          std::vector<int> indices = GetGarbageBinIndices(sr, {13, 22, 11, 111});
          for(unsigned int idx = 0; idx < indices.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            E+=png.weightedCalE;
          }
          
          std::vector<int> muIdx = GetTrueProngIndices(sr, {13});
          // check if muon has BPF track, if yes, take track overlapE
          for(unsigned int idx = 0; idx< muIdx.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            for(unsigned int t = 0; t < png.bpf.size(); ++t) {
              // skip this track if it is not the muon assumption
              if(png.bpf[t].pdg != 13) continue;
              // if the muon track exists add the trackoverlapE to the garbage bin
              E+=png.bpf[t].overlapE;
            }
          }

          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            E+=png.weightedCalE;
          }
          E+=sr->vtx.elastic[0].fuzzyk.orphCalE;

//          return (CalEp0["-mu-EMsep"] + CalEp1["-mu-EMsep"]*E);
          return useSpline(garbageFit_FD_v2_p3, E);
        });

  const Var kNonMuonEMsumEE([](const caf::SRProxy *sr)
        {
          double E = 0;
          std::vector<int> indices = GetGarbageBinIndices(sr, {13, 22, 11, 111});
          for(unsigned int idx = 0; idx < indices.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            E+=png.weightedCalE;
          }
          std::vector<int> muIdx = GetTrueProngIndices(sr, {13});
          // check if muon has BPF track, if yes, take track overlapE
          for(unsigned int idx = 0; idx< muIdx.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            for(unsigned int t = 0; t < png.bpf.size(); ++t) {
              // skip this track if it is not the muon assumption
              if(png.bpf[t].pdg != 13) continue;
              // if the muon track exists add the trackoverlapE to the garbage bin
              E+=png.bpf[t].overlapE;
            }
          }
 
          
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            E+=png.weightedCalE;
          }
          E+=sr->vtx.elastic[0].fuzzyk.orphCalE;

//          return (CalEp0["-mu-EMsum"] + CalEp1["-mu-EMsum"]*E);
          return useSpline(garbageFit_FD_v2_p3, E);
        });

  const Var kNonMuonEMsephadsepEE([](const caf::SRProxy *sr)
        {
          double E = 0;
          std::vector<int> indices = GetGarbageBinIndices(sr, {13, 22, 11, 111, 2212, 211});
          for(unsigned int idx = 0; idx < indices.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            E+=png.weightedCalE;
          }

          std::vector<int> muIdx = GetTrueProngIndices(sr, {13});
          // check if muon has BPF track, if yes, take track overlapE
          for(unsigned int idx = 0; idx< muIdx.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            for(unsigned int t = 0; t < png.bpf.size(); ++t) {
              // skip this track if it is not the muon assumption
              if(png.bpf[t].pdg != 13) continue;
              // if the muon track exists add the trackoverlapE to the garbage bin
              E+=png.bpf[t].overlapE;
            }
          }
 
          
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            E+=png.weightedCalE;
          }
          E+=sr->vtx.elastic[0].fuzzyk.orphCalE;

//          return (CalEp0["-mu-EMsep-hadsep"] + CalEp1["-mu-EMsep-hadsep"]*E);
          return useSpline(garbageFit_FD_v3_p3, E);
        });

  const Var kNonMuonEMsumhadsumEE([](const caf::SRProxy *sr)
        {
          double E = 0;
          std::vector<int> indices = GetGarbageBinIndices(sr, {13, 22, 11, 111, 2212, 211});
          for(unsigned int idx = 0; idx < indices.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            E+=png.weightedCalE;
          }

          std::vector<int> muIdx = GetTrueProngIndices(sr, {13});
          // check if muon has BPF track, if yes, take track overlapE
          for(unsigned int idx = 0; idx< muIdx.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            for(unsigned int t = 0; t < png.bpf.size(); ++t) {
              // skip this track if it is not the muon assumption
              if(png.bpf[t].pdg != 13) continue;
              // if the muon track exists add the trackoverlapE to the garbage bin
              E+=png.bpf[t].overlapE;
            }
          }
 
                    
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            E+=png.weightedCalE;
          }
          E+=sr->vtx.elastic[0].fuzzyk.orphCalE;

//          return (CalEp0["-mu-EMsum-hadsum"] + CalEp1["-mu-EMsum-hadsum"]*E);
          return useSpline(garbageFit_FD_v3_p3, E);
        });

  const Var kNonMuonEMsepprpiEE([](const caf::SRProxy *sr)
        {
          double E = 0;
          std::vector<int> indices = GetGarbageBinIndices(sr, {13, 22, 11, 111, 2212, 211});
          for(unsigned int idx = 0; idx < indices.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            E+=png.weightedCalE;
          }
          std::vector<int> muIdx = GetTrueProngIndices(sr, {13});
          // check if muon has BPF track, if yes, take track overlapE
          for(unsigned int idx = 0; idx< muIdx.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            for(unsigned int t = 0; t < png.bpf.size(); ++t) {
              // skip this track if it is not the muon assumption
              if(png.bpf[t].pdg != 13) continue;
              // if the muon track exists add the trackoverlapE to the garbage bin
              E+=png.bpf[t].overlapE;
            }
          }
 
 
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            E+=png.weightedCalE;
          }
          E+=sr->vtx.elastic[0].fuzzyk.orphCalE;

//          return (CalEp0["-mu-EMsep-pr-pi"] + CalEp1["-mu-EMsep-pr-pi"]*E);
          return useSpline(garbageFit_FD_v3_p3, E);
        });

  const Var kNonMuonEMsumprpiEE([](const caf::SRProxy *sr)
        {
          double E = 0;
          std::vector<int> indices = GetGarbageBinIndices(sr, {13, 22, 11, 111, 2212, 211});
          for(unsigned int idx = 0; idx < indices.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            E+=png.weightedCalE;
          }

          std::vector<int> muIdx = GetTrueProngIndices(sr, {13});
          // check if muon has BPF track, if yes, take track overlapE
          for(unsigned int idx = 0; idx< muIdx.size(); idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png[indices[idx]];
            for(unsigned int t = 0; t < png.bpf.size(); ++t) {
              // skip this track if it is not the muon assumption
              if(png.bpf[t].pdg != 13) continue;
              // if the muon track exists add the trackoverlapE to the garbage bin
              E+=png.bpf[t].overlapE;
            }
          }
 
         
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            E+=png.weightedCalE;
          }
            E+=sr->vtx.elastic[0].fuzzyk.orphCalE;

//        return (CalEp0["-mu-EMsum-pr-pi"] + CalEp1["-mu-EMsum-pr-pi"]*E);
          return useSpline(garbageFit_FD_v3_p3, E);
        });


  // these should be true neutrino energy minus the reco energy of everything else
  const Var kTrueNonMuonTrueE([](const caf::SRProxy *sr)
        {
          return (sr->mc.nu[0].E - kTrueMuonEE(sr));
        });

  const Var kTrueNonMuonNonEMsepTrueE([](const caf::SRProxy *sr)
        {
          return (sr->mc.nu[0].E - kTrueMuonEE(sr) - kTrueEMsepEE(sr));
        });

  const Var kTrueNonMuonNonEMsumTrueE([](const caf::SRProxy *sr)
        {
          return (sr->mc.nu[0].E - kTrueMuonEE(sr) - kTrueEMsumEE(sr));
        });

  const Var kTrueOthersepTrueE([](const caf::SRProxy *sr)
        {
          return (sr->mc.nu[0].E - kTrueMuonEE(sr) - kTrueHadsepEE(sr) - kTrueEMsepEE(sr));
        });

  const Var kTrueOthersumTrueE([](const caf::SRProxy *sr)
        {
          return (sr->mc.nu[0].E - kTrueMuonEE(sr) - kTrueHadsumEE(sr) - kTrueEMsumEE(sr));
        });

  const Var kTrueOthersepwPrPiTrueE([](const caf::SRProxy *sr)
        {
          return (sr->mc.nu[0].E - kTrueMuonEE(sr) - kTrueProtonEE(sr) - kTruePionEE(sr) - kTrueEMsepEE(sr)); 
        });

  const Var kTrueOthersumwPrPiTrueE([](const caf::SRProxy *sr)
        {
          return (sr->mc.nu[0].E - kTrueMuonEE(sr) - kTrueProtonEE(sr) - kTruePionEE(sr) - kTrueEMsumEE(sr));
        });

  // need to add TrueNumuEE for each type of garbage bin... 

  const Var kCVNMuonEE([](const caf::SRProxy *sr)
        {
          double E = 0.0;
          double bpf = 0.0;
          double cal = 0.0;
          unsigned int muIdx = (unsigned int)kCVNMuonIdx(sr);
          for(unsigned int bpf_idx=0; bpf_idx < sr->vtx.elastic[0].fuzzyk.png[muIdx].bpf.size(); bpf_idx++){
            if(sr->vtx.elastic[0].fuzzyk.png[muIdx].bpf[bpf_idx].pdg != 13 || sr->vtx.elastic[0].fuzzyk.png[muIdx].bpf[bpf_idx].energy == -5.0) continue;
            bpf = BPFp0["Muon"] + BPFp1["Muon"]*sr->vtx.elastic[0].fuzzyk.png[muIdx].bpf[bpf_idx].energy;
          }
          if (bpf == 0.0){
            cal = CalEp0["Muon"] + CalEp1["Muon"]*sr->vtx.elastic[0].fuzzyk.png[muIdx].calE;
            E+= cal;
          }
          else { 
            cal = CalEwBPFp0["Muon"] + CalEwBPFp1["Muon"]*sr->vtx.elastic[0].fuzzyk.png[muIdx].calE;
            E+= 0.65*bpf + 0.35*cal;
          } 
          return E;
        });

  const Var kCVNNonMuonEE([](const caf::SRProxy *sr)
        {
          double E = 0;  
          unsigned int muIdx = (unsigned int)kCVNMuonIdx(sr);
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png.size(); png_idx++) {
            if(png_idx == muIdx) continue;
            auto &png = sr->vtx.elastic[0].fuzzyk.png[png_idx];
            E+=png.weightedCalE;
          }
          for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[0].fuzzyk.png2d.size(); png_idx++){
            auto &png = sr->vtx.elastic[0].fuzzyk.png2d[png_idx];
            E+=png.weightedCalE;
          }
          E+=sr->vtx.elastic[0].fuzzyk.orphCalE;

          return (CalEp0["-mu"] + CalEp1["-mu"]*E);
            
        });

  const Var kCVNNonMuonNumuEE([](const caf::SRProxy *sr)
        {
          return kCVNMuonEE(sr) + kCVNNonMuonEE(sr);
        });

  const Var kCVNNonMuRmToT([](const caf::SRProxy *sr)
        {
          return (kCVNNonMuonNumuEE(sr)-kTrueE(sr))/kTrueE(sr); //sr->mc.nu[0].E)/sr->mc.nu[0].E;

        });



  const Var kNonMuNumuEE([](const caf::SRProxy *sr)
        {
          return kTrueMuonEE(sr) + kNonMuonEE(sr);
        });

  const Var kNonMuEMsepNumuEE([](const caf::SRProxy *sr)
        {
          return kTrueMuonEE(sr) + kTrueEMsepEE(sr) + kNonMuonEMsepEE(sr);
        });

  const Var kNonMuEMsumNumuEE([](const caf::SRProxy *sr)
        {
          return kTrueMuonEE(sr) + kTrueEMsumEE(sr) + kNonMuonEMsumEE(sr);
        });

  const Var kNonMuEMsephadsepNumuEE([](const caf::SRProxy *sr)
        {
          return kTrueMuonEE(sr) + kTrueEMsepEE(sr) + kTrueHadsepEE(sr) + kNonMuonEMsephadsepEE(sr);
        });

  const Var kNonMuEMsumhadsumNumuEE([](const caf::SRProxy *sr)
        {
          return kTrueMuonEE(sr) + kTrueEMsumEE(sr) + kTrueHadsumEE(sr) + kNonMuonEMsumhadsumEE(sr);
        });

  const Var kNonMuEMsepprpiNumuEE([](const caf::SRProxy *sr)
        {
          return kTrueMuonEE(sr) + kTrueEMsepEE(sr) + kTrueProtonEE(sr) + kTruePionEE(sr) + kNonMuonEMsepprpiEE(sr);
        });

  const Var kNonMuEMsumprpiNumuEE([](const caf::SRProxy *sr)
        {
          return kTrueMuonEE(sr) + kTrueEMsumEE(sr) + kTrueProtonEE(sr) + kTruePionEE(sr) + kNonMuonEMsumprpiEE(sr);
        });


// ------------------ old and in desperate need of deleting ----------------
  const Var kBPFTruthNumuE([](const caf::SRProxy *sr)
  {
    DeclarePars();
    double TRUEprongRecoE, TRUEprongRecoP;
    double muonCalE, protonCalE, pionCalE;
    double muonBPFP, protonBPFP, pionBPFP;

    double EME3D = 0;
    double Other3D = 0;
    double Other2D = 0;
    double P_EME3D = 0;

    double muonP = 0;
    double protonP = 0;
    double pionP = 0;
    
    double BPFTruthNumuE = -5;
    double muonWt = 0;   // 0.65 if BPF track exists
    double pionWt = 0;   // 0.5 if BPF track exists
    double protonWt = 0; // 0.5 if BPF track exists

    for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
      for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
        auto &png = sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx];
//        if(png.truth.pur < 0.75) return BPFTruthNumuE;
        double prng_calE = png.calE;
        double prng_totalGeV = prng_calE/1.78;
        double TRUEpartmass = NameMass[PDGname[png.truth.pdg]];
       
        double PBPFmuoncorr = 0;
        double PBPFpioncorr = 0;
        double PBPFprotoncorr = 0;
        double PCalEmuoncorr = 0;
        double PCalEpioncorr = 0;
        double PCalEprotoncorr = 0;
 
        // Muon
        if(png.truth.pdg==13){ 
          muonCalE = prng_totalGeV;
          PCalEmuoncorr = P_CalE(muonCalE, PDGname[png.truth.pdg]);
          for(unsigned int bpf_idx = 0; bpf_idx < png.bpf.size(); bpf_idx++) {
            if(png.bpf[bpf_idx].pdg != png.truth.pdg) continue;
            muonWt = 0.65;
            TRUEprongRecoE = png.bpf[bpf_idx].energy;
            muonBPFP = EtoP(TRUEprongRecoE, TRUEpartmass);
            PBPFmuoncorr = P_BPF(muonBPFP, PDGname[png.truth.pdg]);
          }//end BPF tracks
          muonP += muonWt*PBPFmuoncorr + (1-muonWt)*PCalEmuoncorr;
        }//end muon

        // Pion
        else if(png.truth.pdg==211){ 
          pionCalE = prng_totalGeV;
          PCalEpioncorr = P_CalE(pionCalE, PDGname[png.truth.pdg]);
          for(unsigned int bpf_idx = 0; bpf_idx < png.bpf.size(); bpf_idx++) {
            if(png.bpf[bpf_idx].pdg != png.truth.pdg) continue;
            pionWt = 0.5;
            TRUEprongRecoE = png.bpf[bpf_idx].energy;
            pionBPFP = EtoP(TRUEprongRecoE, TRUEpartmass);
            PBPFpioncorr = P_BPF(pionBPFP, PDGname[png.truth.pdg]);
          }//end BPF tracks
          pionP += pionWt*PBPFpioncorr + (1-pionWt)*PCalEpioncorr;
        }//end pion
       

        // Proton
        else if(png.truth.pdg==2212){ 
          protonCalE = prng_totalGeV;
          PCalEprotoncorr = P_CalE(protonCalE, PDGname[png.truth.pdg]);
          for(unsigned int bpf_idx = 0; bpf_idx < png.bpf.size(); bpf_idx++) {
            if(png.bpf[bpf_idx].pdg != png.truth.pdg) continue;
            protonWt = 0.5;   
            TRUEprongRecoE = png.bpf[bpf_idx].energy;
            protonBPFP = EtoP(TRUEprongRecoE, TRUEpartmass);
            PBPFprotoncorr = P_BPF(protonBPFP, PDGname[png.truth.pdg]);
          }//end BPF tracks
          protonP += protonWt*PBPFprotoncorr + (1-protonWt)*PCalEprotoncorr;
        }//end proton

        // EM
        else if (png.truth.pdg==22 || png.truth.pdg==11) EME3D += prng_totalGeV;

        // Other
        else Other3D += prng_totalGeV;
      }//end 3D prong

      P_EME3D = P_CalE(EME3D, "EM3D");

      for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png2d.size(); png_idx++) { 
        auto &png = sr->vtx.elastic[vtx_idx].fuzzyk.png2d[png_idx];
        Other2D += png.calE/1.78;
      }// end 2D prong
    }// end vertex

    double TRUEnuE = sr->mc.nu[0].E;
    if(TRUEnuE == -5 || muonP == -5 || protonP == -5 || pionP == -5) return BPFTruthNumuE;
    if(TRUEnuE < 0 || muonP < 0 || protonP < 0 || pionP < 0 || Other2D < 0 || EME3D < 0 ||  Other3D < 0) return BPFTruthNumuE;

    double Emu = PtoE(muonP, NameMass["Muon"]);
    double Ehad3D = PtoE(pionP, NameMass["Pion"]) + EtoKE(PtoE(protonP, NameMass["Proton"]), NameMass["Proton"]) + P_EME3D;
    double E_unc = Other2D + Other3D;

    BPFTruthNumuE = Emu + Ehad3D + P_CalE(E_unc,"unc");

    return BPFTruthNumuE;
  });

  const Var kBPFCVNNumuE([](const caf::SRProxy *sr)
  {
    DeclarePars();
    double RECOprongRecoE, RECOprongRecoP;
    double muonCalE, protonCalE, pionCalE;
    double muonBPFP, protonBPFP, pionBPFP;

    double EME3D = 0;
    double Other3D = 0;
    double Other2D = 0;
    double P_EME3D = 0;

    double muonP = 0;
    double protonP = 0;
    double pionP = 0;
    
    double BPFCVNNumuE = -5;
    double muonWt = 0;   // 0.65 if BPF track exists
    double pionWt = 0;   // 0.5 if BPF track exists
    double protonWt = 0; // 0.5 if BPF track exists

    int bestMu_prong_Idx_cvn, bestMu_PID_cvn;
    double longestLen = 0.0;
    int longest_prong_Idx; 

    for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
      bestMu_prong_Idx_cvn = -1;
      longest_prong_Idx = -1;
      bestMu_PID_cvn = -5.0;

      for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
        auto &png = sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx];
        if(png.cvnpart.muonid>bestMu_PID_cvn){
          bestMu_prong_Idx_cvn = png_idx;
          bestMu_PID_cvn = png.cvnpart.muonid;
        }
        if(png.len > longestLen){
          longestLen = png.len;
          longest_prong_Idx = png_idx;
        }
      }

      if(longestLen >= 500.0) bestMu_prong_Idx_cvn = longest_prong_Idx;

      for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
        auto &png = sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx];
        double prng_calE = png.calE;
        double prng_totalGeV = prng_calE/1.78;
       
        double PBPFmuoncorr = 0;
        double PBPFpioncorr = 0;
        double PBPFprotoncorr = 0;
        double PCalEmuoncorr = 0;
        double PCalEpioncorr = 0;
        double PCalEprotoncorr = 0;
 
        double muonid     = png.cvnpart.muonid;
        double protonid   = png.cvnpart.protonid;
        double pionid     = png.cvnpart.pionid;
        double electronid = png.cvnpart.electronid;
        double photonid    = png.cvnpart.photonid;

        // Muon
        if(png_idx == bestMu_prong_Idx_cvn){
          muonCalE = prng_totalGeV;
          PCalEmuoncorr = P_CalE(muonCalE, "Muon");
          for(unsigned int bpf_idx = 0; bpf_idx < png.bpf.size(); bpf_idx++) {
            if(png.bpf[bpf_idx].pdg != 13) continue;
            muonWt = 0.65;
            RECOprongRecoE = png.bpf[bpf_idx].energy;
            muonBPFP = EtoP(RECOprongRecoE, NameMass["Muon"]);
            PBPFmuoncorr = P_BPF(muonBPFP, "Muon");
          }//end BPF tracks
          muonP += muonWt*PBPFmuoncorr + (1-muonWt)*PCalEmuoncorr;
          continue;
        }//end muon

        // Proton
        else if(protonid>muonid && protonid>pionid && protonid>electronid && protonid>photonid && protonid>=0.8){
          protonCalE = prng_totalGeV;
          PCalEprotoncorr = P_CalE(protonCalE, "Proton");
          for(unsigned int bpf_idx = 0; bpf_idx < png.bpf.size(); bpf_idx++) {
            if(png.bpf[bpf_idx].pdg != 2212) continue;
            protonWt = 0.5;   
            RECOprongRecoE = png.bpf[bpf_idx].energy;
            protonBPFP = EtoP(RECOprongRecoE, NameMass["Proton"]);
            PBPFprotoncorr = P_BPF(protonBPFP, "Proton");
          }//end BPF tracks
          protonP += protonWt*PBPFprotoncorr + (1-protonWt)*PCalEprotoncorr;
          continue;
        }//end proton

        // Pion - allow to have a higher muon score or pion score than any of these - we've already picked the muon and we know we won't have two
        else if (pionid>protonid && pionid>electronid && pionid>photonid &&
                 muonid>protonid && muonid>electronid && muonid>photonid){
          pionCalE = prng_totalGeV;
          PCalEpioncorr = P_CalE(pionCalE, "Pion");
          for(unsigned int bpf_idx = 0; bpf_idx < png.bpf.size(); bpf_idx++) {
            if(png.bpf[bpf_idx].pdg != 211) continue;
            pionWt = 0.5;
            RECOprongRecoE = png.bpf[bpf_idx].energy;
            pionBPFP = EtoP(RECOprongRecoE, NameMass["Pion"]);
            PBPFpioncorr = P_BPF(pionBPFP, "Pion");
          }//end BPF tracks
          pionP += pionWt*PBPFpioncorr + (1-pionWt)*PCalEpioncorr;
          continue;
        }//end pion

        // EM
        else if ((electronid>muonid && electronid>protonid && electronid>pionid && electronid>=0.8) 
                 || (photonid>muonid && photonid>protonid && photonid>pionid && photonid>=0.8)) EME3D += prng_totalGeV;

        // Other
        else Other3D += prng_totalGeV;
      }//end 3D prong

      P_EME3D = P_CalE(EME3D, "EM3D");

      for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png2d.size(); png_idx++) { 
        auto &png = sr->vtx.elastic[vtx_idx].fuzzyk.png2d[png_idx];
        Other2D += png.calE/1.78;
      }// end 2D prong
    }// end vertex

    double TRUEnuE = sr->mc.nu[0].E;
    if(TRUEnuE == -5 || muonP == -5 || protonP == -5 || pionP == -5) return BPFCVNNumuE;
    if(TRUEnuE < 0 || muonP < 0 || protonP < 0 || pionP < 0 || Other2D < 0 || EME3D < 0 ||  Other3D < 0) return BPFCVNNumuE;

    double Emu = PtoE(muonP, NameMass["Muon"]);
    double Ehad3D = PtoE(pionP, NameMass["Pion"]) + EtoKE(PtoE(protonP, NameMass["Proton"]), NameMass["Proton"]) + P_EME3D;
    double E_unc = Other2D + Other3D;

    BPFCVNNumuE = Emu + Ehad3D + P_CalE(E_unc,"unc");

    return BPFCVNNumuE;

  });


}
