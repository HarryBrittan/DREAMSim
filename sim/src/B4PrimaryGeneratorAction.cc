//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
/// \file B4PrimaryGeneratorAction.cc
/// \brief Implementation of the B4PrimaryGeneratorAction class

#include "B4PrimaryGeneratorAction.hh"

#include "G4RunManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "Py8Jet.h"

#include "B4DetectorConstruction.hh"

// for root tree
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TTree.h"
// #include "TROOT.h"

#include "Py8Jet.h"

#include "CaloTree.h"

using namespace std;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4PrimaryGeneratorAction::B4PrimaryGeneratorAction(B4DetectorConstruction *det, CaloTree *histo)
    : G4VUserPrimaryGeneratorAction(),
      fParticleGun(nullptr), fDetector(det), hh(histo)
{
  cout << "B4PrimaryGeneratorAction constructer is called..." << endl;
  // Create the table containing all particle names
  particleTable = G4ParticleTable::GetParticleTable();

  getParamFromEnvVars(); // get paramters from theenvvariables.

  if (CaloXPythiaON == 1)
  {
    py8eventCounter = 0;
    py8eventNumber = CaloXPythiaSkip;
    string inFileName = CaloXPythiaFile;
    std::cout << "B4PrimaryGeneratorAction:  Using Pythia Event file: " << inFileName << std::endl;
    finPy8 = new TFile(inFileName.c_str());
    TTree *py8tree;
    finPy8->GetObject("py8tree", py8tree);
    py8evt = new Py8Jet(py8tree);
    py8evt->Init(py8tree);
    std::cout << "B4PrimaryGeneratorAction: nentries=" << py8tree->GetEntriesFast() << std::endl;
  }
  else
  {
    G4int nofParticles = 1;
    fParticleGun = new G4ParticleGun(nofParticles);

    // default particle kinematic
    //
    auto particleDefinition = G4ParticleTable::GetParticleTable()->FindParticle("e-");
    fParticleGun->SetParticleDefinition(particleDefinition);
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
    fParticleGun->SetParticleEnergy(50. * MeV);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4PrimaryGeneratorAction::~B4PrimaryGeneratorAction()
{
  delete fParticleGun;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4PrimaryGeneratorAction::GeneratePrimaries(G4Event *anEvent)
{
  // cout<<"B4PrimaryGeneratorAction::GeneratePrimaries is called..."<<endl;
  // This function is called at the begining of event

  // In order to avoid dependence of PrimaryGeneratorAction
  // on DetectorConstruction class we get world volume
  // from G4LogicalVolumeStore
  //
  worldZHalfLength = 0.;
  auto worldLV = G4LogicalVolumeStore::GetInstance()->GetVolume("World");

  // Check that the world volume has box shape
  G4Box *worldBox = nullptr;
  if (worldLV)
  {
    worldBox = dynamic_cast<G4Box *>(worldLV->GetSolid());
  }

  if (worldBox)
  {
    worldZHalfLength = worldBox->GetZHalfLength();
  }
  else
  {
    G4ExceptionDescription msg;
    msg << "World volume of box shape not found." << G4endl;
    msg << "Perhaps you have changed geometry." << G4endl;
    msg << "The gun will be place in the center.";
    G4Exception("B4PrimaryGeneratorAction::GeneratePrimaries()",
                "MyCode0002", JustWarning, msg);
  }

  // The size of Calorimeter volume

  double calorimeterZHalfLength = 0.;
  auto calorimeterLV = G4LogicalVolumeStore::GetInstance()->GetVolume("Calorimeter");

  // Check that the world volume has box shape
  G4Box *calorimeterBox = nullptr;
  if (calorimeterLV)
  {
    calorimeterBox = dynamic_cast<G4Box *>(calorimeterLV->GetSolid());
  }

  if (calorimeterBox)
  {
    calorimeterZHalfLength = calorimeterBox->GetZHalfLength();
  }
  else
  {
    G4ExceptionDescription msg;
    msg << "Calorimeter volume of box shape not found." << G4endl;
    msg << "Perhaps you have changed geometry." << G4endl;
    msg << "The gun will be place in the center.";
    G4Exception("B4PrimaryGeneratorAction::GeneratePrimaries()",
                "MyCode0002", JustWarning, msg);
  }

  if (CaloXPythiaON == 1)
  {
    getPy8Event(anEvent);
  }
  else
  {
    double r1 = G4UniformRand();
    double r2 = G4UniformRand();
    double r3 = G4UniformRand();
    float x = ((hh->getParamF("gun_x_max") - hh->getParamF("gun_x_min")) * G4UniformRand() + hh->getParamF("gun_x_min")) * cm;
    float y = ((hh->getParamF("gun_y_max") - hh->getParamF("gun_y_min")) * G4UniformRand() + hh->getParamF("gun_y_min")) * cm;
    float z = ((hh->getParamF("gun_z_max") - hh->getParamF("gun_z_min")) * G4UniformRand() + hh->getParamF("gun_z_min")) * cm;
    // float z = -calorimeterZHalfLength - 50.0;

    float en = ((hh->getParamF("gun_energy_max") - hh->getParamF("gun_energy_min")) * G4UniformRand() + hh->getParamF("gun_energy_min")) * GeV;
    string ptype = hh->getParamS("gun_particle");

    G4int nofParticles = 1;
    fParticleGun = new G4ParticleGun(nofParticles);

    auto particleDefinition = G4ParticleTable::GetParticleTable()->FindParticle(ptype);
    fParticleGun->SetParticleDefinition(particleDefinition);
    fParticleGun->SetParticlePosition(G4ThreeVector(x, y, z));
    fParticleGun->SetParticleEnergy(en);
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.0, 0.0, 1.0));
    fParticleGun->GeneratePrimaryVertex(anEvent);
    // cout<<"B4PrimaryGeneratorAction::GeneratePrimaries set a particle..."<<endl;
    // cout<<"   (x,y,z,en)="<<x<<",  "<<y<<",  "<<z<<",  "<<en<<",  "<<ptype<<endl;
    int pdgid = particleDefinition->GetPDGEncoding();
    hh->saveBeamXYZE(ptype, pdgid, x, y, z, en);
  }
}

void B4PrimaryGeneratorAction::getParamFromEnvVars()
{
  CaloXPythiaON = 0;
  char *param1;
  param1 = getenv("CaloXPythiaON");
  if (param1 != NULL)
  {
    CaloXPythiaON = atoi(param1);
  }

  CaloXPythiaXmin = 0.0;
  CaloXPythiaXmax = 0.0;
  CaloXPythiaYmin = 0.0;
  CaloXPythiaYmax = 0.0;
  CaloXPythiaZmin = -99999.0;
  CaloXPythiaZmax = CaloXPythiaZmin; // use the World boundary.

  char *param2;
  param2 = getenv("CaloXPythiaXmin");
  if (param2 != NULL)
  {
    CaloXPythiaXmin = atof(param2);
  }

  char *param3;
  param3 = getenv("CaloXPythiaXmax");
  if (param3 != NULL)
  {
    CaloXPythiaXmax = atof(param3);
  }

  char *param4;
  param4 = getenv("CaloXPythiaYmin");
  if (param4 != NULL)
  {
    CaloXPythiaYmin = atof(param4);
  }

  char *param5;
  param5 = getenv("CaloXPythiaYmax");
  if (param5 != NULL)
  {
    CaloXPythiaYmax = atof(param5);
  }

  char *param6;
  param6 = getenv("CaloXPythiaZmin");
  if (param6 != NULL)
  {
    CaloXPythiaZmin = atof(param6);
  }

  char *param7;
  param7 = getenv("CaloXPythiaZmax");
  if (param7 != NULL)
  {
    CaloXPythiaZmax = atof(param7);
  }

  CaloXPythiaSkip = 0;
  CaloXPythiaPrint = 5; // number of pythia events to print.

  char *param8;
  param8 = getenv("CaloXPythiaSkip");
  if (param8 != NULL)
  {
    CaloXPythiaSkip = atof(param8);
  }

  char *param9;
  param9 = getenv("CaloXPythiaPrint");
  if (param9 != NULL)
  {
    CaloXPythiaPrint = atof(param9);
  }

  char *param10;
  param10 = getenv("CaloXPythiaFile");
  if (param10 != NULL)
  {
    CaloXPythiaFile = std::string(param10);
  }

  return;
}

// -----------------------------------------------------------------------------
void B4PrimaryGeneratorAction::getPy8Event(G4Event *anEvent)
{

  py8evt->GetEntry(py8eventNumber);
  if (py8eventCounter < CaloXPythiaPrint)
    printPy8Event();

  py8eventCounter++;
  py8eventNumber++;
  // std::cout<<"B4PrimaryGeneratorAction::getPy8Event  after py8evt->GetEntry="<<std::endl;
  // std::cout<<"py8evt->pid->size()  "<<py8evt->pid->size()<<std::endl;
  // std::cout<<"    pid=py8evt->pid->at(i) ="<<py8evt->pid->at(0)<<std::endl;

  G4ParticleGun mygun;
  double r1 = CLHEP::RandFlat::shoot();
  double r2 = CLHEP::RandFlat::shoot();
  double r3 = CLHEP::RandFlat::shoot();
  double x = CaloXPythiaXmin + (CaloXPythiaXmax - CaloXPythiaXmin) * r1;
  double y = CaloXPythiaYmin + (CaloXPythiaYmax - CaloXPythiaYmin) * r2;
  double z = CaloXPythiaZmin + (CaloXPythiaZmax - CaloXPythiaZmin) * r3;
  if (z < -worldZHalfLength)
    z = -worldZHalfLength + 0.0001; // limit to the World volume.
  double t = 0.0;

  // std::cout<<"B4PrimaryGeneratorAction::getPy8Event  x="<<x
  //<<"   CaloXPythiaXmin "<<CaloXPythiaXmin<<"  max "<<CaloXPythiaXmax<<std::endl;

  G4PrimaryVertex *vertex = new G4PrimaryVertex(G4ThreeVector(x, y, z), t);

  for (int i = 0; i < py8evt->pid->size(); i++)
  {
    if (py8evt->daughter1->at(i) == 0)
    {
      int pid = py8evt->pid->at(i);
      double px = py8evt->px->at(i) * GeV;
      double py = py8evt->py->at(i) * GeV;
      double pz = py8evt->pz->at(i) * GeV;
      G4ParticleDefinition *particle_definition = particleTable->FindParticle(pid);
      G4PrimaryParticle *particle = new G4PrimaryParticle(particle_definition, px, py, pz);
      vertex->SetPrimary(particle);
    }
  } // end of loop over particles...

  anEvent->AddPrimaryVertex(vertex);

  // std::cout<<"anEvent->GetPrimaryVertex(0)->GetNumberOfParticle(): "<<anEvent->GetPrimaryVertex(0)->GetNumberOfParticle()<<std::endl;
  // std::cout<<"anEvent->GetPrimaryVertex(0)->GetZ0(): "<<anEvent->GetPrimaryVertex(0)->GetZ0()<<std::endl;
  // G4PrimaryParticle* primary = anEvent->GetPrimaryVertex(0)->GetPrimary(0);
  // std::cout<<"B4PrimaryGeneratorAction::getPy8Event:"<<primary->GetMomentum()<<std::endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4PrimaryGeneratorAction::printPy8Event()
{
  // Header.
  cout << "\n --------  PY8CaloX Event Listing  " << py8evt->event << "----------"
       << "-------------------------------------------------\n \n    no    "
       << "    id    status     mothers   daughters     colou"
       << "rs      p_x        p_y        p_z         e          m \n";
  cout << endl;

  cout << "Pythia Event " << py8evt->event << endl;
  // Precision. At high energy switch to scientific format for momenta.
  int precision = 3; // use 3 for now (sk)
  int prec = max(3, precision);
  bool useFixed = (py8evt->e->at(0) < 1e5);

  for (int i = 0; i < py8evt->nparticles; i++)
  {
    cout << setw(6) << std::right << i
         << setw(10) << std::right << py8evt->pid->at(i)
         << setw(10) << py8evt->status->at(i)
         << setw(6) << py8evt->mother1->at(i)
         << setw(6) << py8evt->mother2->at(i)
         << setw(6) << py8evt->daughter1->at(i)
         << setw(6) << py8evt->daughter2->at(i)
         << setw(6) << py8evt->color1->at(i)
         << setw(6) << py8evt->color2->at(i)
         << ((useFixed) ? fixed : scientific) << setprecision(prec)
         << setw(8 + prec) << py8evt->px->at(i)
         << setw(8 + prec) << py8evt->py->at(i)
         << setw(8 + prec) << py8evt->pz->at(i)
         << setw(8 + prec) << py8evt->e->at(i)
         << setw(8 + prec) << py8evt->m->at(i)
         << "\n";
  } // end of loop over particles...
}

// -----------------------------------------------------------------------------
void B4PrimaryGeneratorAction::FillHEPparticles(
    std::vector<int> *mHepPID, std::vector<int> *mHepStatus,
    std::vector<int> *mHepMother1, std::vector<int> *mHepMother2,
    std::vector<int> *mHepDaughter1, std::vector<int> *mHepDaughter2,
    std::vector<float> *mHepPx, std::vector<float> *mHepPy,
    std::vector<float> *mHepPz, std::vector<float> *mHepE,
    std::vector<float> *mHepMass)
{
  if (py8evt == NULL)
    return;

  int n = py8evt->pid->size();
  for (int i = 0; i < n; i++)
  {
    mHepPID->push_back(py8evt->pid->at(i));
    mHepStatus->push_back(py8evt->status->at(i));
    mHepMother1->push_back(py8evt->mother1->at(i));
    mHepMother2->push_back(py8evt->mother2->at(i));
    mHepDaughter1->push_back(py8evt->daughter1->at(i));
    mHepDaughter2->push_back(py8evt->daughter2->at(i));
    mHepPx->push_back(py8evt->px->at(i));
    mHepPy->push_back(py8evt->py->at(i));
    mHepPz->push_back(py8evt->pz->at(i));
    mHepE->push_back(py8evt->e->at(i));
    mHepMass->push_back(py8evt->m->at(i));
  } // end of loop over particles...
}
