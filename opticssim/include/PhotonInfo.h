#include "globals.hh"
#include "G4ThreeVector.hh"

struct PhotonInfo
{
    G4int trackID = -1;
    G4ThreeVector productionPosition = G4ThreeVector(0, 0, 0);
    G4ThreeVector exitPosition = G4ThreeVector(0, 0, 0);
    G4ThreeVector productionMomentum = G4ThreeVector(0, 0, 0);
    G4ThreeVector exitMomentum = G4ThreeVector(0, 0, 0);
    G4ThreeVector polarization = G4ThreeVector(0, 0, 0);
    G4double productionTime = 0;
    G4double exitTime = 0;
    G4int productionFiber = -99;
    G4int exitFiber = -99;
    G4bool isCerenkov = false;
    G4bool isScintillation = false;
    G4bool isCerenkovFiber = false;
    G4bool isScintillationFiber = false;
};
