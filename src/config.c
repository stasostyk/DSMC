#include "../include/config.h"

const double KB = 1.380649e-23; // Boltzmann constant in J/K
const double NA = 6.02214076e23; // Avogadro's number in 1/mol

// size of domain (m)
const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;
const double DL = 0.1; // reservoir thickness for boundary conditions

// gas properties
const double molarMass = 0.040;

// simulation loop parameters
const double dt = 1.0e-5;
const int nSteps = 4000;
const int printPeriod = 200;
const int particlesPerCellTarget = 200;
const int firstSampleStep = 1000;
const int samplingPeriod = 10;

// wing
const double WingX = 0.3; // X coordinate of the wing leading edge (m)
const double WingY = 0.5; // Y coordinate of the wing leading edge (m)
const double WingLength = 0.2; // Length of the wing (m)
const double Tw = 300.0; // Temperature of the wing surface (K

// stream
const double MaFree = 4.0; // Free stream Mach number
const double PFree = 0.1; // Free stream pressure (Pa)
const double TFree = 200.0; // Free stream temperature (K)
const double angleOfAttack = 30.0; // Angle of attack (degrees)

// VHS model
const double omega = 0.77;          // or ~0.74 to 0.77 for air-like species
const double dRef  = 4.0e-10;       // reference diameter, meters

// derived parameters
const double moleculeMass = molarMass / NA;