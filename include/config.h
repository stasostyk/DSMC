#ifndef DSMC_CONFIG_H
#define DSMC_CONFIG_H

#define MAX_PARTICLES 200000
#define NX 25
#define NY 25

const double KB = 1.380649e-23; // Boltzmann constant in J/K
const double NA = 6.02214076e23; // Avogadro's number in 1/mol

// size of domain (m)
const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;

// gas properties
const double T0 = 300.0;
const double n0 = 1.0e20;
const double molarMass = 0.028;
// hard sphere assumption
const double diameter = 4.0e-10;

// initial velocity (m/s)
const double ux0 = 0.0;
const double uy0 = 0.0;
const double uz0 = 0.0;

// simulation loop parameters
const double dt = 1.0e-5;
const int nSteps = 200;
const int printPeriod = 20;
const int particlesPerCellTarget = 200;
const int firstSampleStep = 0;
const int samplingPeriod = 1;

// boundary conditions
double Tleft = 400.0;
double Tright = 200.0;

#endif //DSMC_CONFIG_H
