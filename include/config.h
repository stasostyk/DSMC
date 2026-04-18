#ifndef DSMC_CONFIG_H
#define DSMC_CONFIG_H

#define MAX_PARTICLES 200000
#define MAX_PARTICLES_PER_CELL 1000
#define NX 20
#define NY 20

extern const double KB;
extern const double NA;

// size of domain (m)
extern const double Lx;
extern const double Ly;
extern const double Lz;
extern const double DL;

// gas properties
extern const double molarMass;
// hard sphere assumption
extern const double diameter;

// initial velocity (m/s)
extern const double ux0;
extern const double uy0;
extern const double uz0;

// simulation loop parameters
extern const double dt;
extern const int nSteps;
extern const int printPeriod;
extern const int particlesPerCellTarget;
extern const int firstSampleStep;
extern const int samplingPeriod;

// wing
extern const double WingX;
extern const double WingY;
extern const double WingLength;
extern const double Tw;

// stream
extern const double MaFree;
extern const double PFree;
extern const double TFree;
extern const double angleOfAttack;

// VHS model
extern const double omega;
extern const double dRef;

// derived parameters
extern const double moleculeMass;

#endif //DSMC_CONFIG_H
