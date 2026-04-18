#ifndef DSMC_CONFIG_H
#define DSMC_CONFIG_H

#define MAX_PARTICLES 200000
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

// boundary conditions
extern double Tleft;
extern double Tright;

// wing
extern double WingX;
extern double WingY;
extern double WingLength;
extern double Tw;

// stream
extern double MaFree;
extern double PFree;
extern double TFree;
extern double angleOfAttack;

#endif //DSMC_CONFIG_H
