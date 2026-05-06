#include <math.h>
#include "../include/config.h"

void config_setup(Config *config) {
    config->KB = 1.380649e-23; // Boltzmann constant in J/K
    config->NA = 6.02214076e23; // Avogadro's number in 1/mol

    // size of domain (m)
    config->Lx = 1.0;
    config->Ly = 1.0;
    config->Lz = 1.0;
    config->DL = 0.1; // reservoir thickness for boundary conditions

    // gas properties
    config->molarMass = 0.040;

    // simulation loop parameters
    config->dt = 1.0e-5;
    config->nSteps = 2000;
    config->printPeriod = 200;
    config->particlesPerCellTarget = 200;
    config->firstSampleStep = 1000;
    config->samplingPeriod = 10;

    #ifdef WING_CASE
        // wing
        config->WingX = 0.3; // X coordinate of the wing leading edge (m)
        config->WingY = 0.5; // Y coordinate of the wing leading edge (m)
        config->WingLength = 0.2; // Length of the wing (m)
        config->Tw = 300.0; // Temperature of the wing surface (K)

        // stream
        config->MaFree = 4.0; // Free stream Mach number
        config->PFree = 0.1; // Free stream pressure (Pa)
        config->TFree = 200.0; // Free stream temperature (K)
        config->angleOfAttack = 30.0; // Angle of attack (degrees)
    #elif defined(BALL_CASE)
        // ball 
        config->ballCenterX = 0.6; // in m
        config->ballCenterY = 0.6; // in m
        config->ballCenterZ = 0.5; // in m
        config->ballRadius = 0.15; // in m
        config->Tb = 300.0; // Temperature of the ball surface (K)

        // stream
        config->MaFree = 4.0; // Free stream Mach number
        config->PFree = 0.1; // Free stream pressure (Pa)
        config->TFree = 200.0; // Free stream temperature (K)
    #endif

    // VHS model
    config->omega = 0.77;          // or ~0.74 to 0.77 for air-like species
    config->dRef  = 4.0e-10;       // reference diameter, meters

    // derived parameters
    config->moleculeMass = config->molarMass / config->NA;

    // derived, used in collider
    config->sigmaRef = M_PI * config->dRef * config->dRef;
    config->CrRef = sqrt(4.0 * config->KB * config->TFree / config->moleculeMass);
}



