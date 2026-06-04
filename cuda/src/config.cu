#include <math.h>
#include <cuda_runtime.h>
#include "../include/config.h"

__constant__ Config d_conf;

void setup_wing(Wing *wing) {
    wing->WingX = 0.3; // X coordinate of the wing leading edge (m)
    wing->WingY = 0.5; // Y coordinate of the wing leading edge (m)
    wing->WingLength = 0.2; // Length of the wing (m)
    wing->Tw = 300.0; // Temperature of the wing surface (K)
}

void setup_ball(Ball *ball) {
    ball->ballCenterX = 0.6; // in m
    ball->ballCenterY = 0.6; // in m
    ball->ballCenterZ = 0.5; // in m
    ball->ballRadius = 0.15; // in m
    ball->Tb = 300.0; // Temperature of the ball surface (K)

    // derived
    ball->ballRadiusSquared = ball->ballRadius * ball->ballRadius;
}

void setup_ball2(Ball *ball) {
    ball->ballCenterX = 0.2; // in m
    ball->ballCenterY = 0.2; // in m
    ball->ballCenterZ = 0.5; // in m
    ball->ballRadius = 0.1; // in m
    ball->Tb = 300.0; // Temperature of the ball surface (K)

    // derived
    ball->ballRadiusSquared = ball->ballRadius * ball->ballRadius;
}

void setup_ball3(Ball *ball) {
    ball->ballCenterX = 0.8; // in m
    ball->ballCenterY = 0.25; // in m
    ball->ballCenterZ = 0.5; // in m
    ball->ballRadius = 0.2; // in m
    ball->Tb = 300.0; // Temperature of the ball surface (K)

    // derived
    ball->ballRadiusSquared = ball->ballRadius * ball->ballRadius;
}

void config_setup(Config *config, int object_case) {
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
    config->printPeriod = 0;
    config->particlesPerCellTarget = 100;
    config->firstSampleStep = 1000;
    config->samplingPeriod = 10;

    // stream
    config->MaFree = 4.0; // Free stream Mach number
    config->PFree = 0.1; // Free stream pressure (Pa)
    config->TFree = 200.0; // Free stream temperature (K)
    config->angleOfAttack = 0.0; // Angle of attack (degrees)

    if (object_case == 0) {
        // Ball case
        config->wingCnt = 0;
        config->ballCnt = 1;

        setup_ball(&(config->balls[0]));
    } else if (object_case == 1) {
        // Wing case
        config->wingCnt = 1;
        config->ballCnt = 0;

        setup_wing(&(config->wings[0]));
        
        config->angleOfAttack = 30.0; // Angle of attack (degrees)
    } else if (object_case == 2) {
        // COMBO
        config->wingCnt = 1;
        config->ballCnt = 2;

        setup_wing(&(config->wings[0]));
        setup_ball2(&(config->balls[0]));
        setup_ball3(&(config->balls[1]));
        
        config->angleOfAttack = 30.0; // Angle of attack (degrees)
    }

    // VHS model
    // NOTE: collider calculations assume omega=0.75, so be careful when changing
    config->omega = 0.75;          // or ~0.74 to 0.77 for air-like species
    config->dRef  = 4.0e-10;       // reference diameter, meters

    // derived parameters
    config->moleculeMass = config->molarMass / config->NA;

    config->dx = config->Lx / NX;
    config->dy = config->Ly / NY;
    config->dz = config->Lz / NZ;
    config->cellVolume = config->dx * config->dy * config->dz;

    config->inv_dx = NX / config->Lx;
    config->inv_dy = NY / config->Ly;
    config->inv_dz = NZ / config->Lz;

    config->totalCells = NX * NY * NZ;

    config->NFree = config->PFree / ( config->KB * config->TFree );
    config->UFree = config->MaFree * sqrt ( ( 5.0 / 3.0 ) * config->KB * config->TFree / config->moleculeMass );
    config->UxFree = config->UFree * cos ( M_PI * config->angleOfAttack / 180.0 );
    config->UyFree = - config->UFree * sin ( M_PI * config->angleOfAttack / 180.0 );
    config->UzFree = 0.0;

    config->weight = config->NFree * config->cellVolume / config->particlesPerCellTarget;

    // derived, used in collider
    config->sigmaRef = M_PI * config->dRef * config->dRef;
    config->CrRef = sqrt(4.0 * config->KB * config->TFree / config->moleculeMass);

    // precompute multiplier used in no time collision scheme
    double majorant = 9.0 * config->sigmaRef * sqrt(config->KB * config->TFree / config->moleculeMass);
    config->ntcs_collidingPairsMultiplier = 0.5 * config->weight * majorant * config->dt / config->cellVolume;
    config->ntcs_collisionProbExponent = 2.0*config->omega - 1.0;
    config->ntcs_collisionProbMultiplier = pow(config->CrRef, config->ntcs_collisionProbExponent) * config->sigmaRef / majorant;
    config->ntcs_collisionProbMultiplierSquared = config->ntcs_collisionProbMultiplier * config->ntcs_collisionProbMultiplier;

    config->hss_nbatch = 3;
    config->hss_threshold = 400.0; // threshold for switching to HSS, in number of particles per cell
    config->hss_collisionProbMultiplier = (1.0/config->hss_nbatch) * config->weight * config->sigmaRef * config->dt / config->cellVolume;

    config->generation_derivatedMultiplier = sqrt(config->KB * config->TFree / config->moleculeMass);
}



