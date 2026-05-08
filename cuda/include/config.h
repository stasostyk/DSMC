#ifndef DSMC_CONFIG_H
#define DSMC_CONFIG_H

#include <cuda_runtime.h>

#define MAX_PARTICLES 500000000
#define MAX_PARTICLES_PER_CELL 2000
#define NX 100
#define NY 100
#define NZ 100

// TODO calculate once these constants
#define PARTICLES_SZ (MAX_PARTICLES * sizeof(Particle))
#define SAMPLES_SZ (NX * NY * NZ * sizeof(Cell))
#define CELL_COUNT_SZ (NX * NY * NZ * sizeof(int))
#define CELL_LIST_SZ (NX * NY * NZ * MAX_PARTICLES_PER_CELL * sizeof(int))

typedef struct {
    double KB;
    double NA;

    // size of domain (m)
    double Lx;
    double Ly;
    double Lz;
    double DL;

    // gas properties
    double molarMass;

    // simulation loop parameters
    double dt;
    int nSteps;
    int printPeriod;
    int particlesPerCellTarget;
    int firstSampleStep;
    int samplingPeriod;

    #ifdef WING_CASE
        // wing
        double WingX;
        double WingY;
        double WingLength;
        double Tw;

        // stream
        double MaFree;
        double PFree;
        double TFree;
        double angleOfAttack;
    #elif defined(BALL_CASE)
        // ball 
        double ballCenterX;
        double ballCenterY;
        double ballCenterZ;
        double ballRadius;
        double Tb;

        // stream
        double MaFree;
        double PFree;
        double TFree;
    #endif

    // VHS model
    double omega;
    double dRef;

    // derived parameters
    double moleculeMass;
    double dx, dy, dz;
    double cellVolume;
    double weight;
    double NFree;
    double UFree;
    double UxFree, UyFree, UzFree;

    // derived, used in collider
    double sigmaRef;
    double CrRef;
    double ntcs_collisionProbMultiplier; // used in no time collision scheme
    double ntcs_collidingPairsMultiplier; // used in no time collision scheme
    double ntcs_collisionProbExponent;    // used in no time collision scheme

    // derived, used in simulation
    double generation_derivatedMultiplier;
} Config;

extern __constant__ Config d_conf;

void config_setup(Config *config);

#endif //DSMC_CONFIG_H
