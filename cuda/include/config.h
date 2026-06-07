#ifndef DSMC_CONFIG_H
#define DSMC_CONFIG_H

#include "object.h"
#include <cuda_runtime.h>

#define MAX_PARTICLES 30000000
#define MAX_PARTICLES_PER_CELL 750
#define NX 50
#define NY 50
#define NZ 50

#define PARTICLES_FIELD_SZ (MAX_PARTICLES * 3 * sizeof(float))
#define SAMPLES_SZ (NX * NY * NZ * sizeof(Cell))
#define CELL_COUNT_SZ (NX * NY * NZ * sizeof(int))

typedef struct {
    double KB;
    double NA;

    // size of domain (m)
    float Lx;
    float Ly;
    float Lz;
    float DL;

    // gas properties
    double molarMass;

    // simulation loop parameters
    float dt;
    int nSteps;
    int printPeriod;
    int particlesPerCellTarget;
    int firstSampleStep;
    int samplingPeriod;

    // stream
    float MaFree;
    float PFree;
    float TFree;
    float angleOfAttack;

    int wingCnt;
    Wing wings[3]; // max 3 wings, but can be changed

    int sphereCnt;
    Sphere spheres[3]; // max 3 spheres, but can be changed

    // VHS model
    double omega;
    double dRef;

    // derived parameters
    double moleculeMass;
    float dx, dy, dz, inv_dx, inv_dy, inv_dz;
    float cellVolume;
    double weight;
    float NFree;
    float UFree;
    float UxFree, UyFree, UzFree;
    int totalCells;

    // Variable Hard Sphere (VHS) model parameters, used in collider
    double sigmaRef;
    double CrRef;
    // No Time Collision Scheme (NTCS) parameters
    double ntcs_collisionProbMultiplier;
    double ntcs_collisionProbMultiplierSquared;
    double ntcs_collidingPairsMultiplier;
    double ntcs_collisionProbExponent;
    // Half-Split-Shuffle (HSS) scheme parameters
    double hss_nbatch;
    double hss_threshold;
    double hss_collisionProbMultiplierSquared;

    // derived, used in simulation
    float generation_derivatedMultiplier;
} Config;

extern __constant__ Config d_conf;

void config_setup(Config *config, int object_case);

#endif //DSMC_CONFIG_H
