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
#define CELL_LIST_SZ (NX * NY * NZ * MAX_PARTICLES_PER_CELL * sizeof(int))

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

    int ballCnt;
    Ball balls[3]; // max 3 balls, but can be changed

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

    // derived, used in collider
    double sigmaRef;
    double CrRef;
    double ntcs_collisionProbMultiplier; // used in no time collision scheme
    double ntcs_collisionProbMultiplierSquared; // used in no time collision scheme
    double ntcs_collidingPairsMultiplier; // used in no time collision scheme
    double ntcs_collisionProbExponent;    // used in no time collision scheme

    double hss_nbatch; // used in HSS scheme
    double hss_threshold; // used in HSS scheme
    double hss_collisionProbMultiplier; // used in HSS scheme

    // derived, used in simulation
    float generation_derivatedMultiplier;
} Config;

extern __constant__ Config d_conf;

void config_setup(Config *config, int object_case);

#endif //DSMC_CONFIG_H
