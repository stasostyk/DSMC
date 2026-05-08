#ifndef DSMC_PARTICLE_H
#define DSMC_PARTICLE_H

#include <cuda_runtime.h>

// Structure of Arrays (SoA)
typedef struct {
    double *x;
    double *y;
    double *z;
    double *vx;
    double *vy; 
    double *vz;
} ParticleData;

void cudaCopyParticleData(ParticleData *dst, ParticleData *src, cudaMemcpyKind kind);

extern __constant__ ParticleData d_particleData;

#endif // DSMC_PARTICLE_H