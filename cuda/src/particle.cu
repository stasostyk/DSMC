#include <cuda_runtime.h>
#include "../include/particle.h"
#include "../include/config.h"
#include "../include/cuda_utils.h"

__constant__ ParticleData d_particleData;

void cudaCopyParticleData(ParticleData *dst, ParticleData *src, cudaMemcpyKind kind) {
    CHECK(cudaMemcpy(dst->x, src->x, MAX_PARTICLES * sizeof(double), kind));
    CHECK(cudaMemcpy(dst->y, src->y, MAX_PARTICLES * sizeof(double), kind));
    CHECK(cudaMemcpy(dst->z, src->z, MAX_PARTICLES * sizeof(double), kind));

    CHECK(cudaMemcpy(dst->vx, src->vx, MAX_PARTICLES * sizeof(double), kind));
    CHECK(cudaMemcpy(dst->vy, src->vy, MAX_PARTICLES * sizeof(double), kind));
    CHECK(cudaMemcpy(dst->vz, src->vz, MAX_PARTICLES * sizeof(double), kind));
}
